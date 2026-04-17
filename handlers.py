"""Telegram command & callback handlers for the Propr trading bot.

Every handler:

* Verifies that ``update.effective_chat.id`` matches ``TELEGRAM_CHAT_ID``.
* Wraps the body in ``try/except`` so a single error cannot crash the bot.
* Uses ``context.bot_data['propr']`` (a :class:`ProprClient`) and
  ``context.bot_data['account_id']`` which are injected in ``bot.py``.

Replies use MarkdownV2; :func:`escape_md` is used on anything user- or
API-derived.
"""

from __future__ import annotations

import asyncio
import logging
import os
import traceback
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from functools import wraps
from typing import Any, Awaitable, Callable, List, Optional, Tuple

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import ContextTypes
from ulid import ULID

from analysis import (
    PENDING_TRADES,
    PendingTrade,
    parse_groq_recommendation,
    run_analysis,
)
from propr import ProprAPIError, ProprClient, max_leverage_for, normalize_asset

log = logging.getLogger(__name__)

TELEGRAM_MAX_LEN = 4000  # safe under Telegram's 4096 hard cap
MARKDOWN_V2_ESCAPE = r"_*[]()~`>#+-=|{}.!\\"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def escape_md(text: Any) -> str:
    """Escape MarkdownV2 special characters in *text*.

    Pass any user-supplied or API-derived value through this before embedding
    it into a reply that uses ``ParseMode.MARKDOWN_V2``.
    """
    s = "" if text is None else str(text)
    out = []
    for ch in s:
        if ch in MARKDOWN_V2_ESCAPE:
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def fmt_num(value: Any, places: int = 2) -> str:
    """Format *value* as a decimal with up to *places* digits after the point.

    Trailing zeros are stripped. Non-numeric input is returned as-is.
    """
    if value is None or value == "":
        return "-"
    try:
        d = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return str(value)
    quantizer = Decimal("1").scaleb(-places) if places > 0 else Decimal("1")
    try:
        d = d.quantize(quantizer)
    except InvalidOperation:
        pass
    text = format(d, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _split_message(text: str, limit: int = TELEGRAM_MAX_LEN) -> List[str]:
    """Split *text* into chunks ≤ *limit* on line boundaries."""
    if len(text) <= limit:
        return [text]
    lines = text.split("\n")
    chunks: List[str] = []
    current = ""
    for line in lines:
        if len(line) > limit:
            # Single giant line — hard-split it.
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(line), limit):
                chunks.append(line[i:i + limit])
            continue
        candidate = line if not current else current + "\n" + line
        if len(candidate) > limit:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Auth decorator
# ---------------------------------------------------------------------------
HandlerFn = Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]


def authorized(func: HandlerFn) -> HandlerFn:
    """Restrict the handler to the configured Telegram chat id."""

    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        raw = os.getenv("TELEGRAM_CHAT_ID", "")
        try:
            allowed_id = int(raw)
        except ValueError:
            log.error("TELEGRAM_CHAT_ID is not a valid integer: %r", raw)
            if chat:
                await context.bot.send_message(chat_id=chat.id, text="⛔ Unauthorized")
            return
        if chat is None or chat.id != allowed_id:
            if chat:
                await context.bot.send_message(chat_id=chat.id, text="⛔ Unauthorized")
            log.warning("Unauthorized chat %s blocked from %s", chat.id if chat else "?", func.__name__)
            return

        user = update.effective_user
        args = getattr(context, "args", None)
        log.info(
            "handler=%s chat=%s user=%s args=%s",
            func.__name__,
            chat.id,
            user.id if user else "?",
            args,
        )
        return await func(update, context)

    return wrapper


# ---------------------------------------------------------------------------
# Argument / error helpers
# ---------------------------------------------------------------------------
def parse_args(
    context: ContextTypes.DEFAULT_TYPE, min_args: int, max_args: int
) -> Optional[List[str]]:
    """Return ``context.args`` if its length is in ``[min_args, max_args]``.

    Returns ``None`` when the count is out of range; the caller should then
    reply with a usage message.
    """
    args = list(context.args or [])
    if min_args <= len(args) <= max_args:
        return args
    return None


async def _reply(update: Update, text: str, **kwargs: Any) -> Any:
    """Send a MarkdownV2 reply to the current chat."""
    msg = update.effective_message
    if msg is None:
        return None
    return await msg.reply_text(
        text, parse_mode=ParseMode.MARKDOWN_V2, **kwargs
    )


async def _report_error(update: Update, exc: Exception) -> None:
    """Translate *exc* into a friendly MarkdownV2 message and log it."""
    if isinstance(exc, ProprAPIError):
        text = f"❌ API error: {escape_md(exc)}"
    else:
        text = f"❌ Unexpected error: {escape_md(type(exc).__name__)}: {escape_md(exc)}"
    log.error("handler error: %s\n%s", exc, traceback.format_exc())
    try:
        await _reply(update, text)
    except Exception as reply_exc:  # noqa: BLE001
        log.error("failed to send error reply: %s", reply_exc)


def _get_client(context: ContextTypes.DEFAULT_TYPE) -> ProprClient:
    client = context.bot_data.get("propr")
    if client is None:
        raise RuntimeError("ProprClient not initialized in bot_data")
    return client


def _get_account_id(context: ContextTypes.DEFAULT_TYPE) -> Optional[str]:
    return context.bot_data.get("account_id")


async def _ensure_account_id(context: ContextTypes.DEFAULT_TYPE) -> str:
    """Return the cached accountId or fetch and cache it."""
    existing = _get_account_id(context)
    if existing:
        return existing
    client = _get_client(context)
    account_id = await client.get_active_account_id()
    context.bot_data["account_id"] = account_id
    return account_id


def _parse_qty(raw: str) -> Decimal:
    """Parse a quantity string into Decimal, raising ``ValueError`` on junk."""
    try:
        d = Decimal(raw.replace(",", "").replace("_", ""))
    except (InvalidOperation, ValueError):
        raise ValueError(f"invalid quantity: {raw!r}")
    if d <= 0:
        raise ValueError("quantity must be positive")
    return d


def _parse_price(raw: str) -> Decimal:
    """Parse a price string into Decimal."""
    try:
        d = Decimal(raw.replace(",", "").replace("_", ""))
    except (InvalidOperation, ValueError):
        raise ValueError(f"invalid price: {raw!r}")
    if d <= 0:
        raise ValueError("price must be positive")
    return d


def _pos_asset_upper(p: dict) -> str:
    return str(p.get("asset", "")).split(":")[-1].upper()


def _pos_side(p: dict) -> str:
    return str(p.get("positionSide") or p.get("side") or "").lower()


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------
@authorized
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/start`` — welcome, health check, basic account balance."""
    try:
        client = _get_client(context)
        health = await client.health()
        me = await client.me()
        account_id = await _ensure_account_id(context)

        balance = "?"
        if isinstance(me, dict):
            balance = (
                me.get("balance")
                or me.get("equity")
                or (me.get("account") or {}).get("balance")
                or "?"
            )
        if balance == "?":
            # fall back to positions endpoint account summary
            try:
                positions = await client.get_positions(account_id)
                if positions and isinstance(positions[0].get("accountBalance"), (int, float, str)):
                    balance = positions[0]["accountBalance"]
            except ProprAPIError:
                pass

        status = health.get("status") if isinstance(health, dict) else "ok"
        msg = (
            "👋 *Welcome to Propr Bot*\n\n"
            f"API health: `{escape_md(status)}`\n"
            f"Account: `{escape_md(account_id)}`\n"
            f"Balance: *{escape_md(fmt_num(balance))}* USDC\n\n"
            "Type /help for the full command list\\."
        )
        await _reply(update, msg)
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


@authorized
async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/status`` — API health, challenge status, P&L summary."""
    try:
        client = _get_client(context)
        account_id = await _ensure_account_id(context)

        health, attempts, positions, trades = await asyncio.gather(
            client.health(),
            client.get_challenge_attempts(),
            client.get_positions(account_id),
            client.get_trades(account_id, limit=50),
            return_exceptions=True,
        )

        def _ok(v: Any) -> Any:
            return v if not isinstance(v, Exception) else None

        health = _ok(health) or {}
        attempts = _ok(attempts) or []
        positions = _ok(positions) or []
        trades = _ok(trades) or []

        attempt = attempts[0] if attempts else {}
        status_val = health.get("status", "unknown") if isinstance(health, dict) else "unknown"
        attempt_status = attempt.get("status", "?") if isinstance(attempt, dict) else "?"

        realized = sum((Decimal(str(t.get("pnl", 0) or 0)) for t in trades), Decimal(0))
        unrealized = sum(
            (
                Decimal(str(p.get("unrealizedPnl", p.get("uPnl", 0) or 0)))
                for p in positions
            ),
            Decimal(0),
        )

        text = (
            "📊 *Status*\n\n"
            f"API: `{escape_md(status_val)}`\n"
            f"Challenge: `{escape_md(attempt_status)}`\n"
            f"Account: `{escape_md(account_id)}`\n"
            f"Realized PnL \\(recent\\): *{escape_md(fmt_num(realized))}* USDC\n"
            f"Unrealized PnL: *{escape_md(fmt_num(unrealized))}* USDC\n"
            f"Open positions: `{len(positions)}`"
        )
        await _reply(update, text)
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


@authorized
async def positions_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/positions`` — list open positions."""
    try:
        client = _get_client(context)
        account_id = await _ensure_account_id(context)
        positions = await client.get_positions(account_id)
        if not positions:
            await _reply(update, "📂 No open positions\\.")
            return

        lines = ["📈 *Open Positions*"]
        for p in positions:
            asset = p.get("asset", "?")
            side = _pos_side(p) or "?"
            qty = p.get("quantity", p.get("qty", "?"))
            entry = p.get("entryPrice", p.get("avgEntryPrice", "?"))
            mark = p.get("markPrice", p.get("mark", "?"))
            upnl = p.get("unrealizedPnl", p.get("uPnl", "?"))
            lev = p.get("leverage", "?")
            lines.append(
                f"• `{escape_md(asset)}` "
                f"{escape_md(side)} "
                f"`{escape_md(fmt_num(qty, 6))}`"
                f" \\| entry *{escape_md(fmt_num(entry))}*"
                f" \\| mark `{escape_md(fmt_num(mark))}`"
                f" \\| uPnL *{escape_md(fmt_num(upnl))}*"
                f" \\| `{escape_md(lev)}x`"
            )
        await _reply(update, "\n".join(lines))
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


@authorized
async def orders_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/orders`` — list open orders."""
    try:
        client = _get_client(context)
        account_id = await _ensure_account_id(context)
        orders = await client.get_open_orders(account_id)
        if not orders:
            await _reply(update, "📭 No open orders\\.")
            return

        lines = ["📝 *Open Orders*"]
        for o in orders:
            oid = o.get("id", o.get("orderId", "?"))
            side = o.get("side", "?")
            otype = o.get("type", "?")
            asset = o.get("asset", "?")
            qty = o.get("quantity", o.get("qty", "?"))
            price = o.get("price", o.get("triggerPrice", "?"))
            lines.append(
                f"• `{escape_md(oid)}` "
                f"{escape_md(otype)} {escape_md(side)} "
                f"`{escape_md(fmt_num(qty, 6))}` {escape_md(asset)} "
                f"@ *{escape_md(fmt_num(price))}*"
            )
        await _reply(update, "\n".join(lines))
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


@authorized
async def trades_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/trades [n]`` — last *n* trades (default 5)."""
    try:
        args = parse_args(context, 0, 1)
        if args is None:
            await _reply(update, "Usage: `/trades [n]`")
            return
        limit = 5
        if args:
            try:
                limit = max(1, min(50, int(args[0])))
            except ValueError:
                await _reply(update, "⚠️ n must be an integer")
                return

        client = _get_client(context)
        account_id = await _ensure_account_id(context)
        trades = await client.get_trades(account_id, limit=limit)
        if not trades:
            await _reply(update, "📂 No trades yet\\.")
            return

        lines = [f"🧾 *Last {limit} Trades*"]
        for t in trades:
            ts = t.get("createdAt") or t.get("timestamp") or t.get("time") or ""
            asset = t.get("asset", "?")
            side = t.get("side", "?")
            qty = t.get("quantity", t.get("qty", "?"))
            price = t.get("price", t.get("avgPrice", "?"))
            pnl = t.get("pnl", "?")
            lines.append(
                f"• `{escape_md(ts)}` "
                f"{escape_md(side)} `{escape_md(fmt_num(qty, 6))}` "
                f"{escape_md(asset)} @ *{escape_md(fmt_num(price))}* "
                f"\\| PnL {escape_md(fmt_num(pnl))}"
            )
        await _reply(update, "\n".join(lines))
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


@authorized
async def buy_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/buy <asset> <qty> [price]`` — market if no price, limit otherwise."""
    try:
        args = parse_args(context, 2, 3)
        if args is None:
            await _reply(update, "Usage: `/buy <asset> <qty> [price]`")
            return
        asset = args[0].upper()
        qty = _parse_qty(args[1])
        price: Optional[Decimal] = None
        if len(args) == 3:
            price = _parse_price(args[2])

        client = _get_client(context)
        account_id = await _ensure_account_id(context)
        if price is None:
            result = await client.market_buy(account_id, asset, qty)
            mode = "market"
        else:
            result = await client.limit_buy(account_id, asset, qty, price)
            mode = f"limit @ {fmt_num(price)}"

        await _reply(
            update,
            f"🟢 *Buy* `{escape_md(fmt_num(qty, 6))}` {escape_md(asset)} "
            f"\\({escape_md(mode)}\\) submitted\\.",
        )
        log.debug("buy result: %s", result)
    except ValueError as exc:
        await _reply(update, f"⚠️ {escape_md(exc)}")
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


@authorized
async def sell_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/sell <asset> <qty> [price]`` — always reduce-only."""
    try:
        args = parse_args(context, 2, 3)
        if args is None:
            await _reply(update, "Usage: `/sell <asset> <qty> [price]`")
            return
        asset = args[0].upper()
        qty = _parse_qty(args[1])
        price: Optional[Decimal] = None
        if len(args) == 3:
            price = _parse_price(args[2])

        client = _get_client(context)
        account_id = await _ensure_account_id(context)
        if price is None:
            await client.market_sell(account_id, asset, qty, reduce_only=True)
            mode = "market"
        else:
            await client.limit_sell(account_id, asset, qty, price, reduce_only=True)
            mode = f"limit @ {fmt_num(price)}"

        await _reply(
            update,
            f"🔴 *Sell* `{escape_md(fmt_num(qty, 6))}` {escape_md(asset)} "
            f"\\({escape_md(mode)}, reduce\\-only\\) submitted\\.",
        )
    except ValueError as exc:
        await _reply(update, f"⚠️ {escape_md(exc)}")
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


@authorized
async def close_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/close <asset>`` — fully close current position on *asset*."""
    try:
        args = parse_args(context, 1, 1)
        if args is None:
            await _reply(update, "Usage: `/close <asset>`")
            return
        asset = args[0].upper()

        client = _get_client(context)
        account_id = await _ensure_account_id(context)
        positions = await client.get_positions(account_id)
        target = next((p for p in positions if _pos_asset_upper(p) == asset), None)
        if target is None:
            await _reply(update, f"⚠️ No open position on `{escape_md(asset)}`")
            return
        side = _pos_side(target)
        if side not in ("long", "short"):
            await _reply(update, "⚠️ Could not determine position side\\.")
            return

        await client.close_position(account_id, asset, side)
        await _reply(
            update,
            f"🧹 Closing *{escape_md(side.upper())}* position on {escape_md(asset)}\\.",
        )
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


@authorized
async def cancel_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/cancel <orderId>`` — cancel a single order."""
    try:
        args = parse_args(context, 1, 1)
        if args is None:
            await _reply(update, "Usage: `/cancel <orderId>`")
            return
        order_id = args[0]
        client = _get_client(context)
        account_id = await _ensure_account_id(context)
        result = await client.cancel_order(account_id, order_id)
        if isinstance(result, dict) and result.get("status") == "already_done":
            await _reply(update, f"ℹ️ Order `{escape_md(order_id)}` already filled or cancelled\\.")
        else:
            await _reply(update, f"✅ Cancelled `{escape_md(order_id)}`")
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


@authorized
async def cancelall_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/cancelall`` — cancel every open order, tolerating 400s."""
    try:
        client = _get_client(context)
        account_id = await _ensure_account_id(context)
        orders = await client.get_open_orders(account_id)
        if not orders:
            await _reply(update, "📭 No open orders\\.")
            return

        results = await asyncio.gather(
            *[
                client.cancel_order(
                    account_id, str(o.get("id") or o.get("orderId") or "")
                )
                for o in orders
                if (o.get("id") or o.get("orderId"))
            ],
            return_exceptions=True,
        )
        ok = sum(
            1
            for r in results
            if not isinstance(r, Exception)
        )
        failed = sum(1 for r in results if isinstance(r, Exception))
        await _reply(
            update,
            f"🧹 Cancel\\-all: `{ok}` ok, `{failed}` failed",
        )
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


@authorized
async def sl_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/sl <asset> <triggerPrice>`` — stop-loss on the current long."""
    try:
        args = parse_args(context, 2, 2)
        if args is None:
            await _reply(update, "Usage: `/sl <asset> <triggerPrice>`")
            return
        asset = args[0].upper()
        trigger = _parse_price(args[1])

        client = _get_client(context)
        account_id = await _ensure_account_id(context)
        positions = await client.get_positions(account_id)
        target = next(
            (
                p
                for p in positions
                if _pos_asset_upper(p) == asset and _pos_side(p) == "long"
            ),
            None,
        )
        if target is None:
            await _reply(update, f"⚠️ No open long position on `{escape_md(asset)}`")
            return
        qty = target.get("quantity", target.get("qty"))
        await client.stop_loss(account_id, asset, qty, trigger, "long")
        await _reply(
            update,
            f"🛡 Stop\\-loss set on {escape_md(asset)} at *{escape_md(fmt_num(trigger))}*",
        )
    except ValueError as exc:
        await _reply(update, f"⚠️ {escape_md(exc)}")
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


@authorized
async def tp_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/tp <asset> <triggerPrice>`` — take-profit on the current long."""
    try:
        args = parse_args(context, 2, 2)
        if args is None:
            await _reply(update, "Usage: `/tp <asset> <triggerPrice>`")
            return
        asset = args[0].upper()
        trigger = _parse_price(args[1])

        client = _get_client(context)
        account_id = await _ensure_account_id(context)
        positions = await client.get_positions(account_id)
        target = next(
            (
                p
                for p in positions
                if _pos_asset_upper(p) == asset and _pos_side(p) == "long"
            ),
            None,
        )
        if target is None:
            await _reply(update, f"⚠️ No open long position on `{escape_md(asset)}`")
            return
        qty = target.get("quantity", target.get("qty"))
        await client.take_profit(account_id, asset, qty, trigger, "long")
        await _reply(
            update,
            f"🎯 Take\\-profit set on {escape_md(asset)} at *{escape_md(fmt_num(trigger))}*",
        )
    except ValueError as exc:
        await _reply(update, f"⚠️ {escape_md(exc)}")
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


@authorized
async def leverage_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/leverage <asset> <n>`` — change leverage subject to caps."""
    try:
        args = parse_args(context, 2, 2)
        if args is None:
            await _reply(update, "Usage: `/leverage <asset> <n>`")
            return
        asset = args[0].upper()
        try:
            lev = int(args[1])
        except ValueError:
            await _reply(update, "⚠️ Leverage must be an integer")
            return
        cap = max_leverage_for(asset)
        if lev > cap or lev <= 0:
            await _reply(
                update,
                f"⚠️ Leverage for {escape_md(asset)} must be 1\\-{cap}x",
            )
            return

        client = _get_client(context)
        account_id = await _ensure_account_id(context)
        await client.set_leverage(account_id, asset, lev)
        await _reply(
            update,
            f"⚙️ Leverage for {escape_md(asset)} set to *{lev}x*",
        )
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)


@authorized
async def analysis_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/analysis <asset> [timeframe] [direction]`` — expert AI analysis."""
    typing_task: Optional[asyncio.Task] = None
    try:
        args = parse_args(context, 1, 3)
        if args is None:
            await _reply(update, "Usage: `/analysis <asset> [timeframe] [direction]`")
            return

        asset = args[0].upper()
        timeframe = args[1].lower() if len(args) >= 2 else "1h"
        direction = args[2].lower() if len(args) >= 3 else None

        chat_id = update.effective_chat.id

        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        await _reply(update, f"🔍 Analyzing {escape_md(asset)}\\.\\.\\.")

        async def _typing_loop() -> None:
            try:
                while True:
                    await asyncio.sleep(4)
                    await context.bot.send_chat_action(
                        chat_id=chat_id, action=ChatAction.TYPING
                    )
            except asyncio.CancelledError:
                return

        typing_task = asyncio.create_task(_typing_loop())

        client = _get_client(context)
        account_id = await _ensure_account_id(context)
        result = await run_analysis(
            client, account_id, asset, timeframe=timeframe, direction=direction
        )

        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass
        typing_task = None

        raw = result["raw"]
        parsed = result["parsed"]
        fetched_at = result["market_fetched_at"]

        pending_id = str(ULID())
        footer = f"\n\n📡 Market data fetched at {fetched_at}"
        full_text = raw + footer
        chunks = _split_message(full_text, TELEGRAM_MAX_LEN)

        keyboard = InlineKeyboardMarkup(
            [[
                InlineKeyboardButton(
                    "✅ Execute Trade", callback_data=f"exec_trade:{pending_id}"
                ),
                InlineKeyboardButton(
                    "❌ Skip", callback_data=f"skip_trade:{pending_id}"
                ),
            ]]
        )

        last_message_id: Optional[int] = None
        for idx, chunk in enumerate(chunks):
            is_last = idx == len(chunks) - 1
            escaped = escape_md(chunk)
            sent = await context.bot.send_message(
                chat_id=chat_id,
                text=escaped,
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=keyboard if is_last and parsed.get("executable") else None,
            )
            last_message_id = sent.message_id

        PENDING_TRADES[pending_id] = PendingTrade(
            pending_id=pending_id,
            chat_id=chat_id,
            asset=asset,
            timeframe=timeframe,
            raw_analysis=raw,
            parsed=parsed,
            created_at=datetime.now(timezone.utc),
            message_id=last_message_id,
        )

        if not parsed.get("executable"):
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    "ℹ️ This analysis is not auto\\-executable "
                    "\\(missing direction, size, SL, or TP1\\)\\. "
                    "Review and place manually with /buy, /sell, /sl, /tp\\."
                ),
                parse_mode=ParseMode.MARKDOWN_V2,
            )
    except Exception as exc:  # noqa: BLE001
        await _report_error(update, exc)
    finally:
        if typing_task and not typing_task.done():
            typing_task.cancel()


# ---------------------------------------------------------------------------
# Callback handler for Execute / Skip buttons
# ---------------------------------------------------------------------------
def _callback_authorized(query_chat_id: int) -> bool:
    raw = os.getenv("TELEGRAM_CHAT_ID", "")
    try:
        return query_chat_id == int(raw)
    except ValueError:
        return False


async def callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle ``exec_trade`` and ``skip_trade`` inline-button clicks."""
    query = update.callback_query
    if query is None:
        return

    chat = update.effective_chat
    if chat is None or not _callback_authorized(chat.id):
        await query.answer("⛔ Unauthorized", show_alert=False)
        return

    data = query.data or ""
    try:
        action, _, pending_id = data.partition(":")
    except ValueError:
        await query.answer("Invalid callback")
        return

    pending = PENDING_TRADES.get(pending_id)
    if pending is None:
        await query.answer("Expired")
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception as exc:  # noqa: BLE001
            log.debug("edit_message_reply_markup on expired failed: %s", exc)
        return

    if action == "skip_trade":
        PENDING_TRADES.pop(pending_id, None)
        await query.answer("Skipped")
        try:
            await query.edit_message_reply_markup(reply_markup=None)
            await context.bot.send_message(
                chat_id=chat.id, text="❎ Trade skipped."
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("skip cleanup failed: %s", exc)
        return

    if action != "exec_trade":
        await query.answer("Unknown action")
        return

    await query.answer("Executing…")
    try:
        await query.edit_message_reply_markup(reply_markup=None)
    except Exception as exc:  # noqa: BLE001
        log.debug("edit_message_reply_markup failed: %s", exc)

    PENDING_TRADES.pop(pending_id, None)

    # Re-parse from raw text in case the stored dict is stale — spec requires it.
    parsed = parse_groq_recommendation(pending.raw_analysis) or pending.parsed

    if parsed.get("direction") == "no_trade" or not parsed.get("executable"):
        await context.bot.send_message(
            chat_id=chat.id,
            text=(
                "⚠️ Could not auto-parse trade parameters. "
                "Review analysis and place manually with /buy or /sell."
            ),
        )
        return

    await _execute_parsed_trade(context, chat.id, pending.asset, parsed)


async def _execute_parsed_trade(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    asset: str,
    parsed: dict,
) -> None:
    """Turn a parsed Groq recommendation into live Propr orders."""
    client = _get_client(context)
    try:
        account_id = await _ensure_account_id(context)
    except Exception as exc:  # noqa: BLE001
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❌ Execution failed at step account lookup: {exc}",
        )
        return

    direction: str = parsed["direction"]
    quantity: Decimal = parsed["quantity"]
    stop_loss: Decimal = parsed["stop_loss"]
    tp1: Decimal = parsed["take_profit_1"]
    tp2: Optional[Decimal] = parsed.get("take_profit_2")
    entry_price: Optional[Decimal] = parsed.get("entry_price")
    leverage: Optional[int] = parsed.get("leverage")
    order_type: str = parsed.get("order_type") or "market"

    position_side = "long" if direction == "long" else "short"
    taker_side = "buy" if position_side == "long" else "sell"
    opposite_side = "sell" if taker_side == "buy" else "buy"

    # Step 1: leverage
    if leverage is not None and leverage > 0:
        try:
            current_cfg = await client.get_margin_config(account_id, asset)
            if isinstance(current_cfg, dict) and "data" in current_cfg and isinstance(current_cfg["data"], dict):
                current_cfg = current_cfg["data"]
            current_lev = None
            if isinstance(current_cfg, dict):
                current_lev = current_cfg.get("leverage")
            try:
                current_lev_int = int(float(current_lev)) if current_lev is not None else None
            except (TypeError, ValueError):
                current_lev_int = None
            if current_lev_int != leverage:
                await client.set_leverage(account_id, asset, leverage)
        except Exception as exc:  # noqa: BLE001
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"❌ Execution failed at step leverage: {exc}",
            )
            return

    # Step 2: entry
    try:
        if order_type == "limit" and entry_price is not None:
            await client.place_order(
                account_id=account_id,
                asset=asset,
                type="limit",
                side=taker_side,
                positionSide=position_side,
                timeInForce="GTC",
                quantity=quantity,
                price=entry_price,
                reduceOnly=False,
            )
            entry_desc = f"limit @ {fmt_num(entry_price)}"
        else:
            await client.place_order(
                account_id=account_id,
                asset=asset,
                type="market",
                side=taker_side,
                positionSide=position_side,
                timeInForce="IOC",
                quantity=quantity,
                reduceOnly=False,
            )
            entry_desc = "market"
    except Exception as exc:  # noqa: BLE001
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❌ Execution failed at step entry: {exc}",
        )
        return

    # Step 3: stop loss
    try:
        await client.place_order(
            account_id=account_id,
            asset=asset,
            type="stop_market",
            side=opposite_side,
            positionSide=position_side,
            timeInForce="GTC",
            quantity=quantity,
            triggerPrice=stop_loss,
            reduceOnly=True,
        )
    except Exception as exc:  # noqa: BLE001
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❌ Execution failed at step stop_loss: {exc}",
        )
        return

    # Step 4: take profit(s)
    tp_msgs = []
    try:
        if tp2 is not None:
            half_qty = _half_qty(quantity)
            await client.place_order(
                account_id=account_id,
                asset=asset,
                type="take_profit_market",
                side=opposite_side,
                positionSide=position_side,
                timeInForce="GTC",
                quantity=half_qty,
                triggerPrice=tp1,
                reduceOnly=True,
            )
            await client.place_order(
                account_id=account_id,
                asset=asset,
                type="take_profit_market",
                side=opposite_side,
                positionSide=position_side,
                timeInForce="GTC",
                quantity=quantity - half_qty,
                triggerPrice=tp2,
                reduceOnly=True,
            )
            tp_msgs = [
                f"TP1: {fmt_num(tp1)}",
                f"TP2: {fmt_num(tp2)}",
            ]
        else:
            await client.place_order(
                account_id=account_id,
                asset=asset,
                type="take_profit_market",
                side=opposite_side,
                positionSide=position_side,
                timeInForce="GTC",
                quantity=quantity,
                triggerPrice=tp1,
                reduceOnly=True,
            )
            tp_msgs = [f"TP1: {fmt_num(tp1)}"]
    except Exception as exc:  # noqa: BLE001
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❌ Execution failed at step take_profit: {exc}",
        )
        return

    direction_upper = direction.upper()
    summary_parts = [
        f"{direction_upper} {fmt_num(quantity, 6)} {asset} @ {entry_desc}",
        f"SL: {fmt_num(stop_loss)}",
    ]
    summary_parts.extend(tp_msgs)
    summary = " | ".join(summary_parts)
    await context.bot.send_message(chat_id=chat_id, text=f"🚀 Trade executed: {summary}")


def _half_qty(qty: Decimal) -> Decimal:
    """Split *qty* in half, rounded down to a reasonable precision."""
    half = qty / Decimal(2)
    # Preserve up to 8 decimal places to handle small-lot crypto sizes.
    return half.quantize(Decimal("0.00000001"))


@authorized
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/help`` — list every command with a short example."""
    text = (
        "*Propr Bot Commands*\n\n"
        "• `/start` — welcome \\+ health check\n"
        "• `/status` — API \\+ challenge \\+ P&L\n"
        "• `/positions` — open positions\n"
        "• `/orders` — open orders\n"
        "• `/trades [n]` — last n trades \\(default 5\\)\n"
        "• `/buy <asset> <qty> [price]` — e\\.g\\. `/buy BTC 0.001`\n"
        "• `/sell <asset> <qty> [price]` — reduce\\-only\n"
        "• `/close <asset>` — close a position\n"
        "• `/cancel <orderId>` — cancel one order\n"
        "• `/cancelall` — cancel every open order\n"
        "• `/sl <asset> <trigger>` — stop\\-loss on long\n"
        "• `/tp <asset> <trigger>` — take\\-profit on long\n"
        "• `/leverage <asset> <n>` — set leverage\n"
        "• `/analysis <asset> [tf] [dir]` — AI trade plan\n"
        "• `/help` — this message"
    )
    await _reply(update, text)


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log uncaught errors and notify the authorized chat."""
    log.exception("Unhandled error: %s", context.error)
    raw = os.getenv("TELEGRAM_CHAT_ID", "")
    try:
        chat_id = int(raw)
    except ValueError:
        return
    try:
        await context.bot.send_message(
            chat_id=chat_id, text="❌ Internal error — check logs"
        )
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to notify chat of error: %s", exc)


# ---------------------------------------------------------------------------
# Handler registry (consumed by bot.py)
# ---------------------------------------------------------------------------
COMMAND_HANDLERS: List[Tuple[str, HandlerFn]] = [
    ("start", start_cmd),
    ("status", status_cmd),
    ("positions", positions_cmd),
    ("orders", orders_cmd),
    ("trades", trades_cmd),
    ("buy", buy_cmd),
    ("sell", sell_cmd),
    ("close", close_cmd),
    ("cancel", cancel_cmd),
    ("cancelall", cancelall_cmd),
    ("sl", sl_cmd),
    ("tp", tp_cmd),
    ("leverage", leverage_cmd),
    ("analysis", analysis_cmd),
    ("help", help_cmd),
]
