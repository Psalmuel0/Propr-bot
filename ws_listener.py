"""Propr websocket listener — pushes fill/position events to Telegram.

Connects to ``wss://api.propr.xyz/ws`` with the ``X-API-Key`` header, parses
each incoming JSON frame, maps known events onto friendly Telegram messages,
and auto-reconnects on any connection error. A per-message try/except keeps
a handler bug from killing the loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

log = logging.getLogger(__name__)

WS_URL = "wss://api.propr.xyz/ws"
RECONNECT_DELAY = 5  # seconds

# Event-name → emoji + template. Keys are lower-cased, punctuation-normalized.
_EVENT_TEMPLATES: Dict[str, str] = {
    "order.filled": "✅ Filled: {side} {qty} {asset} @ {price}",
    "order.partially_filled": "⏳ Partial fill: {filled_qty}/{total_qty} {asset}",
    "order.cancelled": "❌ Order cancelled",
    "position.opened": "📈 Opened: {side} {qty} {asset} @ {entry}",
    "position.closed": "📊 Closed | PnL: {pnl} USDC",
    "position.liquidated": "🚨 LIQUIDATED: {asset} | Loss: {pnl} USDC",
    "position.take_profit.hit": "🎯 TP hit | PnL: {pnl} USDC",
    "position.stop_loss.hit": "🛑 SL hit | PnL: {pnl} USDC",
}


def _canonical_event(name: str) -> str:
    """Normalize event names so camelCase/dotted/snake variants all match.

    ``Position.TakeProfit.Hit`` → ``position.take_profit.hit``.
    """
    if not name:
        return ""
    lowered = []
    for ch in name:
        if ch.isupper():
            if lowered and lowered[-1] not in (".", "_", ""):
                lowered.append("_")
            lowered.append(ch.lower())
        else:
            lowered.append(ch)
    return "".join(lowered)


def _extract_payload(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Return the event payload — some APIs wrap it in ``data`` or ``payload``."""
    for key in ("data", "payload", "event_data", "body"):
        value = msg.get(key)
        if isinstance(value, dict):
            return {**msg, **value}
    return msg


def _format_event(event: str, payload: Dict[str, Any]) -> Optional[str]:
    """Render *event* using :data:`_EVENT_TEMPLATES`; return ``None`` if unknown."""
    template = _EVENT_TEMPLATES.get(event)
    if template is None:
        return None

    # Provide sensible defaults for missing fields so template formatting
    # cannot crash.
    values = {
        "side": payload.get("side") or payload.get("positionSide") or "?",
        "qty": payload.get("quantity") or payload.get("qty") or "?",
        "filled_qty": payload.get("filledQty")
        or payload.get("filled_quantity")
        or payload.get("filled")
        or "?",
        "total_qty": payload.get("totalQty")
        or payload.get("total_quantity")
        or payload.get("quantity")
        or "?",
        "asset": payload.get("asset") or payload.get("symbol") or "?",
        "price": payload.get("price")
        or payload.get("avgPrice")
        or payload.get("fillPrice")
        or "?",
        "entry": payload.get("entryPrice")
        or payload.get("avgEntryPrice")
        or payload.get("price")
        or "?",
        "pnl": payload.get("pnl")
        or payload.get("realizedPnl")
        or payload.get("unrealizedPnl")
        or "?",
    }
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError) as exc:
        log.debug("format error for %s: %s", event, exc)
        return None


async def _handle_message(bot: Any, chat_id: int, raw: Any) -> None:
    """Parse a single websocket frame and dispatch to Telegram if relevant."""
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            log.debug("non-utf8 frame: %r", raw[:64])
            return

    try:
        msg = json.loads(raw)
    except (TypeError, ValueError):
        log.debug("non-json frame: %r", str(raw)[:200])
        return

    if not isinstance(msg, dict):
        return

    event_raw = msg.get("event") or msg.get("type") or msg.get("eventType") or ""
    event = _canonical_event(str(event_raw))
    if not event:
        log.debug("frame without event/type: %s", msg)
        return

    payload = _extract_payload(msg)
    text = _format_event(event, payload)
    if text is None:
        log.debug("ignoring unknown event: %s", event)
        return

    try:
        await bot.send_message(chat_id=chat_id, text=text)
    except Exception as exc:  # noqa: BLE001 — network / rate-limit errors
        log.error("Failed to send ws event to telegram: %s", exc)


async def run_ws_listener(
    bot: Any, chat_id: int, api_key: str, account_id: str
) -> None:
    """Connect to the Propr websocket and forward events to Telegram.

    Reconnects indefinitely with a 5-second backoff. The ``account_id`` is
    currently unused by the transport layer but kept in the signature so that
    callers can pass it if/when the API requires a subscribe frame.
    """
    headers = {"X-API-Key": api_key}

    while True:
        try:
            log.info("Connecting to Propr websocket %s for account %s",
                     WS_URL, account_id)
            # websockets 12.0 uses ``extra_headers``.
            async with websockets.connect(
                WS_URL,
                extra_headers=headers,
                ping_interval=30,
                ping_timeout=20,
            ) as ws:
                log.info("Propr websocket connected")
                # Some deployments require a subscribe frame; the public docs
                # don't specify one, so we just listen. If the server closes
                # for lack of a subscription, add your subscribe message here.
                async for message in ws:
                    try:
                        await _handle_message(bot, chat_id, message)
                    except Exception as exc:  # noqa: BLE001 — never die on a bad frame
                        log.exception("ws message handler error: %s", exc)
        except asyncio.CancelledError:
            log.info("ws listener cancelled")
            raise
        except (ConnectionClosed, WebSocketException, OSError) as exc:
            log.warning("ws disconnected (%s); reconnecting in %ss",
                        exc, RECONNECT_DELAY)
        except Exception as exc:  # noqa: BLE001
            log.exception("ws loop unexpected error: %s", exc)

        try:
            await asyncio.sleep(RECONNECT_DELAY)
        except asyncio.CancelledError:
            raise
