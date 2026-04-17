"""Propr trading bot — Telegram entry point.

Responsibilities:

* Load environment, validate required vars.
* Health-check Propr, Binance and Fear/Greed on startup.
* Build the ``Application``, register every command + the callback handler.
* Start the websocket listener and the pending-trade sweeper as background
  asyncio tasks via ``post_init``, and cancel them via ``post_shutdown``.
* Run polling.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import List

import httpx
from dotenv import load_dotenv
from telegram.ext import Application, CallbackQueryHandler, CommandHandler

import handlers
from analysis import sweep_pending_trades
from propr import ProprClient
from ws_listener import run_ws_listener

log = logging.getLogger("propr_bot")

REQUIRED_ENV = ("PROPR_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "GROQ_API_KEY")


def _validate_env() -> None:
    """Abort with a clear ``SystemExit`` if any required env var is missing."""
    missing = [name for name in REQUIRED_ENV if not os.getenv(name)]
    if missing:
        raise SystemExit(
            "Missing required env vars: " + ", ".join(missing)
            + ". Copy .env.example to .env and fill it in."
        )
    try:
        int(os.environ["TELEGRAM_CHAT_ID"])
    except ValueError as exc:
        raise SystemExit("TELEGRAM_CHAT_ID must be an integer") from exc


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Quiet down noisy libraries.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.INFO)


async def _ping_market_endpoints() -> None:
    """Warn — but don't abort — if Binance or Fear/Greed are unreachable."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        for name, url in (
            ("Binance", "https://api.binance.com/api/v3/ping"),
            ("FearGreed", "https://api.alternative.me/fng/?limit=1"),
        ):
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                log.info("%s reachable", name)
            except Exception as exc:  # noqa: BLE001
                log.warning("%s unreachable: %s", name, exc)


async def _startup_checks(client: ProprClient) -> str:
    """Run the on-startup health/auth/account validations."""
    log.info("Running startup checks")

    health = await client.health()
    log.info("Propr health: %s", health)

    me = await client.me()
    user_id = (
        me.get("id")
        or me.get("userId")
        or me.get("email")
        or "unknown"
        if isinstance(me, dict)
        else "unknown"
    )
    log.info("Authenticated as user=%s", user_id)

    account_id = await client.get_active_account_id()
    log.info("Active accountId=%s", account_id)

    await _ping_market_endpoints()

    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY empty after validation")
    log.info("Groq key present")

    log.info("Bot ready ✅")
    return account_id


def _register_handlers(app: Application) -> None:
    for name, func in handlers.COMMAND_HANDLERS:
        app.add_handler(CommandHandler(name, func))
    app.add_handler(CallbackQueryHandler(handlers.callback_query))
    app.add_error_handler(handlers.error_handler)


def _make_post_init(api_key: str, chat_id: int):
    """Build the ``post_init`` coroutine bound to the right secrets."""

    async def _post_init(app: Application) -> None:
        client = ProprClient(api_key)
        account_id = await _startup_checks(client)

        app.bot_data["propr"] = client
        app.bot_data["account_id"] = account_id

        loop = asyncio.get_event_loop()
        ws_task = loop.create_task(
            run_ws_listener(app.bot, chat_id, api_key, account_id),
            name="propr-ws-listener",
        )
        sweeper_task = loop.create_task(
            sweep_pending_trades(app.bot),
            name="pending-trade-sweeper",
        )
        app.bot_data["background_tasks"] = [ws_task, sweeper_task]

    return _post_init


async def _post_shutdown(app: Application) -> None:
    tasks: List[asyncio.Task] = app.bot_data.get("background_tasks", [])
    for t in tasks:
        if not t.done():
            t.cancel()
    for t in tasks:
        try:
            await t
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
    client = app.bot_data.get("propr")
    if client is not None:
        try:
            await client.close()
        except Exception as exc:  # noqa: BLE001
            log.debug("propr client close failed: %s", exc)


def main() -> None:
    """Main entry point — loads env, builds the app, starts polling."""
    load_dotenv()
    _validate_env()
    _configure_logging()

    token = os.environ["TELEGRAM_BOT_TOKEN"]
    api_key = os.environ["PROPR_API_KEY"]
    chat_id = int(os.environ["TELEGRAM_CHAT_ID"])

    application = (
        Application.builder()
        .token(token)
        .post_init(_make_post_init(api_key, chat_id))
        .post_shutdown(_post_shutdown)
        .build()
    )
    _register_handlers(application)

    log.info("Starting Telegram polling")
    application.run_polling()


if __name__ == "__main__":
    main()
