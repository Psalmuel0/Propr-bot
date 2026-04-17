# Propr-bot

A production-ready Python Telegram bot that trades [Propr](https://www.propr.xyz)
prop-firm accounts on Hyperliquid. You chat with the bot; it places real
orders through the Propr REST API and streams fills back through the Propr
websocket. It also has an expert AI trading brain powered by Groq
(`llama-3.3-70b-versatile`) that gathers live market data plus your account
state, writes a structured analysis, and can auto-execute the trade when you
tap the `✅ Execute Trade` button.

## Features

- Full Propr REST coverage — positions, orders, trades, leverage, margin.
- Async end-to-end on `python-telegram-bot` v20, `httpx`, and `websockets`.
- Websocket listener pushes fills, liquidations, SL/TP hits to Telegram in
  real time; auto-reconnects on disconnect.
- `/analysis` command: Groq writes a full market-structure + risk report,
  parses back into structured fields, and offers a one-tap execute button
  that places entry + stop-loss + take-profits in sequence.
- Natural-language trading: type what you want in plain English; Groq
  parses the intent, the bot shows a confirmation card with Execute/Cancel
  buttons, and routes to the right Propr helper.
- MarkdownV2 replies with safe escaping.
- Challenge-rule aware: surfaces drawdown and daily-loss headroom in every
  analysis.
- Strict authorization: only the chat id in `TELEGRAM_CHAT_ID` can talk to
  the bot.

## Required environment variables

Copy `.env.example` to `.env` and fill in:

| Variable             | Where to get it                                                                 |
|----------------------|---------------------------------------------------------------------------------|
| `PROPR_API_KEY`      | propr.xyz → sign up → dashboard → generate API key (starts with `pk_live_`)     |
| `TELEGRAM_BOT_TOKEN` | Message [@BotFather](https://t.me/BotFather) → `/newbot` → copy the token       |
| `TELEGRAM_CHAT_ID`   | Message [@userinfobot](https://t.me/userinfobot) → copy your numeric chat id    |
| `GROQ_API_KEY`       | [console.groq.com/keys](https://console.groq.com/keys) → create a key           |

## Install

```bash
pip install -r requirements.txt
```

Python 3.11 or newer is required.

## Run

```bash
python bot.py
```

On startup the bot will:

1. Hit `/health` on Propr.
2. Identify itself with `/users/me`.
3. Cache the active challenge `accountId`.
4. Ping Binance and alternative.me (market data sources).
5. Launch the websocket listener and the pending-trade sweeper.
6. Begin long-polling Telegram.

You will see `Bot ready ✅` in the log when it is listening.

## Commands

| Command                                        | Example                                 | Description                                              |
|------------------------------------------------|-----------------------------------------|----------------------------------------------------------|
| `/start`                                       | `/start`                                | Welcome + health + balance                               |
| `/status`                                      | `/status`                               | API, challenge status, realized + unrealized PnL         |
| `/positions`                                   | `/positions`                            | Open positions (non-zero quantity)                       |
| `/orders`                                      | `/orders`                               | Open orders                                              |
| `/trades [n]`                                  | `/trades 10`                            | Last `n` trades (default 5)                              |
| `/buy <asset> <qty> [price]`                   | `/buy BTC 0.001` / `/buy BTC 0.001 92000` | Market (no price) or limit buy                         |
| `/sell <asset> <qty> [price]`                  | `/sell BTC 0.001`                       | Always reduce-only                                       |
| `/close <asset>`                               | `/close BTC`                            | Fully close an open position                             |
| `/cancel <orderId>`                            | `/cancel 01J8...`                       | Cancel a single order                                    |
| `/cancelall`                                   | `/cancelall`                            | Cancel every open order (400s tolerated)                 |
| `/sl <asset> <trigger>`                        | `/sl BTC 91000`                         | Stop-loss on the current long                            |
| `/tp <asset> <trigger>`                        | `/tp BTC 98000`                         | Take-profit on the current long                          |
| `/leverage <asset> <n>`                        | `/leverage BTC 3`                       | Change leverage (BTC/ETH max 5x, others max 2x)          |
| `/analysis <asset> [timeframe] [direction]`    | `/analysis BTC 1h long`                 | AI analysis with inline execute/skip buttons             |
| `/trade <prompt>`                              | `/trade close all btc`                  | Run the natural-language parser on `<prompt>`            |
| `/help`                                        | `/help`                                 | This command list                                        |

`timeframe` accepts `15m`, `1h`, `4h`, `1d` (defaults to `1h`).
`direction` is an optional user hint — leave it empty to let the AI decide.

## Natural language

You can also just type what you want — no slash command. The bot parses the
message with Groq, shows a confirmation card, and executes on tap.

Examples:

- `hey open long on btc with $4,000 and 5x leverage`
- `short eth 0.5 at 3450 with SL 3550 and TP 3300`
- `close my btc`
- `set SL on BTC at 91000`
- `analyze SOL 4h`

Same prompts work with `/trade` prefixed (useful when the first word would
collide with a reserved command name, e.g. `/trade close all btc`).

Confirmations expire after 5 minutes — send the message again to get a fresh card.

## File structure

- `bot.py` — entry point, wires the Telegram `Application` and background tasks
- `propr.py` — async `ProprClient` wrapping every REST endpoint
- `handlers.py` — Telegram command + callback-query handlers
- `analysis.py` — Groq analysis engine, market-data fetcher, recommendation parser
- `ws_listener.py` — Propr websocket listener with auto-reconnect
- `.env.example` — template for the four required env vars
- `requirements.txt` — pinned dependencies
- `.gitignore` — standard ignores (`.env`, `__pycache__/`, venvs)

## Safety notes

- Leverage is capped in code: 5x for BTC/ETH, 2x for everything else.
  Requests above the cap are rejected before hitting Propr.
- HIP-3 assets (`AAPL`, `TSLA`, `NVDA`, `MSFT`, `META`, `GOOGL`, `AMZN`,
  `GOLD`, `SILVER`, `OIL`) are automatically prefixed with `xyz:` so orders
  route to the correct venue.
- Every order carries a fresh ULID `intentId`, generated inside
  `ProprClient.place_order` so the caller cannot forget.
- Pending analysis confirmations expire after 5 minutes; the keyboard is
  removed and an expiry notice is sent.
