"""PnL snapshot, text formatting, and image-card rendering.

Self-contained module: does all PnL-related work (computation, MarkdownV2
text, PNG card) so that :mod:`handlers` and :mod:`ws_listener` can keep
their own concerns clean. Only depends on :mod:`utils` for shared helpers;
must never import from :mod:`handlers` (cycle risk).

All math uses :class:`decimal.Decimal` \u2014 numeric fields from the Propr API
come as strings, so we parse once via :func:`_dec` and stay in ``Decimal``.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from utils import escape_md as _default_escape_md
from utils import fmt_num

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PositionPnL:
    """A single position's PnL fields, all parsed into :class:`Decimal`."""

    asset: str
    side: str          # "long" | "short"
    quantity: Decimal
    entry_price: Decimal
    mark_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    leverage: Decimal
    margin_used: Decimal
    roe_pct: Decimal   # returnOnEquity * 100


@dataclass
class PnLSnapshot:
    """Aggregated PnL view for all open positions at a point in time."""

    positions: List[PositionPnL]
    total_unrealized: Decimal
    total_realized: Decimal
    total_margin_used: Decimal
    generated_at: datetime  # UTC

    @property
    def total(self) -> Decimal:
        return self.total_unrealized + self.total_realized


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _dec(value: Any, default: str = "0") -> Decimal:
    """Coerce *value* to :class:`Decimal`; return ``Decimal(default)`` on failure."""
    if value is None or value == "":
        return Decimal(default)
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal(default)


def _signed(value: Decimal, places: int = 2) -> str:
    """Render *value* with a leading ``+``/``-`` (bare ``0`` for zero).

    Uses :func:`fmt_num` so trailing zeros are stripped.
    """
    text = fmt_num(value, places)
    if text == "0" or text.startswith("-"):
        return text
    return "+" + text


def _asset_symbol(raw: str) -> str:
    """Strip any ``xyz:`` prefix and upper-case the symbol."""
    if not raw:
        return "?"
    return str(raw).split(":")[-1].upper() or "?"


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------
def build_snapshot(raw_positions: Iterable[dict]) -> PnLSnapshot:
    """Turn the raw Propr position list into a :class:`PnLSnapshot`.

    Positions with ``quantity == 0`` are filtered out; totals sum across the
    remaining rows.
    """
    positions: List[PositionPnL] = []
    for raw in raw_positions or []:
        if not isinstance(raw, dict):
            continue
        quantity = _dec(raw.get("quantity", raw.get("qty")))
        if quantity == 0:
            continue
        side = str(raw.get("positionSide") or raw.get("side") or "").lower()
        roe = _dec(raw.get("returnOnEquity"))
        positions.append(
            PositionPnL(
                asset=_asset_symbol(raw.get("asset", "")),
                side=side,
                quantity=quantity,
                entry_price=_dec(raw.get("entryPrice", raw.get("avgEntryPrice"))),
                mark_price=_dec(
                    raw.get("markPrice", raw.get("mark", raw.get("price")))
                ),
                unrealized_pnl=_dec(raw.get("unrealizedPnl", raw.get("uPnl"))),
                realized_pnl=_dec(raw.get("realizedPnl", raw.get("rPnl"))),
                leverage=_dec(raw.get("leverage"), "1"),
                margin_used=_dec(raw.get("marginUsed", raw.get("margin"))),
                roe_pct=roe * Decimal(100),
            )
        )

    total_u = sum((p.unrealized_pnl for p in positions), Decimal(0))
    total_r = sum((p.realized_pnl for p in positions), Decimal(0))
    total_m = sum((p.margin_used for p in positions), Decimal(0))
    return PnLSnapshot(
        positions=positions,
        total_unrealized=total_u,
        total_realized=total_r,
        total_margin_used=total_m,
        generated_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# MarkdownV2 formatter
# ---------------------------------------------------------------------------
_SEPARATOR = "────────────────────────"  # light box-drawing — no MD2 escape needed


def format_snapshot_markdown(snapshot: PnLSnapshot, escape_md=None) -> str:
    """Return a MarkdownV2 string ready to send to Telegram.

    All dynamic content is pushed through *escape_md* (defaulting to the
    shared :func:`utils.escape_md`); static punctuation is pre-escaped in
    the string literals below. ``|``/``(``/``)``/``+``/``.`` \u2014 the MD2
    specials that matter for this layout \u2014 are either escaped inline or
    come out of ``escape_md``.
    """
    esc = escape_md or _default_escape_md

    if not snapshot.positions:
        return "💭 No open positions\\."

    lines: List[str] = ["💰 *PnL Snapshot*", _SEPARATOR]

    for pos in snapshot.positions:
        side_emoji = "📈" if pos.side == "long" else "📉"
        side_text = pos.side.upper() if pos.side else "?"

        qty_s = fmt_num(pos.quantity, 6)
        entry_s = fmt_num(pos.entry_price)
        mark_s = fmt_num(pos.mark_price)

        upnl_str = _signed(pos.unrealized_pnl)
        notional = pos.quantity * pos.entry_price
        if notional != 0:
            pnl_pct = (pos.unrealized_pnl / notional) * Decimal(100)
        else:
            pnl_pct = Decimal(0)
        pnl_pct_str = _signed(pnl_pct)
        dot = "🟢" if pos.unrealized_pnl >= 0 else "🔴"

        rpnl_s = fmt_num(pos.realized_pnl)
        lev_s = fmt_num(pos.leverage, 0)

        # Header line: emoji, side, asset, qty @ entry -> mark (latter in code spans)
        lines.append(
            f"{side_emoji} {esc(side_text)}  {esc(pos.asset)}  "
            f"`{qty_s} @ {entry_s}` → `{mark_s}`"
        )
        # uPnL line
        lines.append(
            f"   uPnL: *{esc(upnl_str)} USDC* "
            f"\\({esc(pnl_pct_str)}%\\)  {dot}"
        )
        # rPnL + leverage
        lines.append(
            f"   rPnL: {esc(rpnl_s)} USDC  \\|  lev {esc(lev_s)}x"
        )

    lines.append(_SEPARATOR)

    total_dot = "🟢" if snapshot.total_unrealized >= 0 else "🔴"
    lines.append(
        f"Total uPnL: *{esc(_signed(snapshot.total_unrealized))} USDC*  {total_dot}"
    )
    lines.append(
        f"Total rPnL: `{fmt_num(snapshot.total_realized)} USDC`"
    )
    lines.append(
        f"*Combined: {esc(_signed(snapshot.total))} USDC*"
    )
    lines.append(
        f"Margin used: `{fmt_num(snapshot.total_margin_used)} USDC`"
    )
    ts = snapshot.generated_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    lines.append(f"📅 {esc(ts)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PNG card rendering
# ---------------------------------------------------------------------------
_BG = "#0D1117"
_CARD_BG = "#161B22"
_GREEN = "#22C55E"
_RED = "#EF4444"
_WHITE = "#FFFFFF"
_GRAY = "#9CA3AF"
_FOOTER_GRAY = "#6B7280"
_SEP_LINE = "#30363D"

_FONT_CANDIDATES_BOLD = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
_FONT_CANDIDATES_REG = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)

_FONT_FALLBACK_WARNED = False


def _load_font(candidates: Tuple[str, ...], size: int) -> ImageFont.ImageFont:
    """Try each *candidates* path in order; warn once and fall back on default."""
    global _FONT_FALLBACK_WARNED
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except (OSError, IOError):
            continue
    if not _FONT_FALLBACK_WARNED:
        log.warning(
            "DejaVu fonts not found, falling back to PIL default (lower quality)"
        )
        _FONT_FALLBACK_WARNED = True
    return ImageFont.load_default()


def _load_fonts() -> Dict[str, ImageFont.ImageFont]:
    return {
        "title": _load_font(_FONT_CANDIDATES_BOLD, 48),
        "total_num": _load_font(_FONT_CANDIDATES_BOLD, 56),
        "total_lbl": _load_font(_FONT_CANDIDATES_REG, 18),
        "timestamp": _load_font(_FONT_CANDIDATES_REG, 18),
        "asset": _load_font(_FONT_CANDIDATES_BOLD, 28),
        "pill": _load_font(_FONT_CANDIDATES_BOLD, 16),
        "middle": _load_font(_FONT_CANDIDATES_REG, 18),
        "pnl": _load_font(_FONT_CANDIDATES_BOLD, 28),
        "roe": _load_font(_FONT_CANDIDATES_REG, 16),
        "empty": _load_font(_FONT_CANDIDATES_BOLD, 36),
        "more": _load_font(_FONT_CANDIDATES_BOLD, 22),
        "footer": _load_font(_FONT_CANDIDATES_REG, 14),
    }


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int, int, int]:
    """Return (width, height, x_offset, y_offset) for *text* rendered with *font*."""
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1], bbox[0], bbox[1]


def _draw_text_centered(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    """Center-draw *text* inside the ``(x0, y0, x1, y1)`` *box*."""
    x0, y0, x1, y1 = box
    w, h, ox, oy = _text_size(draw, text, font)
    x = x0 + ((x1 - x0) - w) / 2 - ox
    y = y0 + ((y1 - y0) - h) / 2 - oy
    draw.text((x, y), text, font=font, fill=fill)


def _draw_header(
    draw: ImageDraw.ImageDraw,
    width: int,
    snapshot: PnLSnapshot,
    fonts: Dict[str, ImageFont.ImageFont],
    padding: int,
) -> None:
    """Draw the top-left title/timestamp and top-right combined-total block."""
    draw.text((padding, padding), "PnL Snapshot", font=fonts["title"], fill=_WHITE)

    ts = snapshot.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC")
    draw.text((padding, padding + 62), ts, font=fonts["timestamp"], fill=_GRAY)

    # Right-aligned TOTAL label + big combined number
    total_text = f"{_signed(snapshot.total)} USDC"
    total_color = _GREEN if snapshot.total >= 0 else _RED

    lbl_w, _, lbl_ox, _ = _text_size(draw, "TOTAL", fonts["total_lbl"])
    num_w, _, num_ox, _ = _text_size(draw, total_text, fonts["total_num"])

    right_x = width - padding
    draw.text(
        (right_x - lbl_w - lbl_ox, padding + 4),
        "TOTAL",
        font=fonts["total_lbl"],
        fill=_GRAY,
    )
    draw.text(
        (right_x - num_w - num_ox, padding + 28),
        total_text,
        font=fonts["total_num"],
        fill=total_color,
    )


def _draw_position_card(
    draw: ImageDraw.ImageDraw,
    y: int,
    pos: PositionPnL,
    fonts: Dict[str, ImageFont.ImageFont],
    padding: int,
    width: int,
) -> None:
    """Draw a single position row as a rounded card at vertical offset *y*."""
    card_x0 = padding
    card_x1 = width - padding
    card_y1 = y + 90
    draw.rounded_rectangle(
        (card_x0, y, card_x1, card_y1),
        radius=12,
        fill=_CARD_BG,
    )

    # Pill
    pill_color = _GREEN if pos.side == "long" else _RED
    pill_text = pos.side.upper() if pos.side in ("long", "short") else "?"
    pill_x0 = card_x0 + 16
    pill_y0 = y + 29
    pill_w, pill_h = 80, 32
    pill_x1 = pill_x0 + pill_w
    pill_y1 = pill_y0 + pill_h
    draw.rounded_rectangle(
        (pill_x0, pill_y0, pill_x1, pill_y1),
        radius=pill_h // 2,
        fill=pill_color,
    )
    _draw_text_centered(
        draw, (pill_x0, pill_y0, pill_x1, pill_y1), pill_text, fonts["pill"], _WHITE
    )

    # Asset symbol — 28px bold white next to the pill
    asset_x = pill_x1 + 16
    asset_w, asset_h, asset_ox, asset_oy = _text_size(draw, pos.asset, fonts["asset"])
    asset_y = y + (90 - asset_h) / 2 - asset_oy
    draw.text((asset_x - asset_ox, asset_y), pos.asset, font=fonts["asset"], fill=_WHITE)

    # Middle: qty @ entry → mark
    qty_s = fmt_num(pos.quantity, 6)
    entry_s = fmt_num(pos.entry_price)
    mark_s = fmt_num(pos.mark_price)
    mid_text = f"{qty_s} @ {entry_s}  →  {mark_s}"
    mid_x = asset_x + asset_w + 40
    mid_w, mid_h, mid_ox, mid_oy = _text_size(draw, mid_text, fonts["middle"])
    mid_y = y + (90 - mid_h) / 2 - mid_oy
    draw.text((mid_x - mid_ox, mid_y), mid_text, font=fonts["middle"], fill=_GRAY)

    # Right column: uPnL + ROE%
    pnl_color = _GREEN if pos.unrealized_pnl >= 0 else _RED
    pnl_text = f"{_signed(pos.unrealized_pnl)} USDC"
    roe_text = f"ROE {_signed(pos.roe_pct)}%"

    pnl_w, _, pnl_ox, _ = _text_size(draw, pnl_text, fonts["pnl"])
    roe_w, _, roe_ox, _ = _text_size(draw, roe_text, fonts["roe"])

    right_x = card_x1 - 20
    draw.text(
        (right_x - pnl_w - pnl_ox, y + 15),
        pnl_text,
        font=fonts["pnl"],
        fill=pnl_color,
    )
    draw.text(
        (right_x - roe_w - roe_ox, y + 55),
        roe_text,
        font=fonts["roe"],
        fill=_GRAY,
    )


def _draw_positions(
    draw: ImageDraw.ImageDraw,
    snapshot: PnLSnapshot,
    fonts: Dict[str, ImageFont.ImageFont],
    padding: int,
    width: int,
    start_y: int = 150,
) -> None:
    """Draw up to 5 position cards, plus a ``+N more`` row if truncated."""
    visible = snapshot.positions[:5]
    overflow = len(snapshot.positions) - len(visible)
    card_stride = 110  # 90px card + 20px gap

    for i, pos in enumerate(visible):
        _draw_position_card(
            draw, start_y + i * card_stride, pos, fonts, padding, width
        )

    if overflow > 0:
        y = start_y + len(visible) * card_stride
        label = f"+{overflow} more"
        lw, lh, lox, loy = _text_size(draw, label, fonts["more"])
        draw.text(
            ((width - lw) / 2 - lox, y + (30 - lh) / 2 - loy),
            label,
            font=fonts["more"],
            fill=_GRAY,
        )


def _draw_empty(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    fonts: Dict[str, ImageFont.ImageFont],
) -> None:
    """Centered 'No open positions' for the empty snapshot case."""
    _draw_text_centered(
        draw, (0, 0, width, height), "No open positions", fonts["empty"], _GRAY
    )


def _draw_footer(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    fonts: Dict[str, ImageFont.ImageFont],
    padding: int,
) -> None:
    """Thin separator + centered brand footer at the bottom of the canvas."""
    sep_y = height - 40
    draw.line(
        [(padding, sep_y), (width - padding, sep_y)],
        fill=_SEP_LINE,
        width=1,
    )
    text = "propr.xyz · live PnL"
    tw, th, tox, toy = _text_size(draw, text, fonts["footer"])
    draw.text(
        ((width - tw) / 2 - tox, sep_y + 10),
        text,
        font=fonts["footer"],
        fill=_FOOTER_GRAY,
    )


def render_snapshot_image(snapshot: PnLSnapshot) -> bytes:
    """Render the PnL card as a 1200x800 PNG and return its raw bytes.

    Pillow calls are blocking, so callers should invoke this via
    :func:`asyncio.to_thread` from an async context.
    """
    width, height = 1200, 800
    padding = 40

    img = Image.new("RGB", (width, height), _BG)
    draw = ImageDraw.Draw(img)
    fonts = _load_fonts()

    _draw_header(draw, width, snapshot, fonts, padding)

    if not snapshot.positions:
        _draw_empty(draw, width, height, fonts)
    else:
        _draw_positions(draw, snapshot, fonts, padding, width)

    _draw_footer(draw, width, height, fonts, padding)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    fake_raw = [
        {
            "positionId": "pos-1",
            "asset": "BTC",
            "positionSide": "long",
            "status": "open",
            "quantity": "0.010",
            "entryPrice": "94500",
            "markPrice": "94850",
            "unrealizedPnl": "3.5",
            "realizedPnl": "0",
            "leverage": "3",
            "marginUsed": "315",
            "returnOnEquity": "0.011111",
            "cumulativeFunding": "0",
            "cumulativeTradingFees": "0",
        },
        {
            "positionId": "pos-2",
            "asset": "xyz:ETH",
            "positionSide": "short",
            "status": "open",
            "quantity": "0.5",
            "entryPrice": "3500",
            "markPrice": "3565",
            "unrealizedPnl": "-32.5",
            "realizedPnl": "-2.1",
            "leverage": "2",
            "marginUsed": "875",
            "returnOnEquity": "-0.03714",
            "cumulativeFunding": "0",
            "cumulativeTradingFees": "0.25",
        },
    ]

    snap = build_snapshot(fake_raw)
    print(format_snapshot_markdown(snap, _default_escape_md))
    print()

    png = render_snapshot_image(snap)
    out = "/tmp/pnl_smoke.png"
    with open(out, "wb") as fh:
        fh.write(png)
    print(f"wrote {out} ({len(png)} bytes)")

    if not png.startswith(b"\x89PNG"):
        print("ERROR: output is not a PNG", file=sys.stderr)
        sys.exit(1)
    if len(png) < 1024:
        print(f"ERROR: PNG too small ({len(png)} bytes)", file=sys.stderr)
        sys.exit(1)
    print("OK")
