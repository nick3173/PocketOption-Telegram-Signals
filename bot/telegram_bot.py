import os
import math
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("po_telegram_bot")


# ============================================================
# Pocket Option Telegram Robot
# ============================================================
# What this bot does:
# - Sends structure-based trading assessments to Telegram
# - Uses Dow-logic style rules:
#     bullish = HH + HL
#     bearish = LH + LL
# - Detects basic states:
#     PAUSE / ACTIVE BULL / ACTIVE BEAR / STOP MSS
# - Gives a signal only when rules align
#
# What this bot does NOT do:
# - It does not place Pocket Option trades automatically
# - It does not connect to Pocket Option OTC directly
# - It expects candle data from a feed you control
#
# To make it live later:
# - Replace the sample candles in get_latest_candles()
# - Connect your real feed source there
# ============================================================


@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float

    @property
    def bullish(self) -> bool:
        return self.close > self.open

    @property
    def bearish(self) -> bool:
        return self.close < self.open

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low


@dataclass
class Pivot:
    index: int
    price: float
    kind: str  # "high" or "low"
    label: str = ""


@dataclass
class AnalysisResult:
    status: str
    reason: str
    structure: str
    control_candle: str
    confidence: int
    signal: str
    entry: Optional[float]
    stop: Optional[float]
    target: Optional[float]


def detect_pivots(candles: List[Candle], left: int = 2, right: int = 2) -> List[Pivot]:
    pivots: List[Pivot] = []
    for i in range(left, len(candles) - right):
        is_high = all(candles[i].high > candles[j].high for j in range(i - left, i)) and all(
            candles[i].high >= candles[j].high for j in range(i + 1, i + right + 1)
        )
        is_low = all(candles[i].low < candles[j].low for j in range(i - left, i)) and all(
            candles[i].low <= candles[j].low for j in range(i + 1, i + right + 1)
        )
        if is_high:
            pivots.append(Pivot(index=i, price=candles[i].high, kind="high"))
        if is_low:
            pivots.append(Pivot(index=i, price=candles[i].low, kind="low"))
    return pivots


def classify_pivots(pivots: List[Pivot]) -> List[Pivot]:
    last_high: Optional[float] = None
    last_low: Optional[float] = None
    out: List[Pivot] = []
    for p in pivots:
        q = Pivot(index=p.index, price=p.price, kind=p.kind, label="")
        if p.kind == "high":
            if last_high is None:
                q.label = "H?"
            else:
                q.label = "HH" if p.price > last_high else "LH"
            last_high = p.price
        else:
            if last_low is None:
                q.label = "L?"
            else:
                q.label = "HL" if p.price > last_low else "LL"
            last_low = p.price
        out.append(q)
    return out


def average_body(candles: List[Candle], window: int = 10) -> float:
    slice_ = candles[-window:] if len(candles) >= window else candles
    if not slice_:
        return 0.0
    return sum(c.body for c in slice_) / len(slice_)


def detect_control_candle(c: Candle, avg_body_size: float) -> str:
    if c.bullish and c.body > avg_body_size * 1.5 and c.lower_wick <= c.body * 0.25:
        return "Momentum Bull Candle"
    if c.bearish and c.body > avg_body_size * 1.5 and c.upper_wick <= c.body * 0.25:
        return "Momentum Bear Candle"
    if c.lower_wick > c.body * 1.2:
        return "Large Lower Wick"
    if c.upper_wick > c.body * 1.2:
        return "Large Upper Wick"
    return "Neutral Candle"


def detect_compression(candles: List[Candle], lookback: int = 8) -> Tuple[bool, float, float]:
    chunk = candles[-lookback:] if len(candles) >= lookback else candles
    if len(chunk) < 3:
        return False, 0.0, 0.0
    hi = max(c.high for c in chunk)
    lo = min(c.low for c in chunk)
    rng = hi - lo
    avg_b = average_body(chunk, len(chunk))
    return rng <= avg_b * 6, hi, lo


def latest_labels(pivots: List[Pivot]) -> Tuple[Optional[Pivot], Optional[Pivot], Optional[Pivot], Optional[Pivot]]:
    hh = hl = lh = ll = None
    for p in reversed(pivots):
        if p.label == "HH" and hh is None:
            hh = p
        elif p.label == "HL" and hl is None:
            hl = p
        elif p.label == "LH" and lh is None:
            lh = p
        elif p.label == "LL" and ll is None:
            ll = p
        if hh and hl and lh and ll:
            break
    return hh, hl, lh, ll


def analyze_structure(candles: List[Candle]) -> AnalysisResult:
    if len(candles) < 15:
        return AnalysisResult(
            status="PAUSE",
            reason="Not enough candles",
            structure="NEUTRAL",
            control_candle="None",
            confidence=0,
            signal="WAIT",
            entry=None,
            stop=None,
            target=None,
        )

    pivots = classify_pivots(detect_pivots(candles, left=2, right=2))
    avg_b = average_body(candles, 10)
    last = candles[-1]
    prev = candles[-2]
    hh, hl, lh, ll = latest_labels(pivots)
    control = detect_control_candle(last, avg_b)
    is_compression, cons_hi, cons_lo = detect_compression(candles, 8)

    structure = "NEUTRAL"
    if hh and hl and hh.index > hl.index:
        structure = "BULL"
    if lh and ll and ll.index > lh.index:
        structure = "BEAR"

    # MSS
    if structure == "BULL" and hl and last.close < hl.price:
        return AnalysisResult(
            status="STOP",
            reason="MSS - HL breached",
            structure="BULL",
            control_candle=control,
            confidence=70,
            signal="WAIT",
            entry=None,
            stop=None,
            target=None,
        )
    if structure == "BEAR" and lh and last.close > lh.price:
        return AnalysisResult(
            status="STOP",
            reason="MSS - LH breached",
            structure="BEAR",
            control_candle=control,
            confidence=70,
            signal="WAIT",
            entry=None,
            stop=None,
            target=None,
        )

    # Compression / imminent
    if is_compression:
        if last.close > cons_hi or last.close < cons_lo:
            return AnalysisResult(
                status="PAUSE",
                reason="Expansion Imminent",
                structure=structure,
                control_candle=control,
                confidence=68,
                signal="WAIT",
                entry=None,
                stop=None,
                target=None,
            )
        return AnalysisResult(
            status="PAUSE",
            reason="Compression / Consolidation",
            structure=structure,
            control_candle=control,
            confidence=55,
            signal="WAIT",
            entry=None,
            stop=None,
            target=None,
        )

    # Bull logic: entry at HL, stop at LL, target at HH
    if structure == "BULL" and hl and ll and hh:
        near_hl = last.low <= hl.price * 1.0005 or abs(last.low - hl.price) <= avg_b
        bounce = last.close > hl.price and last.close > prev.close
        if near_hl and bounce:
            return AnalysisResult(
                status="ACTIVE",
                reason="Bullish HL retest",
                structure="BULL",
                control_candle=control,
                confidence=82,
                signal="CALL",
                entry=hl.price,
                stop=ll.price,
                target=hh.price,
            )
        return AnalysisResult(
            status="PAUSE",
            reason="Bull pullback / transition",
            structure="BULL",
            control_candle=control,
            confidence=60,
            signal="WAIT",
            entry=None,
            stop=None,
            target=None,
        )

    # Bear logic: entry at LH, stop at HH, target at LL
    if structure == "BEAR" and lh and hh and ll:
        near_lh = last.high >= lh.price * 0.9995 or abs(last.high - lh.price) <= avg_b
        reject = last.close < lh.price and last.close < prev.close
        if near_lh and reject:
            return AnalysisResult(
                status="ACTIVE",
                reason="Bearish LH retest",
                structure="BEAR",
                control_candle=control,
                confidence=82,
                signal="PUT",
                entry=lh.price,
                stop=hh.price,
                target=ll.price,
            )
        return AnalysisResult(
            status="PAUSE",
            reason="Bear pullback / transition",
            structure="BEAR",
            control_candle=control,
            confidence=60,
            signal="WAIT",
            entry=None,
            stop=None,
            target=None,
        )

    return AnalysisResult(
        status="PAUSE",
        reason="No clean structure",
        structure=structure,
        control_candle=control,
        confidence=40,
        signal="WAIT",
        entry=None,
        stop=None,
        target=None,
    )


def format_analysis(symbol: str, tf: str, result: AnalysisResult) -> str:
    lines = [
        f"*Pocket Option Engine*",
        f"Symbol: `{symbol}`",
        f"TF: `{tf}`",
        "",
        f"Status: *{result.status}*",
        f"Reason: {result.reason}",
        f"Structure: {result.structure}",
        f"Control Candle: {result.control_candle}",
        f"Confidence: {result.confidence}%",
        f"Signal: *{result.signal}*",
    ]
    if result.entry is not None:
        lines += [
            f"Entry: `{result.entry:.5f}`",
            f"Stop: `{result.stop:.5f}`",
            f"Target: `{result.target:.5f}`",
        ]
    return "\n".join(lines)


def get_latest_candles(symbol: str, timeframe: str) -> List[Candle]:
    """
    Replace this function with your real live feed.

    For now, this sample data lets the Telegram bot run immediately.
    """
    raw = [
        (1.1000, 1.1010, 1.0990, 1.1008),
        (1.1008, 1.1020, 1.1001, 1.1017),
        (1.1017, 1.1030, 1.1012, 1.1028),
        (1.1028, 1.1034, 1.1018, 1.1020),
        (1.1020, 1.1024, 1.1009, 1.1012),
        (1.1012, 1.1015, 1.0998, 1.1003),
        (1.1003, 1.1014, 1.1000, 1.1011),
        (1.1011, 1.1026, 1.1008, 1.1022),
        (1.1022, 1.1038, 1.1018, 1.1034),
        (1.1034, 1.1040, 1.1020, 1.1025),
        (1.1025, 1.1030, 1.1017, 1.1019),
        (1.1019, 1.1022, 1.1008, 1.1010),
        (1.1010, 1.1018, 1.1007, 1.1016),
        (1.1016, 1.1025, 1.1010, 1.1021),
        (1.1021, 1.1032, 1.1016, 1.1029),
        (1.1029, 1.1030, 1.1020, 1.1022),
        (1.1022, 1.1024, 1.1012, 1.1015),
        (1.1015, 1.1020, 1.1010, 1.1018),
    ]
    return [Candle(*r) for r in raw]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Pocket Option Telegram Robot is live.\n\n"
        "Commands:\n"
        "/status - current bot status\n"
        "/analyze EURUSD M1 - run structure scan\n"
        "/rules - show current trading rules\n"
    )
    await update.message.reply_text(text)


async def rules(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Current rules:\n"
        "- Bull structure = HH + HL\n"
        "- Bear structure = LH + LL\n"
        "- Bull entry at HL\n"
        "- Bear entry at LH\n"
        "- Bull stop at LL\n"
        "- Bear stop at HH\n"
        "- Target swing HH or LL\n"
        "- PAUSE on compression\n"
        "- STOP on MSS\n"
    )
    await update.message.reply_text(text)


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Bot online. Feed source: sample candles. Replace get_latest_candles() for live use.")


async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args
    symbol = args[0].upper() if len(args) >= 1 else "EURUSD"
    tf = args[1].upper() if len(args) >= 2 else "M1"

    candles = get_latest_candles(symbol, tf)
    result = analyze_structure(candles)
    msg = format_analysis(symbol, tf, result)
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN in your environment first.")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("rules", rules))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("analyze", analyze))

    logger.info("Starting Pocket Option Telegram Robot...")
    app.run_polling()


if __name__ == "__main__":
    main()
