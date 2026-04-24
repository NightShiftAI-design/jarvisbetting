"""Shared helpers for Jarvis_Betting.

All user-facing timestamps are rendered in America/New_York with pytz. Internal
calculations can stay UTC-aware, but dashboard surfaces should call format_est or
convert_series_to_est before display.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pytz

from config import LOG_FILE, LOG_LEVEL

EST_TIMEZONE = pytz.timezone("America/New_York")
SHARP_BOOK_KEYS = {"pinnacle", "circa", "circa_sports"}
MAJOR_BOOK_HINTS = {
    "fanduel": "FanDuel",
    "draftkings": "DraftKings",
    "betmgm": "BetMGM",
    "caesars": "Caesars",
    "espnbet": "ESPN BET",
    "fanatics": "Fanatics",
    "pinnacle": "Pinnacle",
    "circa": "Circa",
    "circa_sports": "Circa",
}


def setup_logging(log_level: str = LOG_LEVEL, log_file: Path = LOG_FILE) -> logging.Logger:
    logger = logging.getLogger("jarvis_betting")
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


LOGGER = setup_logging()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def est_now() -> datetime:
    return utcnow().astimezone(EST_TIMEZONE)


def to_est(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        if isinstance(value, pd.Timestamp):
            ts = value if value.tzinfo else value.tz_localize("UTC")
            return ts.tz_convert(EST_TIMEZONE).to_pydatetime()
        if isinstance(value, datetime):
            dt = pytz.UTC.localize(value) if value.tzinfo is None else value
            return dt.astimezone(EST_TIMEZONE)
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.tz_convert(EST_TIMEZONE).to_pydatetime()
    except Exception:
        return None


def format_est(value: Any, fmt: str = "%b %d, %Y %I:%M %p %Z") -> str:
    converted = to_est(value)
    return converted.strftime(fmt) if converted else "N/A"


def convert_series_to_est(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert(EST_TIMEZONE)


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_json_loads(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def normalize_bookmaker(value: Any) -> str:
    key = normalize_text(value).replace(" ", "_").replace("-", "_")
    return key


def display_bookmaker(value: Any) -> str:
    key = normalize_bookmaker(value)
    return MAJOR_BOOK_HINTS.get(key, str(value or "Unknown"))


def is_sharp_book(value: Any) -> bool:
    key = normalize_bookmaker(value)
    return key in SHARP_BOOK_KEYS or "pinnacle" in key or "circa" in key


def american_to_decimal(price: float | int | None) -> float | None:
    price = safe_float(price)
    if price is None:
        return None
    try:
        if price > 0:
            return 1.0 + price / 100.0
        return 1.0 + 100.0 / abs(price)
    except ZeroDivisionError:
        return None


def decimal_to_american(decimal_odds: float | None) -> int | None:
    decimal_odds = safe_float(decimal_odds)
    if decimal_odds is None or decimal_odds <= 1:
        return None
    if decimal_odds >= 2:
        return int(round((decimal_odds - 1) * 100))
    return int(round(-100 / (decimal_odds - 1)))


def american_to_implied_probability(price: float | int | None) -> float | None:
    price = safe_float(price)
    if price is None:
        return None
    if price > 0:
        return 100.0 / (price + 100.0)
    return abs(price) / (abs(price) + 100.0)


def probability_to_american(probability: float | None) -> int | None:
    probability = safe_float(probability)
    if probability is None or probability <= 0 or probability >= 1:
        return None
    if probability >= 0.5:
        return int(round(-(probability / (1 - probability)) * 100))
    return int(round(((1 - probability) / probability) * 100))


def compute_edge(model_probability: float | None, market_probability: float | None) -> float | None:
    model_probability = safe_float(model_probability)
    market_probability = safe_float(market_probability)
    if model_probability is None or market_probability is None:
        return None
    return model_probability - market_probability


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def pct(value: float | None, digits: int = 1) -> str:
    value = safe_float(value)
    if value is None:
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def money(value: float | None) -> str:
    value = safe_float(value)
    if value is None:
        return "N/A"
    return f"${value:,.2f}"


def chunked(values: Iterable[Any], size: int) -> list[list[Any]]:
    bucket: list[Any] = []
    output: list[list[Any]] = []
    for value in values:
        bucket.append(value)
        if len(bucket) >= size:
            output.append(bucket)
            bucket = []
    if bucket:
        output.append(bucket)
    return output


def make_cache_key(*parts: Any) -> str:
    joined = "|".join(str(part) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def trend_label(edge: float | None, confidence: float | None) -> str:
    edge = safe_float(edge) or 0.0
    confidence = safe_float(confidence) or 0.0
    if edge >= 0.06 and confidence >= 0.72:
        return "Premium edge"
    if edge >= 0.035:
        return "Positive value"
    if edge <= -0.035:
        return "Avoid"
    return "Neutral"


if __name__ == "__main__":
    print("Current EST:", format_est(est_now()))
    print("-150 implied:", american_to_implied_probability(-150))
