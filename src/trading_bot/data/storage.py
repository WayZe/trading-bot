"""Local Parquet storage for OHLCV candles.

Canonical Parquet schema (one row per candle):

- ``timestamp``: ``datetime64[ms, UTC]``
- ``open``, ``high``, ``low``, ``close``, ``volume``: ``float64``

Invariants: rows are sorted by ``timestamp`` and contain no duplicate timestamps.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from trading_bot.data.exchange import timeframe_to_ms

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
OHLCV_PRICE_COLUMNS = ("open", "high", "low", "close", "volume")

TIMESTAMP_DTYPE = "datetime64[ms, UTC]"

_MAX_GAP_MESSAGES = 10


def symbol_to_slug(symbol: str) -> str:
    """Convert a ccxt symbol to a filesystem-safe slug (``BTC/USDT`` -> ``BTC_USDT``)."""
    return symbol.replace("/", "_")


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` in the canonical schema.

    Ensures the column order and dtypes, drops duplicate timestamps
    (keeping the last occurrence), and sorts by timestamp.
    """
    missing = [col for col in OHLCV_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    out = df.loc[:, OHLCV_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).astype(TIMESTAMP_DTYPE)
    for col in OHLCV_PRICE_COLUMNS:
        out[col] = out[col].astype("float64")
    out = out.drop_duplicates(subset="timestamp", keep="last")
    return out.sort_values("timestamp").reset_index(drop=True)


def find_gaps(df: pd.DataFrame, timeframe: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Find gaps between consecutive candles.

    Returns a list of ``(before, after)`` pairs where a whole timeframe step
    or more is missing between the ``before`` candle and the ``after`` candle.
    The input does not need to be sorted; only consecutive timestamps in
    sorted order are compared.
    """
    if len(df) < 2:
        return []
    tf = pd.Timedelta(milliseconds=timeframe_to_ms(timeframe))
    ts = df["timestamp"].sort_values()
    diffs = ts.diff()
    is_gap = diffs > tf
    return [(ts.iloc[i - 1], ts.iloc[i]) for i in range(1, len(ts)) if is_gap.iloc[i]]


def validate_ohlcv(df: pd.DataFrame, timeframe: str) -> list[str]:
    """Validate an OHLCV dataframe against the storage invariants.

    Returns a list of human-readable problems; an empty list means the data
    is valid. Gap problems are reported as ``"gap from X to Y"`` entries.
    """
    if df.empty:
        return []
    problems: list[str] = []
    ts = df["timestamp"]

    for col in OHLCV_COLUMNS:
        nan_mask = df[col].isna()
        count = int(nan_mask.sum())
        if count:
            first = ts[nan_mask].iloc[0]
            problems.append(f"{count} NaN value(s) in column '{col}' (first at {first})")

    body_min = df[["open", "close"]].min(axis=1)
    body_max = df[["open", "close"]].max(axis=1)
    _add_mask_problem(problems, ts, df["high"] < body_max, "high < max(open, close)")
    _add_mask_problem(problems, ts, df["low"] > body_min, "low > min(open, close)")
    _add_mask_problem(problems, ts, df["volume"] < 0, "volume < 0")

    duplicate_count = int(ts.duplicated().sum())
    if duplicate_count:
        problems.append(f"{duplicate_count} duplicate timestamp(s)")
    if not ts.is_monotonic_increasing:
        problems.append("timestamps are not in ascending order")

    problems.extend(_interval_problems(df, timeframe))
    return problems


def _add_mask_problem(
    problems: list[str], ts: pd.Series, mask: pd.Series, description: str
) -> None:
    """Append one aggregated problem entry for rows matching ``mask``."""
    mask = mask.fillna(False)
    count = int(mask.sum())
    if count:
        first = ts[mask].iloc[0]
        problems.append(f"{count} candle(s) where {description} (first at {first})")


def _interval_problems(df: pd.DataFrame, timeframe: str) -> list[str]:
    """Report gaps and irregular (too short) intervals between candles."""
    tf = pd.Timedelta(milliseconds=timeframe_to_ms(timeframe))
    zero = pd.Timedelta(0)
    ts = df["timestamp"].sort_values()
    diffs = ts.diff()
    problems: list[str] = []
    gap_count = 0
    irregular_count = 0
    for i in range(1, len(ts)):
        diff = diffs.iloc[i]
        if diff > tf:
            gap_count += 1
            if gap_count <= _MAX_GAP_MESSAGES:
                missing = int(diff / tf) - 1
                problems.append(
                    f"gap from {ts.iloc[i - 1]} to {ts.iloc[i]} ({missing} missing candle(s))"
                )
        elif zero < diff < tf:
            irregular_count += 1
            if irregular_count <= _MAX_GAP_MESSAGES:
                problems.append(
                    f"irregular interval from {ts.iloc[i - 1]} to {ts.iloc[i]} "
                    f"({diff} < expected {tf})"
                )
    if gap_count > _MAX_GAP_MESSAGES:
        problems.append(f"... and {gap_count - _MAX_GAP_MESSAGES} more gap(s)")
    if irregular_count > _MAX_GAP_MESSAGES:
        problems.append(f"... and {irregular_count - _MAX_GAP_MESSAGES} more irregular interval(s)")
    return problems


class CandleStorage:
    """Parquet-backed candle storage rooted at ``root``.

    Files are addressed as ``{root}/{exchange}/{SYMBOL_SLUG}/{timeframe}.parquet``.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def path_for(self, exchange: str, symbol: str, timeframe: str) -> Path:
        """Return the Parquet path for the given exchange/symbol/timeframe."""
        return self.root / exchange / symbol_to_slug(symbol) / f"{timeframe}.parquet"

    def load(self, exchange: str, symbol: str, timeframe: str) -> pd.DataFrame | None:
        """Load candles from storage; return ``None`` if the file does not exist."""
        path = self.path_for(exchange, symbol, timeframe)
        if not path.exists():
            return None
        return pd.read_parquet(path, engine="pyarrow")

    def save(self, exchange: str, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        """Atomically write candles: normalize, write to a temp file, then rename."""
        path = self.path_for(exchange, symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized = normalize_ohlcv(df)
        tmp_path = path.parent / (path.name + ".tmp")
        normalized.to_parquet(tmp_path, engine="pyarrow", index=False)
        os.replace(tmp_path, path)

    def append(
        self, exchange: str, symbol: str, timeframe: str, new_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge ``new_df`` into the stored candles and save the result.

        Rows with duplicate timestamps are resolved in favour of ``new_df``
        (useful for refreshing the still-open last candle). Returns the merged
        dataframe.
        """
        existing = self.load(exchange, symbol, timeframe)
        if existing is None or existing.empty:
            combined = new_df
        else:
            combined = pd.concat([existing, new_df], ignore_index=True)
        merged = normalize_ohlcv(combined)
        self.save(exchange, symbol, timeframe, merged)
        return merged
