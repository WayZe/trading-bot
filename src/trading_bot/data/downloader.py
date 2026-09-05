"""Historical OHLCV downloading with pagination, retries, and gap backfilling."""

from __future__ import annotations

import logging
import time
from datetime import UTC, date, datetime

import ccxt
import pandas as pd

from trading_bot.data.exchange import ExchangeClient, timeframe_to_ms
from trading_bot.data.storage import (
    OHLCV_COLUMNS,
    CandleStorage,
    find_gaps,
    normalize_ohlcv,
    validate_ohlcv,
)

logger = logging.getLogger(__name__)

EXCHANGE_ID = "bybit"
BATCH_SIZE = 1000
MAX_ATTEMPTS = 6
MAX_BACKOFF_SECONDS = 16
PROGRESS_EVERY_BATCHES = 10
# RateLimitExceeded inherits from NetworkError in ccxt, so it is covered too.
RETRYABLE_ERRORS: tuple[type[Exception], ...] = (ccxt.NetworkError,)


def _now_ms() -> int:
    """Current UTC time in milliseconds since the epoch."""
    return int(datetime.now(UTC).timestamp() * 1000)


def _drop_open_candles(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Drop candles whose bucket is still open.

    A candle is closed iff ``timestamp + timeframe_ms <= now``; the
    still-forming last candle must never enter the dataset (it would poison
    both the gap validation and the backtest).
    """
    tf_ms = timeframe_to_ms(timeframe)
    cutoff = pd.Timestamp(_now_ms(), unit="ms", tz="UTC")
    is_closed = (df["timestamp"] + pd.Timedelta(milliseconds=tf_ms)) <= cutoff
    dropped = int((~is_closed).sum())
    if dropped:
        logger.info(
            "dropped %d still-open candle(s): their buckets have not closed yet", dropped
        )
    return df.loc[is_closed].reset_index(drop=True)


def _as_date(value: date | str, *, name: str, default: date | None = None) -> date:
    """Coerce ``date | str`` into a ``date``; use ``default`` for missing values."""
    if value is None:
        if default is not None:
            return default
        raise ValueError(f"{name} is required")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _date_to_ms(value: date) -> int:
    """Convert a UTC calendar date to milliseconds since the epoch."""
    moment = datetime(value.year, value.month, value.day, tzinfo=UTC)
    return int(moment.timestamp() * 1000)


def _ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat()


def _rows_to_df(rows: list[list]) -> pd.DataFrame:
    """Convert raw ``[ts_ms, o, h, l, c, v]`` rows to the canonical schema."""
    df = pd.DataFrame(rows, columns=["timestamp_ms", *OHLCV_COLUMNS[1:]])
    df["timestamp"] = pd.to_datetime(df.pop("timestamp_ms"), unit="ms", utc=True).astype(
        "datetime64[ms, UTC]"
    )
    for col in OHLCV_COLUMNS[1:]:
        df[col] = df[col].astype("float64")
    return df.loc[:, OHLCV_COLUMNS]


class HistoryDownloader:
    """Download OHLCV history from the exchange with retry and validation."""

    def __init__(self, exchange: ExchangeClient) -> None:
        self.exchange = exchange

    def download(
        self,
        symbol: str,
        timeframe: str,
        since: date | str,
        until: date | str | None = None,
    ) -> pd.DataFrame:
        """Download candles for ``[since, until)`` (UTC dates).

        ``until`` defaults to the current UTC date. Candles whose bucket is
        still open (``timestamp + timeframe > now``) are dropped: only closed
        candles are returned. Raises ``ValueError`` if the exchange returns no
        data at all or if validation fails even after gap backfilling.
        """
        since_date = _as_date(since, name="since")
        until_date = _as_date(until, name="until", default=datetime.now(UTC).date())
        since_ms = _date_to_ms(since_date)
        until_ms = _date_to_ms(until_date)
        logger.info(
            "downloading %s %s from %s to %s", symbol, timeframe, since_date, until_date
        )
        return self._download_ms(symbol, timeframe, since_ms, until_ms)

    def update(
        self, symbol: str, timeframe: str, storage: CandleStorage
    ) -> pd.DataFrame:
        """Fetch candles from the last stored one up to now and merge them.

        The last stored candle is re-fetched as well, because it may have been
        written while still open; the still-open tail is dropped before
        saving, so only closed candles ever reach the dataset. Raises
        ``FileNotFoundError`` if there is no stored dataset yet (run a full
        download first) and ``ValueError`` if validation fails after the
        merge.
        """
        existing = storage.load(EXCHANGE_ID, symbol, timeframe)
        if existing is None or existing.empty:
            raise FileNotFoundError(
                f"no stored dataset for {EXCHANGE_ID} {symbol} {timeframe}; "
                "run a full download first"
            )
        last_ms = int(existing["timestamp"].iloc[-1].value // 1_000_000)
        now_ms = _now_ms()
        logger.info(
            "updating %s %s from %s to now", symbol, timeframe, _ms_to_iso(last_ms)
        )
        tail = self._download_ms(symbol, timeframe, last_ms, now_ms)
        combined = storage.append(EXCHANGE_ID, symbol, timeframe, tail)
        problems = validate_ohlcv(combined, timeframe)
        if problems:
            raise ValueError(
                f"OHLCV validation failed for {symbol} {timeframe} after update:\n- "
                + "\n- ".join(problems)
            )
        return combined

    def _download_ms(
        self, symbol: str, timeframe: str, since_ms: int, until_ms: int
    ) -> pd.DataFrame:
        if since_ms >= until_ms:
            raise ValueError(
                f"since ({_ms_to_iso(since_ms)}) must be before until ({_ms_to_iso(until_ms)})"
            )
        rows = self._paginate(symbol, timeframe, since_ms, until_ms)
        if not rows:
            raise ValueError(
                f"no candles returned for {symbol} {timeframe} "
                f"in [{_ms_to_iso(since_ms)}, {_ms_to_iso(until_ms)})"
            )
        df = normalize_ohlcv(_rows_to_df(rows))
        df = self._backfill_gaps(symbol, timeframe, df, until_ms)
        df = _drop_open_candles(df, timeframe)
        if df.empty:
            raise ValueError(
                f"no closed candles available for {symbol} {timeframe} "
                f"in [{_ms_to_iso(since_ms)}, {_ms_to_iso(until_ms)}): "
                "the last bucket is still open"
            )
        problems = validate_ohlcv(df, timeframe)
        if problems:
            raise ValueError(
                f"OHLCV validation failed for {symbol} {timeframe}:\n- "
                + "\n- ".join(problems)
            )
        return df

    def _paginate(
        self, symbol: str, timeframe: str, since_ms: int, until_ms: int
    ) -> list[list]:
        """Fetch all candles in ``[since_ms, until_ms)`` page by page."""
        tf_ms = timeframe_to_ms(timeframe)
        rows: list[list] = []
        cursor = since_ms
        batch_number = 0
        while cursor < until_ms:
            batch = self._fetch_with_retry(symbol, timeframe, cursor, BATCH_SIZE)
            batch_number += 1
            batch = [c for c in batch if since_ms <= c[0] < until_ms]
            if not batch:
                logger.info(
                    "no further candles available (cursor at %s)", _ms_to_iso(cursor)
                )
                break
            rows.extend(batch)
            next_cursor = batch[-1][0] + tf_ms
            if next_cursor <= cursor:
                logger.warning(
                    "exchange did not advance pagination (cursor=%d, last candle=%d); "
                    "stopping to avoid an infinite loop",
                    cursor,
                    batch[-1][0],
                )
                break
            cursor = next_cursor
            if batch_number % PROGRESS_EVERY_BATCHES == 0:
                logger.info(
                    "downloaded %d candles so far (cursor at %s)",
                    len(rows),
                    _ms_to_iso(cursor),
                )
        return rows

    def _fetch_with_retry(
        self, symbol: str, timeframe: str, since_ms: int, limit: int
    ) -> list[list]:
        """Fetch one batch, retrying transient network errors with backoff."""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return self.exchange.fetch_ohlcv(
                    symbol, timeframe, since_ms=since_ms, limit=limit
                )
            except RETRYABLE_ERRORS as exc:
                if attempt == MAX_ATTEMPTS:
                    raise
                delay = min(2 ** (attempt - 1), MAX_BACKOFF_SECONDS)
                logger.warning(
                    "fetch attempt %d/%d failed (%s); retrying in %ds",
                    attempt,
                    MAX_ATTEMPTS,
                    exc,
                    delay,
                )
                time.sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    def _backfill_gaps(
        self, symbol: str, timeframe: str, df: pd.DataFrame, until_ms: int
    ) -> pd.DataFrame:
        """Detect gaps and fetch the missing ranges in separate requests."""
        tf_ms = timeframe_to_ms(timeframe)
        for _ in range(3):
            gaps = find_gaps(df, timeframe)
            if not gaps:
                return df
            logger.info("detected %d gap(s); fetching missing ranges", len(gaps))
            extra_rows: list[list] = []
            for before, after in gaps:
                start_ms = int(before.value // 1_000_000) + tf_ms
                end_ms = min(int(after.value // 1_000_000), until_ms)
                extra_rows.extend(self._paginate(symbol, timeframe, start_ms, end_ms))
            if not extra_rows:
                logger.warning("gap backfill returned no candles; keeping the current dataset")
                return df
            df = normalize_ohlcv(
                pd.concat([df, _rows_to_df(extra_rows)], ignore_index=True)
            )
        return df
