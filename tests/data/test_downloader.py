"""Tests for the history downloader (offline, exchange is faked)."""

from __future__ import annotations

import ccxt
import pandas as pd
import pytest

from tests.conftest import BASE_MS, HOUR_MS, make_candles, rows_to_df
from trading_bot.data.downloader import MAX_ATTEMPTS, HistoryDownloader
from trading_bot.data.storage import CandleStorage, validate_ohlcv


class FakeExchange:
    """Emulates exchange-side OHLCV pagination.

    Behaves like a real exchange: returns up to ``limit`` candles with
    timestamps ``>= since_ms``. A queue of exceptions can be injected to
    simulate transient network failures.
    """

    def __init__(self, rows: list[list], errors: list[Exception] | None = None):
        self.rows = rows
        self.errors = list(errors or [])
        self.calls: list[int | None] = []

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int | None = None,
        limit: int = 1000,
    ) -> list[list]:
        self.calls.append(since_ms)
        if self.errors:
            raise self.errors.pop(0)
        return [r for r in self.rows if r[0] >= since_ms][:limit]


class GapExchange:
    """Hides a range of candles on the first request and reveals them later."""

    def __init__(self, full_rows: list[list], hidden_rows: list[list]):
        self.full_rows = full_rows
        self.hidden_rows = hidden_rows
        self.calls: list[int | None] = []

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int | None = None,
        limit: int = 1000,
    ) -> list[list]:
        self.calls.append(since_ms)
        rows = self.full_rows if len(self.calls) > 1 else [
            r for r in self.full_rows if r not in self.hidden_rows
        ]
        return [r for r in rows if r[0] >= since_ms][:limit]


class TestPagination:
    def test_fetches_all_pages_with_correct_since(self) -> None:
        rows = make_candles(2400)  # 2400 hours = exactly 100 days
        exchange = FakeExchange(rows)
        downloader = HistoryDownloader(exchange)

        df = downloader.download(
            "BTC/USDT", "1h", since="2025-08-01", until="2025-11-09"
        )

        assert len(df) == 2400
        assert len(exchange.calls) == 3  # 1000 + 1000 + 400, no empty trailing call
        assert exchange.calls[0] == BASE_MS
        assert exchange.calls[1] == rows[1000][0]
        assert exchange.calls[2] == rows[2000][0]
        assert df["timestamp"].is_monotonic_increasing

    def test_respects_until_boundary(self) -> None:
        exchange = FakeExchange(make_candles(48))
        downloader = HistoryDownloader(exchange)

        df = downloader.download("BTC/USDT", "1h", since="2025-08-01", until="2025-08-02")

        assert len(df) == 24
        assert exchange.calls == [BASE_MS]

    def test_no_data_raises_value_error(self) -> None:
        downloader = HistoryDownloader(FakeExchange([]))

        with pytest.raises(ValueError, match="no candles"):
            downloader.download("BTC/USDT", "1h", since="2025-08-01", until="2025-08-02")

    def test_since_after_until_raises(self) -> None:
        downloader = HistoryDownloader(FakeExchange(make_candles(5)))

        with pytest.raises(ValueError, match="before until"):
            downloader.download("BTC/USDT", "1h", since="2025-08-10", until="2025-08-01")


class TestRetry:
    @pytest.mark.parametrize(
        "error",
        [ccxt.NetworkError("boom"), ccxt.RateLimitExceeded("limited")],
    )
    def test_retries_then_succeeds(self, mocker, error) -> None:
        exchange = FakeExchange(
            make_candles(5), errors=[type(error)("fail 1"), type(error)("fail 2")]
        )
        sleep = mocker.patch("trading_bot.data.downloader.time.sleep")
        downloader = HistoryDownloader(exchange)

        df = downloader.download("BTC/USDT", "1h", since="2025-08-01", until="2025-08-02")

        assert len(df) == 5
        # Exponential backoff: 1s after the first failure, 2s after the second.
        sleep.assert_has_calls([mocker.call(1), mocker.call(2)])

    def test_gives_up_after_max_attempts(self) -> None:
        exchange = FakeExchange(make_candles(5), errors=[ccxt.NetworkError("down")] * 99)
        downloader = HistoryDownloader(exchange)

        with pytest.raises(ccxt.NetworkError):
            downloader.download("BTC/USDT", "1h", since="2025-08-01", until="2025-08-02")

        assert len(exchange.calls) == MAX_ATTEMPTS


class TestGapBackfill:
    def test_backfills_missing_range(self) -> None:
        full = make_candles(96)
        hidden = full[40:45]
        exchange = GapExchange(full, hidden)
        downloader = HistoryDownloader(exchange)

        df = downloader.download("BTC/USDT", "1h", since="2025-08-01", until="2025-08-05")

        # The main pass saw 91 candles (96 minus the 5 hidden, bounded by until);
        # the backfill request must start at the first missing candle.
        assert len(df) == 96
        assert len(exchange.calls) == 2
        assert exchange.calls[0] == BASE_MS
        assert exchange.calls[1] == hidden[0][0]
        assert validate_ohlcv(df, "1h") == []

    def test_unfixable_gap_raises_value_error(self) -> None:
        rows = make_candles(10)
        rows = rows[:4] + rows[6:]
        downloader = HistoryDownloader(FakeExchange(rows))

        with pytest.raises(ValueError, match="gap from"):
            downloader.download("BTC/USDT", "1h", since="2025-08-01", until="2025-08-02")


class TestUpdate:
    def test_appends_from_last_candle_to_now(self, tmp_path) -> None:
        storage = CandleStorage(tmp_path)
        stored = make_candles(50)
        storage.save("bybit", "BTC/USDT", "1h", rows_to_df(stored))
        exchange = FakeExchange(make_candles(80))
        downloader = HistoryDownloader(exchange)

        df = downloader.update("BTC/USDT", "1h", storage)

        assert len(df) == 80
        # The last stored candle is re-fetched (it may have been still open).
        assert exchange.calls[0] == stored[-1][0]
        pd.testing.assert_frame_equal(storage.load("bybit", "BTC/USDT", "1h"), df)

    def test_raises_without_stored_data(self, tmp_path) -> None:
        storage = CandleStorage(tmp_path)
        downloader = HistoryDownloader(FakeExchange(make_candles(10)))

        with pytest.raises(FileNotFoundError, match="full download"):
            downloader.update("BTC/USDT", "1h", storage)

    def test_open_tail_is_dropped_before_saving(self, tmp_path, mocker) -> None:
        # now is inside the bucket of the candle at BASE + 50h: it and every
        # later candle are still open and must not reach the dataset.
        mocker.patch(
            "trading_bot.data.downloader._now_ms",
            return_value=BASE_MS + 50 * HOUR_MS + HOUR_MS // 2,
        )
        storage = CandleStorage(tmp_path)
        stored_df = rows_to_df(make_candles(50))
        storage.save("bybit", "BTC/USDT", "1h", stored_df)
        exchange = FakeExchange(make_candles(80))
        downloader = HistoryDownloader(exchange)

        df = downloader.update("BTC/USDT", "1h", storage)

        # No *closed* candles beyond the stored 50: the dataset is unchanged.
        assert len(df) == 50
        pd.testing.assert_frame_equal(
            storage.load("bybit", "BTC/USDT", "1h"), stored_df
        )

    def test_validation_problems_after_update_raise(self, tmp_path) -> None:
        storage = CandleStorage(tmp_path)
        storage.save("bybit", "BTC/USDT", "1h", rows_to_df(make_candles(10)))
        # The tail introduces a gap: candles 10-11 are missing, 12+ arrive.
        gappy = make_candles(20)[12:]
        downloader = HistoryDownloader(FakeExchange(gappy))

        with pytest.raises(ValueError, match=r"(?s)after update.*gap from"):
            downloader.update("BTC/USDT", "1h", storage)


class TestOpenCandleFilter:
    def test_download_drops_candle_whose_bucket_is_still_open(self, mocker) -> None:
        # now is inside the bucket of the last candle (BASE + 4h): it closes
        # at BASE + 5h and must not be returned (and thus not saved).
        mocker.patch(
            "trading_bot.data.downloader._now_ms",
            return_value=BASE_MS + 4 * HOUR_MS + HOUR_MS // 2,
        )
        exchange = FakeExchange(make_candles(5))
        downloader = HistoryDownloader(exchange)

        df = downloader.download("BTC/USDT", "1h", since="2025-08-01")

        assert len(df) == 4
        last_ms = int(df["timestamp"].iloc[-1].value // 1_000_000)
        assert last_ms == BASE_MS + 3 * HOUR_MS

    def test_download_closed_history_is_untouched(self, mocker) -> None:
        # now is far beyond the last candle: nothing is dropped.
        mocker.patch(
            "trading_bot.data.downloader._now_ms",
            return_value=BASE_MS + 100 * HOUR_MS,
        )
        exchange = FakeExchange(make_candles(5))
        downloader = HistoryDownloader(exchange)

        df = downloader.download("BTC/USDT", "1h", since="2025-08-01")

        assert len(df) == 5

    def test_download_with_only_open_candles_raises(self, mocker) -> None:
        # The single available candle is still open: nothing closed to store.
        mocker.patch(
            "trading_bot.data.downloader._now_ms",
            return_value=BASE_MS + HOUR_MS // 2,
        )
        exchange = FakeExchange(make_candles(1))
        downloader = HistoryDownloader(exchange)

        with pytest.raises(ValueError, match="still open"):
            downloader.download("BTC/USDT", "1h", since="2025-08-01")
