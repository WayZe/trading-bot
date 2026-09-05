"""Тесты Parquet-хранилища свечей и валидации OHLCV."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.conftest import BASE_MS, HOUR_MS, make_candles, rows_to_df
from trading_bot.data.storage import (
    CandleStorage,
    find_gaps,
    symbol_to_slug,
    validate_ohlcv,
)


@pytest.fixture()
def storage(tmp_path):
    return CandleStorage(tmp_path)


class TestStoragePaths:
    def test_symbol_to_slug(self) -> None:
        assert symbol_to_slug("BTC/USDT") == "BTC_USDT"
        assert symbol_to_slug("ETH/USDT") == "ETH_USDT"

    def test_symbol_to_slug_rejects_traversal_and_empty_parts(self) -> None:
        for bad in ("", "..", "../BTC", "BTC/../USDT", "BTC//USDT", "/BTC"):
            with pytest.raises(ValueError, match="invalid symbol"):
                symbol_to_slug(bad)

    def test_path_for(self, storage, tmp_path) -> None:
        path = storage.path_for("bybit", "BTC/USDT", "4h")
        assert path == tmp_path / "bybit" / "BTC_USDT" / "4h.parquet"


class TestSaveLoad:
    def test_roundtrip_preserves_schema(self, storage) -> None:
        df = rows_to_df(make_candles(10))
        storage.save("bybit", "BTC/USDT", "1h", df)

        loaded = storage.load("bybit", "BTC/USDT", "1h")
        assert loaded is not None
        pd.testing.assert_frame_equal(loaded, df)
        assert str(loaded["timestamp"].dtype) == "datetime64[ms, UTC]"
        for col in ("open", "high", "low", "close", "volume"):
            assert str(loaded[col].dtype) == "float64"

    def test_load_missing_returns_none(self, storage) -> None:
        assert storage.load("bybit", "BTC/USDT", "1h") is None

    def test_save_is_atomic_no_tmp_files_left(self, storage) -> None:
        storage.save("bybit", "BTC/USDT", "1h", rows_to_df(make_candles(5)))
        directory = storage.path_for("bybit", "BTC/USDT", "1h").parent
        assert sorted(p.name for p in directory.iterdir()) == ["1h.parquet"]

    def test_save_sorts_and_deduplicates(self, storage) -> None:
        rows = make_candles(5)
        shuffled = [rows[2], rows[0], rows[1], rows[2]]
        storage.save("bybit", "BTC/USDT", "1h", rows_to_df(shuffled))
        loaded = storage.load("bybit", "BTC/USDT", "1h")
        assert len(loaded) == 3
        assert loaded["timestamp"].is_monotonic_increasing


class TestAppend:
    def test_merges_overlapping_ranges_without_duplicates(self, storage) -> None:
        storage.save("bybit", "BTC/USDT", "1h", rows_to_df(make_candles(10)))
        overlapping = rows_to_df(make_candles(5, start_ms=BASE_MS + 5 * HOUR_MS))

        merged = storage.append("bybit", "BTC/USDT", "1h", overlapping)

        assert len(merged) == 10
        assert not merged["timestamp"].duplicated().any()
        assert merged["timestamp"].is_monotonic_increasing
        pd.testing.assert_frame_equal(storage.load("bybit", "BTC/USDT", "1h"), merged)

    def test_overwrites_last_candle(self, storage) -> None:
        storage.save("bybit", "BTC/USDT", "1h", rows_to_df(make_candles(5)))
        refreshed = make_candles(1, start_ms=BASE_MS + 4 * HOUR_MS)
        refreshed[0][5] = 999.0  # новый объём для той же метки времени

        merged = storage.append("bybit", "BTC/USDT", "1h", rows_to_df(refreshed))

        assert len(merged) == 5
        assert merged.iloc[-1]["volume"] == 999.0

    def test_creates_file_when_nothing_stored(self, storage) -> None:
        merged = storage.append("bybit", "BTC/USDT", "1h", rows_to_df(make_candles(3)))
        assert len(merged) == 3
        assert storage.load("bybit", "BTC/USDT", "1h") is not None


class TestValidate:
    def test_valid_data_has_no_problems(self) -> None:
        assert validate_ohlcv(rows_to_df(make_candles(24)), "1h") == []

    def test_empty_dataframe_is_ok(self) -> None:
        assert validate_ohlcv(rows_to_df([]), "1h") == []

    def test_detects_high_below_body_and_low_above_body(self) -> None:
        rows = make_candles(3)
        rows[1][2] = 1.0  # high ниже max(open, close)
        rows[1][3] = 500.0  # low выше min(open, close)

        problems = validate_ohlcv(rows_to_df(rows), "1h")

        assert any("high < max(open, close)" in p for p in problems)
        assert any("low > min(open, close)" in p for p in problems)

    def test_detects_gap(self) -> None:
        rows = make_candles(10)
        rows = rows[:4] + rows[6:]  # выбрасываем свечи 4 и 5

        problems = validate_ohlcv(rows_to_df(rows), "1h")

        assert len(problems) == 1
        assert problems[0].startswith("gap from ")

    def test_detects_irregular_short_interval(self) -> None:
        rows = make_candles(3)
        rows[2][0] = rows[1][0] + 60_000  # 1 минута вместо 1 часа

        problems = validate_ohlcv(rows_to_df(rows), "1h")

        assert any("irregular interval" in p for p in problems)

    def test_detects_nan(self) -> None:
        rows = make_candles(3)
        rows[1][4] = float("nan")

        problems = validate_ohlcv(rows_to_df(rows), "1h")

        assert any("NaN" in p and "'close'" in p for p in problems)

    def test_detects_duplicate_timestamps(self) -> None:
        rows = make_candles(3) + make_candles(1)

        problems = validate_ohlcv(rows_to_df(rows), "1h")

        assert any("duplicate" in p for p in problems)

    def test_detects_unsorted_timestamps(self) -> None:
        rows = list(reversed(make_candles(5)))

        problems = validate_ohlcv(rows_to_df(rows), "1h")

        assert any("not in ascending order" in p for p in problems)

    def test_detects_negative_volume(self) -> None:
        rows = make_candles(2)
        rows[1][5] = -1.0

        problems = validate_ohlcv(rows_to_df(rows), "1h")

        assert any("volume < 0" in p for p in problems)


class TestFindGaps:
    def test_returns_gap_boundaries(self) -> None:
        rows = make_candles(10)
        rows = rows[:4] + rows[7:]

        gaps = find_gaps(rows_to_df(rows), "1h")

        assert len(gaps) == 1
        before, after = gaps[0]
        assert before.value // 1_000_000 == rows[3][0]
        assert after.value // 1_000_000 == rows[4][0]

    def test_no_gaps_in_continuous_data(self) -> None:
        assert find_gaps(rows_to_df(make_candles(50)), "1h") == []
