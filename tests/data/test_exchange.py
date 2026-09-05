"""Tests for the exchange client wrapper."""

from __future__ import annotations

import pytest

from trading_bot.data.exchange import ExchangeClient, timeframe_to_ms


def test_fetch_ohlcv_delegates_to_ccxt(mocker) -> None:
    bybit_cls = mocker.patch("trading_bot.data.exchange.ccxt.bybit")
    client = ExchangeClient()
    bybit_cls.assert_called_once_with({"enableRateLimit": True})

    client.exchange.fetch_ohlcv.return_value = [[1000, 1.0, 2.0, 0.5, 1.5, 42.0]]
    rows = client.fetch_ohlcv("BTC/USDT", "4h", since_ms=123, limit=500)

    assert rows == [[1000, 1.0, 2.0, 0.5, 1.5, 42.0]]
    client.exchange.fetch_ohlcv.assert_called_once_with(
        "BTC/USDT", timeframe="4h", since=123, limit=500
    )


@pytest.mark.parametrize(
    ("timeframe", "expected_ms"),
    [
        ("15m", 900_000),
        ("1h", 3_600_000),
        ("4h", 14_400_000),
        ("1d", 86_400_000),
        ("2w", 1_209_600_000),
    ],
)
def test_timeframe_to_ms(timeframe: str, expected_ms: int) -> None:
    assert timeframe_to_ms(timeframe) == expected_ms


@pytest.mark.parametrize("bad", ["", "m", "4x", "h4", "0h", "1M", "1y"])
def test_timeframe_to_ms_rejects_invalid(bad: str) -> None:
    with pytest.raises(ValueError, match="timeframe"):
        timeframe_to_ms(bad)
