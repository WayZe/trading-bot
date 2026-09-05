"""Thin wrapper around the ccxt Bybit exchange client."""

from __future__ import annotations

import ccxt

_TIMEFRAME_UNITS_MS: dict[str, int] = {
    "m": 60_000,
    "h": 3_600_000,
    "d": 86_400_000,
    "w": 604_800_000,
}


class ExchangeClient:
    """Minimal wrapper over ``ccxt.bybit`` limited to public OHLCV access.

    Only public market data is used, so no API keys are required.
    Rate limiting is delegated to ccxt via ``enableRateLimit``.
    """

    def __init__(self) -> None:
        self.exchange = ccxt.bybit({"enableRateLimit": True})

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int | None = None,
        limit: int = 1000,
    ) -> list[list]:
        """Fetch raw OHLCV candles from Bybit.

        Args:
            symbol: ccxt symbol, e.g. ``"BTC/USDT"``.
            timeframe: ccxt timeframe, e.g. ``"15m"``, ``"4h"``, ``"1d"``.
            since_ms: fetch candles starting at this timestamp (ms since epoch).
            limit: maximum number of candles per request.

        Returns:
            Raw candle rows ``[timestamp_ms, open, high, low, close, volume]``.
        """
        return self.exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=since_ms, limit=limit
        )


def timeframe_to_ms(timeframe: str) -> int:
    """Convert a ccxt-style timeframe string to its duration in milliseconds.

    Supported units: ``m`` (minute), ``h`` (hour), ``d`` (day), ``w`` (week).
    Variable-length units (month, year) are not supported.

    Raises:
        ValueError: if the timeframe is not a supported ``<count><unit>`` string.
    """
    if len(timeframe) < 2:
        raise ValueError(f"invalid timeframe: {timeframe!r}")
    unit = timeframe[-1]
    if unit not in _TIMEFRAME_UNITS_MS:
        raise ValueError(f"unsupported timeframe unit in {timeframe!r}: {unit!r}")
    try:
        count = int(timeframe[:-1])
    except ValueError as exc:
        raise ValueError(f"invalid timeframe: {timeframe!r}") from exc
    if count < 1:
        raise ValueError(f"invalid timeframe: {timeframe!r}")
    return count * _TIMEFRAME_UNITS_MS[unit]
