"""Тонкая обёртка над биржевым клиентом ccxt.bybit."""

from __future__ import annotations

import ccxt

_TIMEFRAME_UNITS_MS: dict[str, int] = {
    "m": 60_000,
    "h": 3_600_000,
    "d": 86_400_000,
    "w": 604_800_000,
}


class ExchangeClient:
    """Минимальная обёртка над ``ccxt.bybit``, ограниченная публичным доступом к OHLCV.

    Используются только публичные рыночные данные, поэтому API-ключи не нужны.
    Ограничение частоты запросов делегировано ccxt через ``enableRateLimit``.
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
        """Получить сырые OHLCV-свечи с Bybit.

        Args:
            symbol: символ в формате ccxt, напр. ``"BTC/USDT"``.
            timeframe: таймфрейм в формате ccxt, напр. ``"15m"``, ``"4h"``, ``"1d"``.
            since_ms: получать свечи начиная с этого момента времени (мс от эпохи).
            limit: максимальное число свечей в одном запросе.

        Returns:
            Сырые строки свечей ``[timestamp_ms, open, high, low, close, volume]``.
        """
        return self.exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=since_ms, limit=limit
        )


def timeframe_to_ms(timeframe: str) -> int:
    """Перевести строку таймфрейма в стиле ccxt в длительность в миллисекундах.

    Поддерживаемые единицы: ``m`` (минута), ``h`` (час), ``d`` (день),
    ``w`` (неделя). Единицы переменной длины (месяц, год) не поддерживаются.

    Raises:
        ValueError: если таймфрейм — не поддерживаемая строка ``<count><unit>``.
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
