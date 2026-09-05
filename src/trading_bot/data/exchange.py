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
    """Обёртка над ``ccxt.bybit``: публичные рыночные данные + опциональная торговля.

    Без ключей клиент работает только с публичным API (OHLCV, тикер) — этого
    достаточно и downloader'у, и paper-режиму live-раннера. Ключи передаются
    явно (live-CLI берёт их из переменных окружения, никогда не из конфига и
    не из git) и включают приватные методы: размещение ордеров, их статус и
    баланс. ``sandbox=True`` переключает ccxt на testnet-эндпоинт Bybit.
    Ограничение частоты запросов делегировано ccxt через ``enableRateLimit``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret: str | None = None,
        sandbox: bool = False,
    ) -> None:
        params: dict = {"enableRateLimit": True}
        if api_key and secret:
            params["apiKey"] = api_key
            params["secret"] = secret
            params["options"] = {"defaultType": "spot"}
        self.exchange = ccxt.bybit(params)
        if sandbox:
            self.exchange.set_sandbox_mode(True)

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

    def fetch_ticker_last(self, symbol: str) -> float:
        """Получить последнюю цену тикера (публичный API).

        Raises:
            ValueError: если биржа не вернула цену ``last``.
        """
        ticker = self.exchange.fetch_ticker(symbol)
        last = ticker.get("last") if ticker else None
        if last is None:
            raise ValueError(f"ticker for {symbol} has no 'last' price")
        return float(last)

    def create_market_order(self, symbol: str, side: str, quantity: float) -> dict:
        """Разместить рыночный ордер (приватный API, только testnet-режим).

        Raises:
            ccxt.ExchangeError: если биржа отклонила ордер.
        """
        return self.exchange.create_order(symbol, "market", side, quantity)

    def fetch_order(self, order_id: str, symbol: str) -> dict:
        """Получить статус ордера (приватный API): средняя цена, объём, комиссия."""
        return self.exchange.fetch_order(order_id, symbol)

    def fetch_free_balance(self, currency: str = "USDT") -> float:
        """Получить свободный баланс валюты (приватный API)."""
        balance = self.exchange.fetch_balance()
        free = balance.get("free") or {}
        return float(free.get(currency, 0.0))


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
