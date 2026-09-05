"""Плагины стратегий и реестр стратегий."""

from __future__ import annotations

from trading_bot.strategy.base import Fill, Signal, SignalKind, Strategy
from trading_bot.strategy.sma_cross import SmaCrossStrategy

STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "sma_cross": SmaCrossStrategy,
}


def create_strategy(name: str, params: dict | None = None) -> Strategy:
    """Создать экземпляр плагина стратегии по имени из реестра.

    Raises:
        ValueError: если имя неизвестно или параметры не соответствуют
            сигнатуре конструктора стратегии.
    """
    strategy_cls = STRATEGY_REGISTRY.get(name)
    if strategy_cls is None:
        available = ", ".join(sorted(STRATEGY_REGISTRY))
        raise ValueError(f"unknown strategy {name!r}; available: {available}")
    try:
        return strategy_cls(**(params or {}))
    except TypeError as error:
        raise ValueError(f"invalid strategy_params for {name!r}: {error}") from error


__all__ = [
    "STRATEGY_REGISTRY",
    "Fill",
    "Signal",
    "SignalKind",
    "SmaCrossStrategy",
    "Strategy",
    "create_strategy",
]
