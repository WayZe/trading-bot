"""Плагины стратегий и реестр стратегий."""

from __future__ import annotations

from collections.abc import Callable

from trading_bot.strategy.base import Fill, Signal, SignalKind, Strategy
from trading_bot.strategy.donchian import DonchianBreakoutStrategy
from trading_bot.strategy.sma_cross import SmaCrossStrategy
from trading_bot.strategy.trend_filter import TrendFiltered

# Запись реестра: класс стратегии либо фабрика-функция (для композитных
# стратегий, чьи параметры делятся между внутренней стратегией и обёрткой).
StrategyFactory = type[Strategy] | Callable[..., Strategy]

# Параметры, уходящие во внутреннюю SmaCrossStrategy фабрики sma_cross_trend;
# остальные параметры передаются обёртке TrendFiltered.
_SMA_CROSS_PARAM_KEYS = ("fast", "slow", "atr_period", "atr_mult")

# Параметры, уходящие во внутреннюю DonchianBreakoutStrategy фабрики
# donchian_trend; остальные параметры передаются обёртке TrendFiltered.
_DONCHIAN_PARAM_KEYS = ("entry_period", "exit_period", "atr_period", "atr_mult")


def _create_sma_cross_trend(**params) -> Strategy:
    """Собрать ``sma_cross_trend``: SmaCrossStrategy внутри тренд-фильтра.

    Параметры ``fast``/``slow``/``atr_period``/``atr_mult`` уходят во
    внутреннюю стратегию, остальные (``trend_period``, ``trend_source``) —
    в обёртку :class:`TrendFiltered`.
    """
    inner_params = {key: params.pop(key) for key in _SMA_CROSS_PARAM_KEYS if key in params}
    return TrendFiltered(SmaCrossStrategy(**inner_params), **params)


def _create_donchian_trend(**params) -> Strategy:
    """Собрать ``donchian_trend``: DonchianBreakoutStrategy внутри тренд-фильтра.

    Параметры ``entry_period``/``exit_period``/``atr_period``/``atr_mult``
    уходят во внутреннюю стратегию, остальные (``trend_period``,
    ``trend_source``) — в обёртку :class:`TrendFiltered`.
    """
    inner_params = {key: params.pop(key) for key in _DONCHIAN_PARAM_KEYS if key in params}
    return TrendFiltered(DonchianBreakoutStrategy(**inner_params), **params)


STRATEGY_REGISTRY: dict[str, StrategyFactory] = {
    "sma_cross": SmaCrossStrategy,
    "sma_cross_trend": _create_sma_cross_trend,
    "donchian": DonchianBreakoutStrategy,
    "donchian_trend": _create_donchian_trend,
}


def create_strategy(name: str, params: dict | None = None) -> Strategy:
    """Создать экземпляр плагина стратегии по имени из реестра.

    Raises:
        ValueError: если имя неизвестно или параметры не соответствуют
            сигнатуре конструктора стратегии (фабрики).
    """
    factory = STRATEGY_REGISTRY.get(name)
    if factory is None:
        available = ", ".join(sorted(STRATEGY_REGISTRY))
        raise ValueError(f"unknown strategy {name!r}; available: {available}")
    try:
        return factory(**(params or {}))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid strategy_params for {name!r}: {error}") from error


__all__ = [
    "STRATEGY_REGISTRY",
    "DonchianBreakoutStrategy",
    "Fill",
    "Signal",
    "SignalKind",
    "SmaCrossStrategy",
    "Strategy",
    "StrategyFactory",
    "TrendFiltered",
    "create_strategy",
]
