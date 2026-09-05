"""Единая точка правды для активных уровней стоп-лосс/тейк-профит.

Инвариант «движок — единственный владелец SL/TP» одинаков для бэктеста и
live-торговли: стратегия задаёт уровни как *дистанции* от close сигнальной
свечи, а исполнитель (``engine.backtest`` или ``live.runner``) после
фактического исполнения входа переносит эти дистанции на цену исполнения.
Чтобы live-стоп никогда не разошёлся с бэктест-стопом на том же сценарии,
переякоривание вынесено в публичные функции ниже и используется обоими
исполнителями (parity-тест в ``tests/live/test_runner.py`` следит за этим).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Причины выхода по защитным уровням; попадают в записи сделок и логи.
REASON_STOP_LOSS = "stop loss"
REASON_TAKE_PROFIT = "take profit"


def reanchor_stop_below(
    level: float | None, ref_close: float | None, fill_price: float
) -> float | None:
    """Переякорить уровень стопа на фактическую цену исполнения.

    Стратегия задаёт дистанцию стопа относительно close сигнальной свечи
    (``ref_close - level``); исполнитель переносит эту дистанцию на
    фактическую цену входа, чтобы стоп следовал за реально заплаченной
    ценой. Отсутствие или неположительность дистанции отключает уровень.

    Args:
        level: уровень стопа относительно ``ref_close`` (``None`` — нет стопа).
        ref_close: close сигнальной свечи, от которого задан уровень.
        fill_price: фактическая цена исполнения входа.

    Returns:
        Активный уровень стопа или ``None``, если уровень отключён.
    """
    if level is None or ref_close is None:
        return None
    distance = ref_close - level
    if distance <= 0.0:
        logger.warning(
            "stop level %.4f is not below the signal close %.4f: stop disabled",
            level,
            ref_close,
        )
        return None
    return fill_price - distance


def reanchor_take_profit_above(
    level: float | None, ref_close: float | None, fill_price: float
) -> float | None:
    """Переякорить уровень тейк-профита на фактическую цену исполнения (симметрично).

    Args:
        level: уровень тейк-профита относительно ``ref_close`` (``None`` — нет тейка).
        ref_close: close сигнальной свечи, от которого задан уровень.
        fill_price: фактическая цена исполнения входа.

    Returns:
        Активный уровень тейк-профита или ``None``, если уровень отключён.
    """
    if level is None or ref_close is None:
        return None
    distance = level - ref_close
    if distance <= 0.0:
        logger.warning(
            "take-profit level %.4f is not above the signal close %.4f: "
            "take-profit disabled",
            level,
            ref_close,
        )
        return None
    return fill_price + distance
