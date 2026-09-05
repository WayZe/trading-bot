"""Walk-forward анализ: переоптимизация на ин-семпл, проверка на аут-оф-семпл.

История делится на последовательные окна из ин-семпл (IS) отрезка, за которым
следует аут-оф-семпл (OOS) отрезок. Для каждого окна:

1. сетка параметров прогоняется по IS-отрезку (переиспользуя
   :func:`run_sweep`);
2. выбирается лучшая валидная комбинация по ``objective``;
3. эта комбинация один раз прогоняется по OOS-отрезку, перед которым идёт
   *lead-in* из ``strategy.warmup_period`` свечей, чтобы индикаторы
   разогрелись на данных до начала OOS (движок стартует каждое окно с
   пустого портфеля, поэтому позиция в lead-in может только *открыться*);
4. в OOS-метрики и сшитую кривую идёт только часть эквити с ``oos_start`` —
   PnL, заработанный до ``oos_start``, отбрасывается. Сделка, открытая в
   lead-in, но закрытая внутри OOS-отрезка, всё же вносит свою OOS-часть в
   эквити/доходность, но исключена из trade-метрик — из-за этого окно может
   показать ненулевую доходность при нуле записанных сделок.

Сшитая кривая — конкатенация OOS-частей эквити по окнам, каждая нормирована
к своему первому значению и умножена на накопленный рост: оценка того, что
стратегия дала бы при периодической переоптимизации, без заглядывания в
будущее. Календарные разрывы между окнами становятся нулевыми доходностями
в ``pct_change``, что слегка искажает аннуализированный Шарп сшитой кривой —
сознательное упрощение.

Чистые функции над свечами; весь I/O (артефакты, графики) живёт в CLI.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from trading_bot.config import BacktestConfig
from trading_bot.data.exchange import timeframe_to_ms
from trading_bot.engine.backtest import build_engine
from trading_bot.engine.portfolio import TRADE_RECORD_FIELDS
from trading_bot.report.metrics import compute_metrics
from trading_bot.research.sweep import MAX_COMBINATIONS, run_sweep
from trading_bot.strategy import create_strategy

logger = logging.getLogger(__name__)

# Предохранительный потолок: walk-forward гоняет полный sweep на каждое окно;
# слишком много окон — значит, пользователь скорее всего запросил слишком
# малые OOS-дни для этого датасета.
MAX_WINDOWS = 50

MODES = ("rolling", "anchored")
OBJECTIVES = ("sharpe", "total_return_pct")

# Колонки строки результата walk-forward (после колонок параметров):
# имена параметров сетки, совпадающие с ними, испортили бы results.csv,
# поэтому они отбрасываются на этапе валидации ``--param``.
WINDOW_COLUMNS = (
    "is_start",
    "is_end",
    "oos_start",
    "oos_end",
    "mode",
    "is_objective",
    "oos_return_pct",
    "oos_sharpe",
    "oos_max_dd_pct",
    "oos_trades",
    "error",
)

_DAY = pd.Timedelta(days=1)


@dataclass(frozen=True)
class Window:
    """Одно окно walk-forward; ``is_end == oos_start`` (без разрывов и перекрытий)."""

    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp


@dataclass
class WindowResult:
    """Исход одного окна walk-forward.

    ``best_params``/``is_objective`` равны ``None``, а ``error`` несёт причину,
    если окно упало (нет валидных IS-комбинаций или ошибка OOS-прогона);
    метрики ``oos_*`` равны ``None`` для упавших окон.
    """

    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    mode: str
    best_params: dict | None = None
    is_objective: float | None = None
    oos_return_pct: float | None = None
    oos_sharpe: float | None = None
    oos_max_dd_pct: float | None = None
    oos_trades: int | None = None
    error: str | None = None


@dataclass
class WalkForwardResult:
    """Артефакты прогона walk-forward.

    Attributes:
        windows: по одному :class:`WindowResult` на запланированное окно,
            хронологически, включая упавшие окна (они просто не дают
            OOS-данных).
        stitched_equity: конкатенация OOS-частей эквити, каждая нормирована
            к своему первому значению и умножена на накопленный рост;
            начинается с 1.0. Пуста, если ни одно окно не выполнилось.
        trades: все OOS-сделки по всем окнам (схема ``TradeRecord``).
        meta: JSON-совместимые параметры прогона (mode, дни, objective,
            сетка, ...).
    """

    windows: list[WindowResult] = field(default_factory=list)
    stitched_equity: pd.Series = field(
        default_factory=lambda: pd.Series(
            dtype="float64", name="equity", index=pd.DatetimeIndex([], tz="UTC")
        )
    )
    trades: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=TRADE_RECORD_FIELDS)
    )
    meta: dict[str, Any] = field(default_factory=dict)

    def to_frame(self, param_grid: dict[str, list]) -> pd.DataFrame:
        """Записи окон как фрейм (параметры сетки, даты, метрики, ошибка)."""
        rows = []
        for window in self.windows:
            row: dict[str, Any] = dict.fromkeys(param_grid, math.nan)
            for name, value in (window.best_params or {}).items():
                row[name] = value
            row.update(
                is_start=window.is_start.isoformat(),
                is_end=window.is_end.isoformat(),
                oos_start=window.oos_start.isoformat(),
                oos_end=window.oos_end.isoformat(),
                mode=window.mode,
                is_objective=window.is_objective,
                oos_return_pct=window.oos_return_pct,
                oos_sharpe=window.oos_sharpe,
                oos_max_dd_pct=window.oos_max_dd_pct,
                oos_trades=window.oos_trades,
                error=window.error,
            )
            rows.append(row)
        return pd.DataFrame(rows, columns=[*param_grid, *WINDOW_COLUMNS])


def plan_windows(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    is_days: int,
    oos_days: int,
    mode: str = "rolling",
) -> list[Window]:
    """Спланировать walk-forward окна по ``[start_ts, end_ts)``.

    Границы — в календарных днях.

    ``rolling``: блоки IS/OOS фиксированной длины выкладываются на таймлайн
    встык — новый IS начинается там, где закончился предыдущий OOS, поэтому
    оптимизатор никогда не видит OOS-данные предыдущего окна (OOS-отрезки
    разнесены на ``is_days``).

    ``anchored``: ``is_start`` остаётся в общем начале, а IS-отрезок растёт
    на ``oos_days`` за каждый шаг; OOS-отрезки выкладываются по хвосту
    без перекрытий.

    Окно планируется, только если его OOS-отрезок заканчивается не позже
    ``end_ts``.

    Args:
        start_ts: общее начало (включительно), tz-aware UTC.
        end_ts: общий конец (не включительно), tz-aware UTC.
        is_days: длина ин-семпл отрезка в днях; должна быть положительной.
        oos_days: длина аут-оф-семпл отрезка в днях; должна быть положительной.
        mode: ``"rolling"`` или ``"anchored"``.

    Returns:
        Окна в хронологическом порядке; каждое окно удовлетворяет
        ``is_end == oos_start``.

    Raises:
        ValueError: если ``is_days``/``oos_days`` неположительны, ``mode``
            неизвестен, протяжённость данных меньше ``is_days + oos_days``
            (ни одно окно не помещается) или окон получилось бы больше,
            чем :data:`MAX_WINDOWS`.
    """
    if is_days <= 0:
        raise ValueError(f"is_days must be positive, got {is_days}")
    if oos_days <= 0:
        raise ValueError(f"oos_days must be positive, got {oos_days}")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; available: {', '.join(MODES)}")

    start_ts = pd.Timestamp(start_ts)
    end_ts = pd.Timestamp(end_ts)
    is_delta = pd.Timedelta(days=is_days)
    oos_delta = pd.Timedelta(days=oos_days)

    windows: list[Window] = []
    step = 0
    while True:
        if mode == "rolling":
            is_start = start_ts + step * (is_delta + oos_delta)
            is_end = is_start + is_delta
        else:  # anchored
            is_start = start_ts
            is_end = start_ts + is_delta + step * oos_delta
        oos_start = is_end
        oos_end = oos_start + oos_delta
        if oos_end > end_ts:
            break
        windows.append(Window(is_start, is_end, oos_start, oos_end))
        step += 1

    if not windows:
        available = (end_ts - start_ts) / _DAY
        raise ValueError(
            f"data span {available:.1f} days is shorter than the walk-forward "
            f"requirement of {is_days + oos_days} days "
            f"(is_days={is_days} + oos_days={oos_days}); "
            "download more history or reduce --is-days/--oos-days"
        )
    if len(windows) > MAX_WINDOWS:
        raise ValueError(
            f"{len(windows)} walk-forward windows exceed the limit of "
            f"{MAX_WINDOWS}; increase --oos-days or narrow the data range"
        )
    return windows


def run_walkforward(
    candles: pd.DataFrame,
    base_config: BacktestConfig,
    param_grid: dict[str, list],
    *,
    is_days: int,
    oos_days: int,
    mode: str = "rolling",
    objective: str = "sharpe",
) -> WalkForwardResult:
    """Прогнать walk-forward: оптимизация на IS в каждом окне -> проверка на OOS.

    Окна приходят из :func:`plan_windows` по доступному диапазону свечей.
    IS-sweep переиспользует :func:`run_sweep` (включая семантику строк с
    ошибками) через копию ``base_config`` с изменённым периодом; лучшая
    валидная строка по ``objective`` (значения NaN/None неранжируемы) затем
    один раз прогоняется по OOS-отрезку с lead-in из ``strategy.warmup_period``
    свечей перед ``oos_start`` (меньше, если данные не дотягиваются назад).
    OOS-метрики считаются по части эквити с ``oos_start`` и сделкам со входом
    с ``oos_start``, так что PnL до ``oos_start`` сознательно отбрасывается;
    движок стартует каждое окно с пустого портфеля, поэтому lead-in может
    только открыть позиции, а сделка из lead-in, закрытая внутри OOS, вносит
    вклад в эквити, но не в trade-метрики.

    Окно, чей IS-sweep упал (пустой IS-срез из-за разрыва в данных,
    невалидные свечи, ...), становится строкой с ошибкой; остальные окна
    всё равно выполняются.

    Raises:
        ValueError: если ``candles`` пуст, ``objective`` неизвестен, сетка
            превышает :data:`~trading_bot.research.sweep.MAX_COMBINATIONS`
            (потолок один и тот же для каждого окна, поэтому проверяется
            один раз заранее) или не удалось спланировать окна
            (см. :func:`plan_windows`).
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; available: {', '.join(OBJECTIVES)}")
    if candles.empty:
        raise ValueError("no candles: the dataset is empty")
    n_combinations = math.prod(len(values) for values in param_grid.values()) or 1
    if n_combinations > MAX_COMBINATIONS:
        raise ValueError(
            f"parameter grid expands to {n_combinations} combinations, "
            f"above the limit of {MAX_COMBINATIONS}; narrow the grid"
        )

    step_delta = pd.Timedelta(milliseconds=timeframe_to_ms(base_config.timeframe))
    start_ts = pd.Timestamp(candles["timestamp"].iloc[0])
    end_ts = pd.Timestamp(candles["timestamp"].iloc[-1]) + step_delta
    windows = plan_windows(start_ts, end_ts, is_days, oos_days, mode)
    logger.info(
        "walk-forward: %d %s window(s), IS %dd / OOS %dd, objective=%s",
        len(windows),
        mode,
        is_days,
        oos_days,
        objective,
    )

    results = WalkForwardResult(
        meta={
            "symbol": base_config.symbol,
            "timeframe": base_config.timeframe,
            "strategy": base_config.strategy,
            "mode": mode,
            "is_days": is_days,
            "oos_days": oos_days,
            "objective": objective,
            "param_grid": {name: list(values) for name, values in param_grid.items()},
            "n_windows": len(windows),
            "data_start": start_ts.isoformat(),
            "data_end": end_ts.isoformat(),
        }
    )

    segments: list[pd.Series] = []
    accumulated = 1.0
    for window in windows:
        window_result, oos_part, oos_trades = _run_window(
            candles, base_config, param_grid, window, mode, objective
        )
        results.windows.append(window_result)
        if window_result.error is not None:
            continue
        segments.append(_stitch_segment(oos_part, accumulated))
        accumulated *= float(oos_part.iloc[-1]) / float(oos_part.iloc[0])
        results.trades = (
            oos_trades if results.trades.empty
            else pd.concat([results.trades, oos_trades], ignore_index=True)
        )

    if segments:
        results.stitched_equity = pd.concat(segments)
    return results


def _run_window(
    candles: pd.DataFrame,
    base_config: BacktestConfig,
    param_grid: dict[str, list],
    window: Window,
    mode: str,
    objective: str,
) -> tuple[WindowResult, pd.Series | None, pd.DataFrame | None]:
    """Прогнать одно окно: IS-sweep, выбор лучшей комбинации, OOS-проверка.

    Возвращает ``(result, oos_equity_part, oos_trades)``; последние два
    равны ``None``, если окно упало.
    """
    result = WindowResult(
        is_start=window.is_start,
        is_end=window.is_end,
        oos_start=window.oos_start,
        oos_end=window.oos_end,
        mode=mode,
    )
    # Тот же срез, который run_sweep построил бы из периода конфига;
    # проверяется здесь, чтобы разрыв данных (например, пропавший кусок
    # истории) уронил только это окно.
    if _slice_range(candles, window.is_start, window.is_end).empty:
        result.error = "no candles in IS window"
        return result, None, None
    try:
        is_frame = run_sweep(
            base_config.model_copy(
                update={
                    "start": window.is_start.strftime("%Y-%m-%d %H:%M:%S"),
                    "end": window.is_end.strftime("%Y-%m-%d %H:%M:%S"),
                }
            ),
            param_grid,
            candles,
        )
    except ValueError as error:  # одно плохое IS-окно не должно убить прогон
        # потолок комбинаций проверяется заранее в run_walkforward, поэтому
        # ValueError про потолок сюда дойти не может
        logger.debug("walk-forward window %s: IS sweep failed", window, exc_info=True)
        result.error = str(error) or type(error).__name__
        return result, None, None
    valid = is_frame[is_frame["error"].isna()]
    if valid.empty:
        result.error = "no valid combos in IS"
        return result, None, None
    scores = pd.to_numeric(valid[objective], errors="coerce")
    if not bool(scores.notna().any()):
        result.error = f"no valid combos in IS with a finite {objective}"
        return result, None, None

    best = valid.loc[scores.idxmax()]
    # ячейки фрейма возвращаются как numpy-скаляры; конструкторы стратегий
    # (и json.dumps) ждут обычные Python int/float
    result.best_params = {name: _pythonize(best[name]) for name in param_grid}
    result.is_objective = float(best[objective])

    try:
        strategy = create_strategy(
            base_config.strategy, {**base_config.strategy_params, **result.best_params}
        )
        lead_in = pd.Timedelta(
            milliseconds=strategy.warmup_period * timeframe_to_ms(base_config.timeframe)
        )
        oos_slice = _slice_range(candles, window.oos_start - lead_in, window.oos_end)
        engine_result = build_engine(base_config, strategy).run(oos_slice)
    except Exception as error:  # noqa: BLE001 - одно плохое окно не должно убить прогон
        logger.debug("walk-forward window %s failed", window, exc_info=True)
        result.error = str(error) or type(error).__name__
        return result, None, None

    oos_part = engine_result.equity_curve.loc[engine_result.equity_curve.index >= window.oos_start]
    if oos_part.empty:
        result.error = "no OOS candles at/after oos_start"
        return result, None, None
    oos_trades = _filter_trades_by_entry(engine_result.trades, window.oos_start)
    metrics = compute_metrics(oos_part, oos_trades, base_config.timeframe)
    result.oos_return_pct = metrics.total_return_pct
    result.oos_sharpe = metrics.sharpe
    result.oos_max_dd_pct = metrics.max_drawdown_pct
    result.oos_trades = metrics.n_trades
    return result, oos_part, oos_trades


def _pythonize(value: Any) -> Any:
    """Перевести numpy-скаляр из ячейки фрейма в обычный Python-скаляр."""
    return value.item() if isinstance(value, np.generic) else value


def _slice_range(candles: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Оставить свечи с ``start <= timestamp < end`` (``start`` может быть раньше данных)."""
    ts = candles["timestamp"]
    return candles.loc[(ts >= start) & (ts < end)].reset_index(drop=True)


def _filter_trades_by_entry(trades: pd.DataFrame, oos_start: pd.Timestamp) -> pd.DataFrame:
    """Оставить сделки со входом не раньше ``oos_start``."""
    if trades.empty:
        return trades
    entry_ts = pd.to_datetime(trades["entry_ts"], utc=True)
    return trades.loc[entry_ts >= oos_start].reset_index(drop=True)


def _stitch_segment(oos_part: pd.Series, accumulated: float) -> pd.Series:
    """Нормировать OOS-часть эквити к её первому значению и умножить на ``accumulated``."""
    return oos_part.astype("float64") / float(oos_part.iloc[0]) * accumulated
