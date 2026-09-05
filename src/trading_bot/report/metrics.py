"""Метрики производительности для бэктест-прогонов (чистые функции, без I/O)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_bot.data.exchange import timeframe_to_ms

# Год в миллисекундах (365.25 дня): общая временная база для числа лет в CAGR
# и аннуализации Шарпа.
YEAR_MS = 365.25 * 24 * 60 * 60 * 1000
MS_PER_DAY = 24 * 60 * 60 * 1000

_DAY = pd.Timedelta(days=1)
_HOUR = pd.Timedelta(hours=1)


@dataclass(frozen=True)
class MetricsReport:
    """Метрики производительности одного бэктест-прогона.

    Метрики эквити:
        total_return_pct: ``(final / first - 1) * 100``.
        final_equity: последнее значение эквити, в котируемой валюте.
        cagr_pct: годовая (аннуализированная) доходность за
            ``span_days / 365.25`` лет; ``None``, когда у кривой нет
            временной протяжённости (одна точка).
        sharpe: аннуализированный коэффициент Шарпа доходностей эквити
            по свечам (безрисковая ставка 0); ``None``, когда дисперсия
            доходностей нулевая или не определена.
        max_drawdown_pct: наибольшее падение от пика до дна, в процентах (<= 0).
        max_drawdown_days: дни от пика наибольшей просадки до её восстановления
            либо до конца кривой, если восстановления не было; ``None``, когда
            просадок не было вовсе.
        span_days: временная протяжённость кривой эквити, в днях.

    Метрики сделок (все ``None``, если прогон не закрыл ни одной сделки):
        n_trades: число закрытых круговых сделок (всегда int).
        winrate_pct: доля сделок с ``pnl > 0``.
        profit_factor: валовая прибыль / валовый убыток; ``inf``, когда
            убыточных сделок нет.
        avg_trade_pnl, avg_win, avg_loss, best_trade, worst_trade: статистика
            pnl в котируемой валюте (``avg_win``/``avg_loss`` равны ``None``,
            когда нет соответственно выигрышей/проигрышей).
        avg_holding_hours: среднее ``exit_ts - entry_ts`` в часах.
        total_fees: оценка комиссий, если передан ``fee_rate`` (см.
            :func:`compute_metrics`), иначе ``None``.
    """

    total_return_pct: float
    final_equity: float
    cagr_pct: float | None
    sharpe: float | None
    max_drawdown_pct: float
    max_drawdown_days: float | None
    span_days: float

    n_trades: int
    winrate_pct: float | None
    profit_factor: float | None
    avg_trade_pnl: float | None
    avg_win: float | None
    avg_loss: float | None
    best_trade: float | None
    worst_trade: float | None
    avg_holding_hours: float | None
    total_fees: float | None

    @property
    def short_span(self) -> bool:
        """True, когда прогон короче года (CAGR экстраполирован)."""
        return self.span_days * MS_PER_DAY < YEAR_MS


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Метрики бенчмарка buy & hold за тот же период, что и бэктест.

    Attributes:
        total_return_pct: ``(final / first - 1) * 100``.
        cagr_pct: годовая (аннуализированная) доходность; ``None`` для
            кривой из одной точки.
        sharpe: аннуализированный коэффициент Шарпа доходностей по свечам
            (безрисковая ставка 0); ``None``, когда дисперсия нулевая
            или не определена.
        max_drawdown_pct: наибольшее падение от пика до дна, в процентах (<= 0).
    """

    total_return_pct: float
    cagr_pct: float | None
    sharpe: float | None
    max_drawdown_pct: float


def compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    timeframe: str,
    *,
    fee_rate: float | None = None,
) -> MetricsReport:
    """Вычислить метрики производительности по артефактам бэктеста.

    Args:
        equity: кривая эквити по рынку, индексированная метками времени свечей.
        trades: закрытые сделки со схемой ``TradeRecord`` (``entry_ts``,
            ``exit_ts``, ``entry_price``, ``exit_price``, ``quantity``,
            ``pnl``, ...); может быть пустым.
        timeframe: таймфрейм свечи (``"15m"``, ``"4h"``, ``"1d"`` ...),
            используемый для аннуализации Шарпа:
            периодов в год = ``YEAR_MS / timeframe_ms``.
        fee_rate: необязательная taker-комиссия за сторону; если задана,
            ``total_fees`` считается как ``fee_rate * sum(quantity *
            (entry_price + exit_price))`` по записанным ценам исполнения
            (они уже включают проскальзывание — совпадает с моделью
            комиссий брокера).

    Raises:
        ValueError: если кривая эквити пуста или таймфрейм некорректен.
    """
    if equity.empty:
        raise ValueError("equity curve is empty: nothing to report")

    equity_metrics = _equity_metrics(equity, timeframe)
    trade_metrics = _trade_metrics(trades, fee_rate=fee_rate)

    return MetricsReport(**equity_metrics, **trade_metrics)


def _equity_metrics(equity: pd.Series, timeframe: str) -> dict[str, float | None]:
    """Базовые метрики кривой эквити, общие для отчётов стратегии и бенчмарка.

    Возвращает dict с ключами ``total_return_pct``, ``final_equity``,
    ``cagr_pct``, ``sharpe``, ``max_drawdown_pct``, ``max_drawdown_days``
    и ``span_days``.
    """
    values = equity.to_numpy(dtype="float64")
    start_equity = float(values[0])
    final_equity = float(values[-1])
    total_return_pct = (final_equity / start_equity - 1.0) * 100.0

    span = equity.index[-1] - equity.index[0]
    span_days = float(span / _DAY)
    if span_days > 0.0:
        years = span_days * MS_PER_DAY / YEAR_MS
        cagr_pct = ((final_equity / start_equity) ** (1.0 / years) - 1.0) * 100.0
    else:
        cagr_pct = None

    returns = equity.astype("float64").pct_change().dropna()
    sharpe: float | None = None
    if len(returns) >= 2:
        std = float(returns.std())  # ddof=1
        if not math.isnan(std) and std > 0.0:
            periods_per_year = YEAR_MS / timeframe_to_ms(timeframe)
            sharpe = float(returns.mean()) / std * math.sqrt(periods_per_year)

    max_drawdown_pct, max_drawdown_days = _drawdown(equity, values)

    return {
        "total_return_pct": total_return_pct,
        "final_equity": final_equity,
        "cagr_pct": cagr_pct,
        "sharpe": sharpe,
        "max_drawdown_pct": max_drawdown_pct,
        "max_drawdown_days": max_drawdown_days,
        "span_days": span_days,
    }


def benchmark_equity(close: pd.Series, start_cash: float) -> pd.Series:
    """Построить кривую эквити buy & hold из закрытий свечей.

    Весь ``start_cash`` инвестируется по первому закрытию; кривая — это
    ``start_cash * close / close[0]`` на том же индексе. Чистая функция.

    Модель нарочно наивная: разовая покупка по первому закрытию без комиссий
    и проскальзывания. При сравнении с бэктестом, который их платит, бенчмарк
    систематически в выигрыше — читайте сравнение как оптимистичную оценку
    снизу для buy & hold.

    Raises:
        ValueError: если ``close`` пуст, начинается с нуля или его первое
            значение NaN.
    """
    if close.empty:
        raise ValueError("close series is empty: nothing to benchmark")
    first_close = float(close.iloc[0])
    if math.isnan(first_close):
        raise ValueError("first close is NaN: cannot build the benchmark curve")
    if first_close == 0.0:
        raise ValueError("first close is zero: cannot build the benchmark curve")
    return start_cash * close.astype("float64") / first_close


def compute_benchmark_metrics(equity: pd.Series, timeframe: str) -> BenchmarkMetrics:
    """Вычислить метрики buy & hold по кривой эквити бенчмарка.

    Переиспользует те же формулы, что и :func:`compute_metrics` (через
    общие хелперы эквити), без метрик сделок.

    Raises:
        ValueError: если кривая эквити пуста или таймфрейм некорректен.
    """
    if equity.empty:
        raise ValueError("equity curve is empty: nothing to report")
    fields = _equity_metrics(equity, timeframe)
    return BenchmarkMetrics(
        total_return_pct=fields["total_return_pct"],
        cagr_pct=fields["cagr_pct"],
        sharpe=fields["sharpe"],
        max_drawdown_pct=fields["max_drawdown_pct"],
    )


def _drawdown(equity: pd.Series, values: np.ndarray) -> tuple[float, float | None]:
    """Вернуть ``(макс. просадка %, длительность в днях)`` наибольшей просадки.

    Длительность считается от пика, предшествующего самому глубокому дну,
    до первого восстановления до этого пика (или до конца кривой, если
    эквити так и не восстановилось).
    """
    running_max = np.maximum.accumulate(values)
    drawdown = values / running_max - 1.0
    max_dd_pct = float(drawdown.min() * 100.0)
    if max_dd_pct >= 0.0:
        return 0.0, None

    trough_pos = int(drawdown.argmin())
    peak_pos = int(values[: trough_pos + 1].argmax())
    peak_value = values[peak_pos]
    recovered = np.nonzero(values[trough_pos:] >= peak_value)[0]
    end_pos = trough_pos + int(recovered[0]) if recovered.size else len(values) - 1
    duration = float((equity.index[end_pos] - equity.index[peak_pos]) / _DAY)
    return max_dd_pct, duration


def _trade_metrics(
    trades: pd.DataFrame, *, fee_rate: float | None
) -> dict[str, float | int | None]:
    """Посчитать метрики по сделкам; каждая метрика — ``None`` при нуле сделок."""
    n_trades = int(len(trades))
    if n_trades == 0:
        return {
            "n_trades": 0,
            "winrate_pct": None,
            "profit_factor": None,
            "avg_trade_pnl": None,
            "avg_win": None,
            "avg_loss": None,
            "best_trade": None,
            "worst_trade": None,
            "avg_holding_hours": None,
            "total_fees": None,
        }

    pnl = trades["pnl"].astype("float64")
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())

    entry_ts = pd.to_datetime(trades["entry_ts"], utc=True)
    exit_ts = pd.to_datetime(trades["exit_ts"], utc=True)
    avg_holding_hours = float((exit_ts - entry_ts).mean() / _HOUR)

    total_fees: float | None = None
    if fee_rate is not None and fee_rate >= 0:
        notional = trades["quantity"].astype("float64") * (
            trades["entry_price"].astype("float64")
            + trades["exit_price"].astype("float64")
        )
        total_fees = float(fee_rate * notional.sum())

    return {
        "n_trades": n_trades,
        "winrate_pct": len(wins) / n_trades * 100.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0.0 else math.inf,
        "avg_trade_pnl": float(pnl.mean()),
        "avg_win": float(wins.mean()) if len(wins) else None,
        "avg_loss": float(losses.mean()) if len(losses) else None,
        "best_trade": float(pnl.max()),
        "worst_trade": float(pnl.min()),
        "avg_holding_hours": avg_holding_hours,
        "total_fees": total_fees,
    }
