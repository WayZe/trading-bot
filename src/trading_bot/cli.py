"""Typer CLI entry point."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from trading_bot.config import BacktestConfig, load_config
from trading_bot.data.downloader import EXCHANGE_ID, HistoryDownloader
from trading_bot.data.exchange import ExchangeClient
from trading_bot.data.storage import CandleStorage
from trading_bot.engine.backtest import BacktestEngine, BacktestResult
from trading_bot.engine.broker import SimulatedBroker
from trading_bot.report import (
    BenchmarkMetrics,
    MetricsReport,
    benchmark_equity,
    compute_benchmark_metrics,
    compute_metrics,
)
from trading_bot.report.plots import plot_equity, plot_trades
from trading_bot.research import run_sweep, slice_candles
from trading_bot.risk import RiskManager
from trading_bot.strategy import create_strategy

logger = logging.getLogger(__name__)

app = typer.Typer(help="Educational crypto trading bot (Bybit spot, backtest-first).")

DATA_ROOT = Path("data")
REPORTS_DIR = Path("reports")
LAST_RUN_DIR = REPORTS_DIR / "last_run"
SWEEP_DIR = REPORTS_DIR / "sweep" / "last"

RUN_FILES = ("equity.parquet", "trades.csv", "meta.json")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.callback()
def _root_callback() -> None:
    """Configure console logging for all commands."""
    _setup_logging()


@app.command()
def download(
    symbol: Annotated[str, typer.Option(help="Trading pair in ccxt format, e.g. BTC/USDT.")],
    timeframe: Annotated[str, typer.Option(help="Candle timeframe: 15m, 1h, 4h, 1d, ...")] = "4h",
    since: Annotated[
        str | None,
        typer.Option(
            help="Start date (UTC, YYYY-MM-DD) for a full download; "
            "not needed with --update when a dataset already exists."
        ),
    ] = None,
    until: Annotated[
        str | None,
        typer.Option(help="End date (UTC, YYYY-MM-DD, exclusive). Defaults to today."),
    ] = None,
    update: Annotated[
        bool, typer.Option("--update", help="Incrementally update an existing dataset.")
    ] = False,
) -> None:
    """Download historical candles from Bybit into local Parquet storage."""
    storage = CandleStorage(DATA_ROOT)
    downloader = HistoryDownloader(ExchangeClient())
    path = storage.path_for(EXCHANGE_ID, symbol, timeframe)

    if update and path.exists():
        typer.echo(f"Updating {symbol} {timeframe} incrementally ...")
        df = downloader.update(symbol, timeframe, storage)
    else:
        if since is None:
            typer.echo(
                "--since is required for a full download "
                "(or pass --update with an existing dataset)."
            )
            raise typer.Exit(code=1)
        if update:
            typer.echo(f"No existing dataset at {path}; running a full download.")
        df = downloader.download(symbol, timeframe, since=since, until=until)
        storage.save(EXCHANGE_ID, symbol, timeframe, df)

    typer.echo(
        f"Saved {len(df)} candles ({symbol} {timeframe}): "
        f"{df['timestamp'].iloc[0]} .. {df['timestamp'].iloc[-1]} -> {path}"
    )


@app.command()
def backtest(
    config: Annotated[
        Path, typer.Option(help="Path to the backtest YAML config.")
    ] = Path("config/backtest.yaml"),
) -> None:
    """Run a backtest over stored candles and save run artifacts."""
    cfg = load_config(config)
    candles = _load_candles_for_period(cfg)

    try:
        strategy = create_strategy(cfg.strategy, cfg.strategy_params)
    except ValueError as error:
        typer.echo(f"Ошибка конфигурации стратегии: {error}")
        raise typer.Exit(code=1) from error

    engine = BacktestEngine(
        strategy=strategy,
        risk=RiskManager(
            position_size_pct=cfg.position_size_pct,
            quantity_precision=cfg.quantity_precision,
            min_notional=cfg.min_notional,
        ),
        broker=SimulatedBroker(fee_rate=cfg.fee_rate, slippage_bps=cfg.slippage_bps),
        start_cash=cfg.start_cash,
        config=cfg,
    )
    result = engine.run(candles)

    # Buy & hold benchmark over exactly the tested range (the candle slice
    # fed to the engine), fully invested in the asset at the first close.
    benchmark = benchmark_equity(candles.set_index("timestamp")["close"], cfg.start_cash)

    summary = _summary(cfg, result)
    _print_summary(summary)
    _save_artifacts(result, benchmark, cfg, summary)


def _load_candles_for_period(cfg: BacktestConfig) -> pd.DataFrame:
    """Load candles for the config symbol/timeframe sliced to the period.

    Exits with a user-facing hint when the dataset is missing or the period
    is empty. Shared by the ``backtest`` and ``sweep`` commands.
    """
    storage = CandleStorage(DATA_ROOT)
    candles = storage.load(EXCHANGE_ID, cfg.symbol, cfg.timeframe)
    if candles is None:
        path = storage.path_for(EXCHANGE_ID, cfg.symbol, cfg.timeframe)
        typer.echo(
            f"Нет данных: {path} не найден.\n"
            f"Сначала скачай историю: trading-bot download "
            f"--symbol {cfg.symbol} --timeframe {cfg.timeframe} --since <YYYY-MM-DD>"
        )
        raise typer.Exit(code=1)

    sliced = slice_candles(candles, cfg)
    if sliced.empty:
        typer.echo(
            f"В данных {cfg.symbol} {cfg.timeframe} нет свечей за период "
            f"{cfg.start} .. {cfg.end or 'конец'}."
        )
        raise typer.Exit(code=1)
    return sliced


def _parse_grid(specs: list[str] | None) -> dict[str, list]:
    """Parse repeated ``--param name=v1,v2,...`` options into a grid dict.

    Values are coerced: int-like strings become ``int``, other numeric
    strings become ``float``, anything else stays a string. The strategy
    constructor validates the final types and values.
    """
    if not specs:
        return {}
    grid: dict[str, list] = {}
    for spec in specs:
        name, _, raw_values = spec.partition("=")
        name = name.strip()
        values = [item.strip() for item in raw_values.split(",") if item.strip()]
        if not name or not values:
            raise typer.BadParameter(f"--param expects 'name=v1,v2,...', got {spec!r}")
        if name in grid:
            raise typer.BadParameter(f"--param {name!r} задан дважды")
        grid[name] = [_coerce_scalar(item) for item in values]
    return grid


def _coerce_scalar(raw: str) -> int | float | str:
    """Coerce a grid value: int-like strings to int, numeric to float, else str."""
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


@app.command()
def sweep(
    config: Annotated[
        Path, typer.Option(help="Path to the backtest YAML config.")
    ] = Path("config/backtest.yaml"),
    param: Annotated[
        list[str] | None,
        typer.Option(
            "--param",
            help="Параметр сетки 'name=v1,v2,...' (повторяемый): "
            "--param fast=10,15,20 --param slow=30,50.",
        ),
    ] = None,
) -> None:
    """Прогнать бэктест по сетке параметров стратегии и свести результаты."""
    cfg = load_config(config)
    grid = _parse_grid(param)
    if not grid:
        typer.echo(
            "Укажи хотя бы один параметр сетки: "
            "--param fast=10,20,30 --param slow=50,100"
        )
        raise typer.Exit(code=1)

    candles = _load_candles_for_period(cfg)

    try:
        results = run_sweep(cfg, grid, candles)
    except ValueError as error:
        typer.echo(f"Ошибка sweep: {error}")
        raise typer.Exit(code=1) from error

    _save_sweep_artifacts(cfg, grid, results)
    _print_sweep(cfg, grid, results)


def _save_sweep_artifacts(
    cfg: BacktestConfig, grid: dict[str, list], results: pd.DataFrame
) -> None:
    """Write results.csv and meta.json (config + grid) to reports/sweep/last/."""
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(SWEEP_DIR / "results.csv", index=False)
    meta = {
        "config": cfg.model_dump(mode="json"),
        "grid": grid,
        "n_combinations": int(len(results)),
    }
    with (SWEEP_DIR / "meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)
    typer.echo(f"Результаты: {SWEEP_DIR / 'results.csv'}")


def _print_sweep(cfg: BacktestConfig, grid: dict[str, list], results: pd.DataFrame) -> None:
    """Print the sweep results table and the best row by total return.

    Grids with more than 20 rows show only the top-10 successful
    combinations; failed combinations are always listed last with a
    truncated error text.
    """
    table = Table(
        title=f"Sweep: {cfg.strategy} · {cfg.symbol} · {cfg.timeframe} "
        f"({len(results)} комбинаций)",
        title_justify="left",
    )
    for name in grid:
        table.add_column(name, justify="right", no_wrap=True)
    table.add_column("Доходность", justify="right")
    table.add_column("CAGR", justify="right")
    table.add_column("Шарп", justify="right")
    table.add_column("Макс. просадка", justify="right")
    table.add_column("Сделок", justify="right")
    table.add_column("Ошибка", no_wrap=True, max_width=48, overflow="fold")

    valid = results[results["error"].isna()]
    failed = results[results["error"].notna()]

    shown = valid
    note = None
    if len(results) > 20 and not valid.empty:
        shown = valid.nlargest(10, "total_return_pct")
        note = (
            f"(показаны топ-10 из {len(valid)}; "
            f"полные результаты в {SWEEP_DIR / 'results.csv'})"
        )
    for _, row in shown.iterrows():
        table.add_row(*(_sweep_cell(row, name) for name in grid),
                      _sweep_cell(row, "total_return_pct"),
                      _sweep_cell(row, "cagr_pct"),
                      _sweep_cell(row, "sharpe"),
                      _sweep_cell(row, "max_drawdown_pct"),
                      _sweep_cell(row, "n_trades"),
                      "")
    for _, row in failed.iterrows():
        table.add_row(*(_sweep_cell(row, name) for name in grid),
                      *(_sweep_cell(row, col) for col in
                        ("total_return_pct", "cagr_pct", "sharpe", "max_drawdown_pct", "n_trades")),
                      str(row["error"]))

    Console().print(table)
    if note:
        typer.echo(note)

    if valid.empty:
        typer.echo("Ни одна комбинация не выполнилась успешно — см. тексты ошибок выше.")
        return
    best = valid.loc[valid["total_return_pct"].idxmax()]
    params_str = ", ".join(f"{name}={best[name]}" for name in grid)
    typer.echo(
        f"Лучшая комбинация: {params_str} — "
        f"доходность {best['total_return_pct']:+.2f}%, "
        f"CAGR {_fmt_opt(_opt(best['cagr_pct']), _fmt_pct)}, "
        f"Шарп {_fmt_opt(_opt(best['sharpe']), lambda v: f'{v:.2f}')}, "
        f"макс. просадка {best['max_drawdown_pct']:+.2f}%, "
        f"итоговый капитал {_fmt_money(_opt(best['final_equity']))} USDT"
    )


def _opt(value) -> float | None:
    """Convert a possibly-NaN sweep cell to ``float | None``."""
    return None if pd.isna(value) else float(value)


def _sweep_cell(row: pd.Series, name: str) -> str:
    """Format one sweep table cell (NaN as an em dash, floats with 2 dp)."""
    value = row[name]
    if pd.isna(value):
        return "—"
    if name == "n_trades":
        return str(int(value))
    if isinstance(value, float):
        if math.isinf(value):
            return "∞"
        sign = "+" if name in ("total_return_pct", "max_drawdown_pct") else ""
        return f"{value:{sign}.2f}"
    return str(value)


def _summary(cfg: BacktestConfig, result: BacktestResult) -> dict:
    """Build a JSON-safe run summary."""
    final_equity = float(result.equity_curve.iloc[-1])
    position = result.open_position
    return {
        "symbol": cfg.symbol,
        "timeframe": cfg.timeframe,
        "strategy": cfg.strategy,
        "candles_start": result.candles_start.isoformat() if result.candles_start else None,
        "candles_end": result.candles_end.isoformat() if result.candles_end else None,
        "n_candles": int(len(result.equity_curve)),
        "start_cash": cfg.start_cash,
        "final_equity": final_equity,
        "total_return_pct": (final_equity / cfg.start_cash - 1.0) * 100.0,
        "n_trades": int(len(result.trades)),
        "n_winning_trades": int((result.trades["pnl"] > 0).sum()) if len(result.trades) else 0,
        "open_position": None
        if position is None
        else {
            "quantity": position.quantity,
            "entry_price": position.entry_price,
            "entry_ts": position.entry_ts.isoformat(),
        },
        "n_pending_unfilled": result.n_pending_unfilled,
    }


def _print_summary(summary: dict) -> None:
    typer.echo(
        f"Бэктест «{summary['strategy']}» {summary['symbol']} {summary['timeframe']}: "
        f"{summary['candles_start']} .. {summary['candles_end']} "
        f"({summary['n_candles']} свечей)"
    )
    typer.echo(
        f"Капитал: {summary['start_cash']:.2f} -> {summary['final_equity']:.2f} USDT "
        f"({summary['total_return_pct']:+.2f}%)"
    )
    typer.echo(
        f"Закрытых сделок: {summary['n_trades']}"
        f" (прибыльных: {summary['n_winning_trades']})"
    )
    if summary["open_position"] is not None:
        position = summary["open_position"]
        typer.echo(
            f"Открытая позиция: {position['quantity']} @ {position['entry_price']:.2f} "
            f"с {position['entry_ts']}"
        )
    else:
        typer.echo("Открытая позиция: нет")
    typer.echo(f"Артефакты прогона: {LAST_RUN_DIR}/")
    typer.echo("Сводка и графики: uv run trading-bot report")


def _save_artifacts(
    result: BacktestResult, benchmark: pd.Series, cfg: BacktestConfig, summary: dict
) -> None:
    """Write equity.parquet, trades.csv, benchmark.parquet and meta.json."""
    LAST_RUN_DIR.mkdir(parents=True, exist_ok=True)
    result.equity_curve.to_frame().to_parquet(LAST_RUN_DIR / "equity.parquet", engine="pyarrow")
    benchmark.to_frame("equity").to_parquet(
        LAST_RUN_DIR / "benchmark.parquet", engine="pyarrow"
    )
    result.trades.to_csv(LAST_RUN_DIR / "trades.csv", index=False)
    meta = {"config": cfg.model_dump(mode="json"), "summary": summary}
    with (LAST_RUN_DIR / "meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)


@app.command()
def report(
    run_dir: Annotated[
        Path,
        typer.Option(help="Каталог артефактов прогона (equity.parquet, trades.csv, meta.json)."),
    ] = LAST_RUN_DIR,
    candles: Annotated[
        Path | None,
        typer.Option(
            help="Parquet со свечами для графика сделок; по умолчанию ищется в data/ по meta.json."
        ),
    ] = None,
) -> None:
    """Построить сводку метрик и графики по завершённому бэктесту."""
    equity, trades, meta = _load_run(run_dir)
    cfg = meta.get("config", {})
    timeframe = str(cfg.get("timeframe", "1d"))

    metrics = compute_metrics(equity, trades, timeframe, fee_rate=cfg.get("fee_rate"))
    benchmark = _load_benchmark(run_dir)
    benchmark_metrics = (
        compute_benchmark_metrics(benchmark, timeframe) if benchmark is not None else None
    )
    _print_report(cfg, equity, metrics, benchmark_metrics)

    equity_png = run_dir / "equity.png"
    plot_equity(equity, equity_png, benchmark=benchmark)
    typer.echo(f"График эквити: {equity_png}")

    candles_path = _resolve_candles(candles, cfg, timeframe)
    if candles_path is not None:
        plot_trades(pd.read_parquet(candles_path, engine="pyarrow"), trades, run_dir / "trades.png")
        typer.echo(f"График сделок: {run_dir / 'trades.png'}")


def _load_run(run_dir: Path) -> tuple[pd.Series, pd.DataFrame, dict]:
    """Load run artifacts; exit with a hint when they are missing or broken."""
    missing = [name for name in RUN_FILES if not (run_dir / name).is_file()]
    if missing:
        typer.echo(f"В каталоге {run_dir} нет артефактов прогона: {', '.join(missing)}.")
        typer.echo("Сначала запусти бэктест: uv run trading-bot backtest")
        raise typer.Exit(code=1)

    try:
        frame = pd.read_parquet(run_dir / "equity.parquet", engine="pyarrow")
        equity = frame["equity"] if isinstance(frame, pd.DataFrame) else frame
        trades = pd.read_csv(run_dir / "trades.csv", parse_dates=["entry_ts", "exit_ts"])
        with (run_dir / "meta.json").open(encoding="utf-8") as file:
            meta = json.load(file)
    except (OSError, ValueError, KeyError) as error:
        typer.echo(f"Не удалось прочитать артефакты прогона в {run_dir}: {error}")
        raise typer.Exit(code=1) from error
    return equity, trades, meta


def _load_benchmark(run_dir: Path) -> pd.Series | None:
    """Load the optional buy & hold benchmark artifact (``None`` when absent)."""
    path = run_dir / "benchmark.parquet"
    if not path.is_file():
        return None
    try:
        frame = pd.read_parquet(path, engine="pyarrow")
        return frame.iloc[:, 0] if isinstance(frame, pd.DataFrame) else frame
    except (OSError, ValueError) as error:
        typer.echo(f"Не удалось прочитать {path}: сравнение с buy & hold пропущено ({error})")
        return None


def _resolve_candles(candles: Path | None, cfg: dict, timeframe: str) -> Path | None:
    """Resolve the candles file for the trades plot, with hints when absent."""
    if candles is not None:
        if not candles.is_file():
            typer.echo(f"Файл свечей не найден: {candles}")
            raise typer.Exit(code=1)
        return candles

    symbol = str(cfg.get("symbol", ""))
    expected = CandleStorage(DATA_ROOT).path_for(EXCHANGE_ID, symbol, timeframe)
    if expected.is_file():
        return expected
    typer.echo(f"Свечи не найдены ({expected}): график сделок пропущен.")
    typer.echo(
        "Скачай историю: uv run trading-bot download "
        f"--symbol {symbol or 'BTC/USDT'} --timeframe {timeframe} --since <YYYY-MM-DD>, "
        "или укажи файл: uv run trading-bot report --candles <файл.parquet>"
    )
    return None


def _fmt_money(value: float) -> str:
    return f"{value:,.2f}".replace(",", " ")


def _fmt_pct(value: float) -> str:
    return f"{value:+.2f}%"


def _fmt_opt(value: float | None, fmt) -> str:
    return "—" if value is None else fmt(value)


def _print_report(
    cfg: dict,
    equity: pd.Series,
    metrics: MetricsReport,
    benchmark: BenchmarkMetrics | None = None,
) -> None:
    """Print the metrics table to the console (rich is a typer dependency)."""
    console = Console()
    table = Table(
        title=f"Отчёт бэктеста: {cfg.get('strategy', '?')} · "
        f"{cfg.get('symbol', '?')} · {cfg.get('timeframe', '?')}",
        title_justify="left",
    )
    table.add_column("Метрика", no_wrap=True)
    table.add_column("Значение", justify="right")

    start, end = equity.index[0], equity.index[-1]
    table.add_row("Период", f"{start:%Y-%m-%d} — {end:%Y-%m-%d} ({metrics.span_days:.1f} дн)")
    table.add_row("Итоговый капитал", f"{_fmt_money(metrics.final_equity)} USDT")
    table.add_row("Доходность", _fmt_pct(metrics.total_return_pct))
    table.add_row("Годовая доходность (CAGR)", _fmt_opt(metrics.cagr_pct, _fmt_pct))
    table.add_row("Коэффициент Шарпа (аннуал.)", _fmt_opt(metrics.sharpe, lambda v: f"{v:.2f}"))
    table.add_row("Максимальная просадка", _fmt_pct(metrics.max_drawdown_pct))
    table.add_row(
        "Длительность макс. просадки",
        _fmt_opt(metrics.max_drawdown_days, lambda v: f"{v:.1f} дн"),
    )
    table.add_row("Закрытых сделок", str(metrics.n_trades))
    table.add_row("Winrate", _fmt_opt(metrics.winrate_pct, lambda v: f"{v:.2f}%"))
    profit_factor = _fmt_opt(
        metrics.profit_factor,
        lambda v: "∞" if math.isinf(v) else f"{v:.2f}",
    )
    table.add_row("Profit factor", profit_factor)
    table.add_row("Средний PnL сделки", _fmt_opt(metrics.avg_trade_pnl, _fmt_money))
    table.add_row(
        "Средний выигрыш / проигрыш",
        f"{_fmt_opt(metrics.avg_win, _fmt_money)} / {_fmt_opt(metrics.avg_loss, _fmt_money)}",
    )
    table.add_row(
        "Лучшая / худшая сделка",
        f"{_fmt_opt(metrics.best_trade, _fmt_money)} / {_fmt_opt(metrics.worst_trade, _fmt_money)}",
    )
    table.add_row(
        "Среднее удержание позиции", _fmt_opt(metrics.avg_holding_hours, lambda v: f"{v:.1f} ч")
    )
    table.add_row("Комиссии (оценка)", _fmt_opt(metrics.total_fees, _fmt_money))

    console.print(table)
    if metrics.short_span:
        typer.echo("* период меньше года: годовая доходность (CAGR) экстраполирована")

    if benchmark is not None:
        _print_benchmark_comparison(console, metrics, benchmark)


def _print_benchmark_comparison(
    console: Console, metrics: MetricsReport, benchmark: BenchmarkMetrics
) -> None:
    """Print the strategy vs buy & hold comparison table."""
    table = Table(title="Стратегия vs Buy & hold", title_justify="left")
    table.add_column("Метрика", no_wrap=True)
    table.add_column("Стратегия", justify="right")
    table.add_column("Buy & hold", justify="right")
    table.add_row(
        "Доходность",
        _fmt_pct(metrics.total_return_pct),
        _fmt_pct(benchmark.total_return_pct),
    )
    table.add_row(
        "Годовая доходность (CAGR)",
        _fmt_opt(metrics.cagr_pct, _fmt_pct),
        _fmt_opt(benchmark.cagr_pct, _fmt_pct),
    )
    table.add_row(
        "Максимальная просадка",
        _fmt_pct(metrics.max_drawdown_pct),
        _fmt_pct(benchmark.max_drawdown_pct),
    )
    table.add_row(
        "Коэффициент Шарпа (аннуал.)",
        _fmt_opt(metrics.sharpe, lambda v: f"{v:.2f}"),
        _fmt_opt(benchmark.sharpe, lambda v: f"{v:.2f}"),
    )
    console.print(table)


def main() -> None:
    """Script entry point."""
    app()
