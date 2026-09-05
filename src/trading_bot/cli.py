"""Typer CLI entry point."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from trading_bot.config import BacktestConfig, load_config
from trading_bot.data.downloader import EXCHANGE_ID, HistoryDownloader
from trading_bot.data.exchange import ExchangeClient
from trading_bot.data.storage import CandleStorage
from trading_bot.engine.backtest import BacktestEngine, BacktestResult
from trading_bot.engine.broker import SimulatedBroker
from trading_bot.risk import RiskManager
from trading_bot.strategy import create_strategy

logger = logging.getLogger(__name__)

app = typer.Typer(help="Educational crypto trading bot (Bybit spot, backtest-first).")

DATA_ROOT = Path("data")
REPORTS_DIR = Path("reports")
LAST_RUN_DIR = REPORTS_DIR / "last_run"


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
        str, typer.Option(help="Start date (UTC, YYYY-MM-DD) for a full download.")
    ] = ...,
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

    candles = _slice_candles(candles, cfg)
    if candles.empty:
        typer.echo(
            f"В данных {cfg.symbol} {cfg.timeframe} нет свечей за период "
            f"{cfg.start} .. {cfg.end or 'конец'}."
        )
        raise typer.Exit(code=1)

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

    summary = _summary(cfg, result)
    _print_summary(summary)
    _save_artifacts(result, cfg, summary)


def _slice_candles(candles: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """Keep candles with start <= timestamp < end (dates from the config)."""
    ts = candles["timestamp"]
    start_ts = pd.Timestamp(cfg.start, tz="UTC")
    if ts.iloc[0] > start_ts:
        logger.warning(
            "данные начинаются с %s, позже запрошенного start=%s",
            ts.iloc[0],
            cfg.start,
        )
    mask = ts >= start_ts
    if cfg.end is not None:
        mask &= ts < pd.Timestamp(cfg.end, tz="UTC")
    return candles.loc[mask].reset_index(drop=True)


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


def _save_artifacts(result: BacktestResult, cfg: BacktestConfig, summary: dict) -> None:
    """Write equity.parquet, trades.csv and meta.json to reports/last_run/."""
    LAST_RUN_DIR.mkdir(parents=True, exist_ok=True)
    result.equity_curve.to_frame().to_parquet(LAST_RUN_DIR / "equity.parquet", engine="pyarrow")
    result.trades.to_csv(LAST_RUN_DIR / "trades.csv", index=False)
    meta = {"config": cfg.model_dump(mode="json"), "summary": summary}
    with (LAST_RUN_DIR / "meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)


@app.command()
def report() -> None:
    """Render a report for a finished backtest (stub)."""
    typer.echo("Report generation будет реализован на этапе 4.")


def main() -> None:
    """Script entry point."""
    app()
