"""Точка входа CLI на typer."""

from __future__ import annotations

import fcntl
import json
import logging
import math
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from trading_bot.config import BacktestConfig, LiveConfig, load_config, load_live_config
from trading_bot.data.downloader import EXCHANGE_ID, HistoryDownloader
from trading_bot.data.exchange import ExchangeClient
from trading_bot.data.storage import CandleStorage
from trading_bot.engine.backtest import BacktestResult, build_engine
from trading_bot.engine.broker import SimulatedBroker
from trading_bot.live.execution import PaperAdapter, TestnetAdapter
from trading_bot.live.runner import LiveRunner
from trading_bot.live.state import load_or_fresh_state
from trading_bot.report import (
    BenchmarkMetrics,
    MetricsReport,
    benchmark_equity,
    compute_benchmark_metrics,
    compute_metrics,
)
from trading_bot.report.plots import plot_equity, plot_stitched_equity, plot_trades
from trading_bot.research import (
    WINDOW_COLUMNS,
    WalkForwardResult,
    run_sweep,
    run_walkforward,
    slice_candles,
)
from trading_bot.research.sweep import METRIC_COLUMNS
from trading_bot.strategy import create_strategy

logger = logging.getLogger(__name__)

app = typer.Typer(help="Educational crypto trading bot (Bybit spot, backtest-first).")

DATA_ROOT = Path("data")
REPORTS_DIR = Path("reports")
LAST_RUN_DIR = REPORTS_DIR / "last_run"
SWEEP_DIR = REPORTS_DIR / "sweep" / "last"
WALKFORWARD_DIR = REPORTS_DIR / "walkforward" / "last"

RUN_FILES = ("equity.parquet", "trades.csv", "meta.json")

# Имена переменных окружения с ключами Bybit testnet (никогда не в конфиге/git/логах).
ENV_API_KEY = "BYBIT_API_KEY"
ENV_API_SECRET = "BYBIT_API_SECRET"

# Имена сеток, которые конфликтовали бы с колонками результата sweep.
_RESERVED_PARAM_NAMES = frozenset(METRIC_COLUMNS) | {"error"}
# results.csv у walkforward также несёт колонки окон; команда walkforward
# отклоняет оба множества (см. WINDOW_COLUMNS в research.walkforward).
_WALKFORWARD_RESERVED_PARAM_NAMES = _RESERVED_PARAM_NAMES | frozenset(WINDOW_COLUMNS)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.callback()
def _root_callback() -> None:
    """Настроить консольное логирование для всех команд."""
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
    """Загрузить исторические свечи с Bybit в локальное Parquet-хранилище."""
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
def live(
    config: Annotated[
        Path, typer.Option(help="Path to the live YAML config.")
    ] = Path("config/live.yaml"),
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Force paper mode: no API keys, no private exchange calls, "
            "even if the config says testnet.",
        ),
    ] = False,
    once: Annotated[
        bool,
        typer.Option(
            "--once", help="Run a single cycle and exit (cron-friendly)."
        ),
    ] = False,
) -> None:
    """Запустить live/paper-раннер: торговля стратегией на закрытых свечах Bybit.

    Режим исполнения берётся из конфига (paper по умолчанию); ``--dry-run``
    форсирует paper. Состояние (позиция, стопы, последняя обработанная
    свеча) переживает рестарты через JSON-файл из конфига. Параллельный
    запуск над тем же state-файлом запрещён эксклюзивной блокировкой
    (``<state>.lock``): два раннера на одном состоянии задвоили бы ордера.
    """
    try:
        cfg = load_live_config(config)
    except ValueError as error:
        # Невалидный YAML/поля конфига (включая совпадающие пути свитчей):
        # печатаем короткое сообщение вместо traceback pydantic.
        typer.echo(f"Ошибка live-конфига: {error}")
        raise typer.Exit(code=1) from error
    if dry_run and cfg.mode != "paper":
        typer.echo("--dry-run: forcing paper mode (testnet adapter is disabled)")
        cfg = LiveConfig.model_validate({**cfg.model_dump(), "mode": "paper"})

    state_lock = _acquire_state_lock(cfg.state_path)
    if state_lock is None:
        typer.echo(
            f"live runner is already running for state {cfg.state_path} "
            "(lock file is held); refusing to start a second instance"
        )
        raise typer.Exit(code=1)

    try:
        strategy = create_strategy(cfg.strategy, cfg.strategy_params)
    except ValueError as error:
        typer.echo(f"Ошибка конфигурации стратегии: {error}")
        raise typer.Exit(code=1) from error

    exchange_client = _build_exchange_client(cfg)
    try:
        state = load_or_fresh_state(cfg.state_path, cfg)
        state.ensure_matches_config(cfg)
    except ValueError as error:
        # Битое или чужое состояние: молча перезаписывать нельзя, нужна подсказка.
        typer.echo(f"Ошибка состояния live: {error}")
        raise typer.Exit(code=1) from error

    if cfg.mode == "testnet":
        adapter = TestnetAdapter(exchange_client, cfg.symbol)
    else:
        broker = SimulatedBroker(fee_rate=cfg.fee_rate, slippage_bps=cfg.slippage_bps)
        adapter = PaperAdapter(
            broker,
            price_source=lambda: exchange_client.fetch_ticker_last(cfg.symbol),
            equity_source=lambda: state.equity,
        )

    runner = LiveRunner(
        config=cfg,
        state=state,
        strategy=strategy,
        adapter=adapter,
        exchange_client=exchange_client,
        storage=CandleStorage(cfg.data_root),
    )
    if cfg.mode == "testnet":
        runner.reconcile()
    _print_live_banner(cfg, state, runner)
    _setup_live_logging(cfg.log_file)

    if once:
        runner.run_once()
    else:
        runner.run_forever()


def _acquire_state_lock(state_path: str | Path):
    """Эксклюзивная блокировка против параллельного запуска (``fcntl.flock``).

    Блокируется отдельный файл ``<state>.lock`` рядом с state: если другой
    процесс раннера уже держит его, возвращается ``None`` — второй запуск
    над тем же состоянием должен отказаться стартовать (задвоил бы ордера
    и перезаписывал state друг друга). Хендлер держится открытым всё время
    работы команды: закрытие процесса снимает блокировку автоматически,
    даже после крэша.
    """
    lock_path = Path(str(state_path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def _build_exchange_client(cfg: LiveConfig) -> ExchangeClient:
    """Создать клиент биржи под режим конфига (ключи — только из окружения).

    Raises:
        typer.Exit: в режиме testnet без ``BYBIT_API_KEY``/``BYBIT_API_SECRET``
            — молча торговать без ключей нельзя.
    """
    if cfg.mode != "testnet":
        return ExchangeClient()
    api_key = os.environ.get(ENV_API_KEY)
    secret = os.environ.get(ENV_API_SECRET)
    if not api_key or not secret:
        typer.echo(
            f"mode=testnet requires {ENV_API_KEY} and {ENV_API_SECRET} environment "
            "variables (get testnet keys at https://testnet.bybit.com); "
            "no keys found, refusing to start."
        )
        raise typer.Exit(code=1)
    return ExchangeClient(api_key=api_key, secret=secret, sandbox=True)


def _print_live_banner(cfg: LiveConfig, state, runner: LiveRunner) -> None:
    """Напечатать стартовую сводку: режим, контур, свитчи, позиция."""
    typer.echo(
        f"Live runner: mode={cfg.mode} symbol={cfg.symbol} timeframe={cfg.timeframe}"
    )
    typer.echo(f"Strategy: {cfg.strategy} {cfg.strategy_params}")
    kill_state = "ACTIVE (all orders forbidden)" if runner.kill_switch_active() else "off"
    typer.echo(f"Kill switch ({cfg.kill_switch_path}): {kill_state}")
    pause_state = (
        "ACTIVE (entries forbidden, exits allowed)"
        if runner.pause_switch_active()
        else "off"
    )
    typer.echo(f"Pause switch ({cfg.pause_switch_path}): {pause_state}")
    if state.position is None:
        typer.echo("Position: flat")
    else:
        position = state.position
        typer.echo(
            f"Position: {position.quantity:.6f} @ {position.entry_price:.4f} "
            f"since {position.entry_ts} ({position.entry_reason})"
        )


def _setup_live_logging(log_file: str) -> None:
    """Добавить rotating file handler к консольному логированию live-раннера."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_path, maxBytes=1_000_000, backupCount=5, encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)


@app.command()
def backtest(
    config: Annotated[
        Path, typer.Option(help="Path to the backtest YAML config.")
    ] = Path("config/backtest.yaml"),
    symbol: Annotated[
        str | None, typer.Option(help="Переопределить пару из конфига, напр. ETH/USDT.")
    ] = None,
    timeframe: Annotated[
        str | None, typer.Option(help="Переопределить таймфрейм из конфига, напр. 1h.")
    ] = None,
) -> None:
    """Прогнать бэктест по сохранённым свечам и сохранить артефакты прогона."""
    cfg = _apply_overrides(load_config(config), symbol, timeframe)
    candles = _load_candles_for_period(cfg)

    try:
        strategy = create_strategy(cfg.strategy, cfg.strategy_params)
    except ValueError as error:
        typer.echo(f"Ошибка конфигурации стратегии: {error}")
        raise typer.Exit(code=1) from error

    engine = build_engine(cfg, strategy)
    result = engine.run(candles)

    # Бенчмарк buy & hold ровно за тестируемый период (срез свечей, который
    # получает движок), полностью инвестирован в актив по первому close.
    benchmark = benchmark_equity(candles.set_index("timestamp")["close"], cfg.start_cash)

    summary = _summary(cfg, result)
    _print_summary(summary)
    _save_artifacts(result, benchmark, cfg, summary)


def _apply_overrides(
    cfg: BacktestConfig, symbol: str | None, timeframe: str | None
) -> BacktestConfig:
    """Применить CLI-переопределения symbol/timeframe (``None`` сохраняет значение конфига).

    Конфиг ревалидируется, чтобы плохое переопределение (например,
    некорректный таймфрейм) падало со стандартным сообщением валидации.
    """
    updates: dict[str, str] = {}
    if symbol is not None:
        updates["symbol"] = symbol
    if timeframe is not None:
        updates["timeframe"] = timeframe
    if not updates:
        return cfg
    try:
        return BacktestConfig.model_validate({**cfg.model_dump(), **updates})
    except ValueError as error:
        typer.echo(f"Ошибка переопределения конфига: {error}")
        raise typer.Exit(code=1) from error


def _load_candles(cfg: BacktestConfig) -> pd.DataFrame:
    """Загрузить сырой датасет свечей для symbol/timeframe из конфига.

    Выходит с пользовательской подсказкой, если датасет отсутствует или
    символ — не валидная пара ccxt. Общая для команд ``backtest`` и ``sweep``.
    """
    storage = CandleStorage(DATA_ROOT)
    try:
        candles = storage.load(EXCHANGE_ID, cfg.symbol, cfg.timeframe)
    except ValueError as error:
        # symbol_to_slug отклоняет символы, которые могли бы выйти за корень хранилища.
        typer.echo(f"Некорректная пара {cfg.symbol!r}: {error}")
        raise typer.Exit(code=1) from error
    if candles is None:
        path = storage.path_for(EXCHANGE_ID, cfg.symbol, cfg.timeframe)
        typer.echo(
            f"Нет данных: {path} не найден.\n"
            f"Сначала скачай историю: trading-bot download "
            f"--symbol {cfg.symbol} --timeframe {cfg.timeframe} --since <YYYY-MM-DD>"
        )
        raise typer.Exit(code=1)
    return candles


def _load_candles_for_period(cfg: BacktestConfig) -> pd.DataFrame:
    """Загрузить свечи для symbol/timeframe из конфига, срезанные до периода.

    Команда ``sweep`` вместо этого грузит через :func:`_load_candles` и
    оставляет единственный срез :func:`run_sweep`.
    """
    candles = _load_candles(cfg)
    sliced = slice_candles(candles, cfg)
    if sliced.empty:
        typer.echo(
            f"В данных {cfg.symbol} {cfg.timeframe} нет свечей за период "
            f"{cfg.start} .. {cfg.end or 'конец'}."
        )
        raise typer.Exit(code=1)
    return sliced


def _parse_grid(
    specs: list[str] | None, *, reserved: frozenset[str] = _RESERVED_PARAM_NAMES
) -> dict[str, list]:
    """Разобрать повторяемые опции ``--param name=v1,v2,...`` в словарь сетки.

    Значения приводятся к типам: строки, похожие на int, становятся ``int``,
    остальные числовые — ``float``, всё прочее остаётся строкой. Финальные
    типы и значения проверяет конструктор стратегии. Имена, конфликтующие
    с колонками результата (метрики и ``error``; для walkforward также
    колонки окон), отбрасываются через ``reserved``.
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
        if name in reserved:
            raise typer.BadParameter(
                f"--param {name!r} конфликтует с колонкой результата; "
                "выбери другое имя параметра"
            )
        grid[name] = [_coerce_scalar(item) for item in values]
    return grid


def _coerce_scalar(raw: str) -> int | float | str:
    """Привести значение сетки к типу: int-подобные строки к int, числовые к float, иначе str.

    Не конечные числа (``nan``, ``inf``) отбрасываются: они отравили бы все
    комбинации sweep вместо того, чтобы один раз громко упасть.
    """
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        value = float(raw)
    except ValueError:
        return raw
    if not math.isfinite(value):
        raise typer.BadParameter(
            f"значение {raw!r} должно быть конечным числом "
            "(nan/inf в сетке параметров не допускаются)"
        )
    return value


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
    symbol: Annotated[
        str | None, typer.Option(help="Переопределить пару из конфига, напр. ETH/USDT.")
    ] = None,
    timeframe: Annotated[
        str | None, typer.Option(help="Переопределить таймфрейм из конфига, напр. 1h.")
    ] = None,
) -> None:
    """Прогнать бэктест по сетке параметров стратегии и свести результаты."""
    cfg = _apply_overrides(load_config(config), symbol, timeframe)
    grid = _parse_grid(param)
    if not grid:
        typer.echo(
            "Укажи хотя бы один параметр сетки: "
            "--param fast=10,20,30 --param slow=50,100"
        )
        raise typer.Exit(code=1)

    candles = _load_candles(cfg)

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
    """Записать results.csv и meta.json (конфиг + сетка) в reports/sweep/last/."""
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
    """Напечатать таблицу результатов sweep и лучшую строку по полной доходности.

    Для сеток больше 20 строк показываются только топ-10 успешных
    комбинаций; упавшие комбинации всегда выводятся в конце с укороченным
    текстом ошибки.
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
    """Перевести возможно-NaN ячейку sweep в ``float | None``."""
    return None if pd.isna(value) else float(value)


def _sweep_cell(row: pd.Series, name: str) -> str:
    """Отформатировать одну ячейку таблицы sweep (NaN как тире, float с 2 знаками)."""
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


@app.command()
def walkforward(
    config: Annotated[
        Path, typer.Option(help="Path to the backtest YAML config.")
    ] = Path("config/backtest.yaml"),
    param: Annotated[
        list[str] | None,
        typer.Option(
            "--param",
            help="Параметр сетки 'name=v1,v2,...' (повторяемый): "
            "--param fast=10,20,30 --param slow=50,100.",
        ),
    ] = None,
    is_days: Annotated[
        int, typer.Option(help="Длина ин-семпл окна (обучение), в днях.")
    ] = 180,
    oos_days: Annotated[
        int, typer.Option(help="Длина аут-оф-семпл окна (проверка), в днях.")
    ] = 60,
    mode: Annotated[
        str, typer.Option(help="Режим окон: rolling (скользящий) или anchored (якорный).")
    ] = "rolling",
    objective: Annotated[
        str, typer.Option(help="Метрика выбора лучшей комбинации: sharpe или total_return_pct.")
    ] = "sharpe",
    symbol: Annotated[
        str | None, typer.Option(help="Переопределить пару из конфига, напр. ETH/USDT.")
    ] = None,
    timeframe: Annotated[
        str | None, typer.Option(help="Переопределить таймфрейм из конфига, напр. 1h.")
    ] = None,
) -> None:
    """Walk-forward: перебор сетки на IS, проверка лучшей на OOS, сшитая кривая.

    История делится на окна «ин-семпл → аут-оф-семпл». На каждом шаге сетка
    параметров прогоняется на IS, лучшая комбинация (по --objective)
    проверяется на следующем OOS-отрезке; сшитая кривая из OOS-частей —
    оценка стратегии с периодической переоптимизацией без заглядывания
    в будущее.
    """
    cfg = _apply_overrides(load_config(config), symbol, timeframe)
    grid = _parse_grid(param, reserved=_WALKFORWARD_RESERVED_PARAM_NAMES)
    if not grid:
        typer.echo(
            "Укажи хотя бы один параметр сетки: "
            "--param fast=10,20,30 --param slow=50,100"
        )
        raise typer.Exit(code=1)

    candles = _load_candles(cfg)

    try:
        wf = run_walkforward(
            candles,
            cfg,
            grid,
            is_days=is_days,
            oos_days=oos_days,
            mode=mode,
            objective=objective,
        )
    except ValueError as error:
        typer.echo(f"Ошибка walkforward: {error}")
        raise typer.Exit(code=1) from error

    _save_walkforward_artifacts(cfg, grid, wf)
    _print_walkforward(cfg, grid, wf, candles)


def _save_walkforward_artifacts(
    cfg: BacktestConfig, grid: dict[str, list], wf: WalkForwardResult
) -> None:
    """Записать results.csv, stitched_equity.parquet и meta.json в reports/walkforward/last/."""
    WALKFORWARD_DIR.mkdir(parents=True, exist_ok=True)
    wf.to_frame(grid).to_csv(WALKFORWARD_DIR / "results.csv", index=False)
    meta = {"config": cfg.model_dump(mode="json"), "walkforward": wf.meta}
    with (WALKFORWARD_DIR / "meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)
    if not wf.stitched_equity.empty:
        wf.stitched_equity.to_frame().to_parquet(
            WALKFORWARD_DIR / "stitched_equity.parquet", engine="pyarrow"
        )
    else:
        # Полностью упавший прогон не должен оставлять артефакты предыдущего.
        for stale in ("stitched_equity.parquet", "walkforward.png"):
            (WALKFORWARD_DIR / stale).unlink(missing_ok=True)
    typer.echo(f"Результаты: {WALKFORWARD_DIR / 'results.csv'}")


def _print_walkforward(
    cfg: BacktestConfig,
    grid: dict[str, list],
    wf: WalkForwardResult,
    candles: pd.DataFrame,
) -> None:
    """Напечатать таблицу по окнам плюс сводку stitched-кривой против buy & hold."""
    ok = [w for w in wf.windows if w.error is None]
    failed = [w for w in wf.windows if w.error is not None]

    table = Table(
        title=f"Walk-forward: {cfg.strategy} · {cfg.symbol} · {cfg.timeframe} · "
        f"{wf.meta['mode']} · IS {wf.meta['is_days']}д / OOS {wf.meta['oos_days']}д "
        f"({len(wf.windows)} окон)",
        title_justify="left",
    )
    table.add_column("OOS начало", no_wrap=True)
    table.add_column("OOS конец", no_wrap=True)
    for name in grid:
        table.add_column(name, justify="right", no_wrap=True)
    table.add_column(f"IS {wf.meta['objective']}", justify="right")
    table.add_column("OOS доходность", justify="right")
    table.add_column("OOS Шарп", justify="right")
    table.add_column("Сделок", justify="right")
    table.add_column("Ошибка", no_wrap=True, max_width=48, overflow="fold")

    for window in ok:
        table.add_row(
            f"{window.oos_start:%Y-%m-%d}",
            f"{window.oos_end:%Y-%m-%d}",
            *(_fmt_grid_param(window.best_params, name) for name in grid),
            _fmt_wf_num(window.is_objective),
            _fmt_wf_num(window.oos_return_pct, signed=True),
            _fmt_wf_num(window.oos_sharpe),
            str(window.oos_trades if window.oos_trades is not None else 0),
            "",
        )
    for window in failed:
        table.add_row(
            f"{window.oos_start:%Y-%m-%d}",
            f"{window.oos_end:%Y-%m-%d}",
            *("—" for _ in grid),
            "—",
            "—",
            "—",
            "—",
            str(window.error),
        )

    Console().print(table)
    if failed:
        typer.echo(f"Окон с ошибкой: {len(failed)} (в конце таблицы).")

    if not ok:
        typer.echo("Ни одно окно не выполнилось успешно — см. тексты ошибок выше.")
        return

    _print_walkforward_summary(cfg, wf, candles, ok)


def _print_walkforward_summary(
    cfg: BacktestConfig, wf: WalkForwardResult, candles: pd.DataFrame, ok: list
) -> None:
    """Напечатать stitched-метрики против buy & hold за тот же OOS-период + график."""
    stitched = wf.stitched_equity
    wf_start, wf_end = ok[0].oos_start, ok[-1].oos_end
    closes = candles.loc[
        (candles["timestamp"] >= wf_start) & (candles["timestamp"] < wf_end)
    ].set_index("timestamp")["close"]
    benchmark = benchmark_equity(closes.astype("float64"), 1.0)

    strategy_metrics = compute_metrics(stitched, wf.trades, cfg.timeframe)
    benchmark_metrics = compute_benchmark_metrics(benchmark, cfg.timeframe)

    table = Table(title="Walk-forward (сшитая OOS-кривая) vs Buy & hold", title_justify="left")
    table.add_column("Метрика", no_wrap=True)
    table.add_column("Стратегия", justify="right")
    table.add_column("Buy & hold", justify="right")
    table.add_row(
        "Доходность",
        _fmt_pct(strategy_metrics.total_return_pct),
        _fmt_pct(benchmark_metrics.total_return_pct),
    )
    table.add_row(
        "Годовая доходность (CAGR)",
        _fmt_opt(strategy_metrics.cagr_pct, _fmt_pct),
        _fmt_opt(benchmark_metrics.cagr_pct, _fmt_pct),
    )
    table.add_row(
        "Коэффициент Шарпа (аннуал.)",
        _fmt_opt(strategy_metrics.sharpe, lambda v: f"{v:.2f}"),
        _fmt_opt(benchmark_metrics.sharpe, lambda v: f"{v:.2f}"),
    )
    table.add_row(
        "Максимальная просадка",
        _fmt_pct(strategy_metrics.max_drawdown_pct),
        _fmt_pct(benchmark_metrics.max_drawdown_pct),
    )
    Console().print(table)

    n_trades = [w.oos_trades or 0 for w in ok]
    typer.echo(
        f"Сшитая кривая из {len(ok)} OOS-отрезков, "
        f"{sum(n_trades)} сделок (в среднем {sum(n_trades) / len(ok):.1f} на окно)."
    )
    typer.echo(
        "OOS-метрики считаются от первой свечи OOS-отрезка: lead-in перед окном "
        "разогревает индикаторы (движок стартует с пустого портфеля), его PnL "
        "до oos_start отбрасывается; сделка, открытая в lead-in и закрытая в OOS, "
        "учитывается в кривой, но не в trade-метриках."
    )

    plot_stitched_equity(stitched, WALKFORWARD_DIR / "walkforward.png", benchmark=benchmark)
    typer.echo(f"График stitched-кривой: {WALKFORWARD_DIR / 'walkforward.png'}")


def _fmt_wf_num(value: float | None, signed: bool = False) -> str:
    """Отформатировать необязательную walk-forward метрику (2 знака, опциональный знак)."""
    if value is None:
        return "—"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def _fmt_grid_param(params: dict | None, name: str) -> str:
    return "—" if params is None else str(params.get(name, "—"))


def _summary(cfg: BacktestConfig, result: BacktestResult) -> dict:
    """Собрать JSON-совместимую сводку прогона."""
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
    """Записать equity.parquet, trades.csv, benchmark.parquet и meta.json."""
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
    """Загрузить артефакты прогона; выйти с подсказкой, если их нет или они битые."""
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
    """Загрузить опциональный артефакт бенчмарка buy & hold (``None``, если его нет)."""
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
    """Разрешить файл свечей для графика сделок, с подсказками при отсутствии."""
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
    """Напечатать таблицу метрик в консоль (rich — зависимость typer)."""
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
    """Напечатать таблицу сравнения стратегии с buy & hold."""
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
    """Точка входа скрипта."""
    app()
