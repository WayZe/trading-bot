# AGENTS.md

Инструкции для агентских сессий, работающих в этом репозитории.

## Что это

Учебный крипто-торговый бот: биржа **Bybit** (спот), свинг-стратегии на свечах,
подход **backtest-first**. MVP — офлайн-бэктестер с честной моделью исполнения;
сверху — слой исследований (sweep по сетке параметров, walk-forward валидация).
Спецификация: `docs/superpowers/specs/2026-09-05-trading-bot-design.md`.

## Команды

```bash
uv sync                      # установить зависимости (Python 3.12+)
uv run pytest                # тесты (офлайн — сеть не нужна)
uv run ruff check .          # линтер
```

CLI (`uv run trading-bot ...`):

- `download --symbol BTC/USDT --timeframe 4h --since 2024-01-01` — полная загрузка истории;
  инкрементальное обновление существующего датасета: `--update` (без `--since`).
- `backtest --config config/backtest.yaml` — бэктест; `--symbol`/`--timeframe`
  переопределяют конфиг.
- `report` — метрики и графики по артефактам `reports/last_run/`.
- `sweep --config ... --param fast=10,15,20 --param slow=30,50` — перебор сетки параметров.
- `walkforward --config ... --param ... --is-days 180 --oos-days 60 --mode rolling --objective sharpe`
  — walk-forward валидация (IS→OOS окна, stitched-кривая).

Make-цели (см. Makefile): `make sync`, `make test`, `make lint`, `make download`,
`make backtest`, `make report`, `make sweep`, `make walkforward`, `make smoke`,
`make docker-build`, `make docker-run CMD=...`.

## Архитектура

Зависимости направлены сверху вниз (CLI → engine → data), `src/trading_bot/`:

- `data/` — слой данных:
  - `exchange.py` — тонкая обёртка над `ccxt.bybit` (публичный OHLCV, ключи не нужны);
  - `storage.py` — Parquet-хранилище (`data/{exchange}/{SYMBOL_SLUG}/{timeframe}.parquet`),
    каноничная схема, дедупликация и сортировка, `validate_ohlcv`/`find_gaps`;
  - `downloader.py` — пагинация по `since_ms`, ретраи с экспоненциальным backoff
    (NetworkError, включая RateLimitExceeded), дозаполнение гэпов, отбрасывание
    ещё не закрытых свечей (закрыта ⟺ `timestamp + timeframe <= now`).
- `indicators.py` — векторные индикаторы (sma, ema, rsi, atr) с единой семантикой
  прогрева (ведущие NaN).
- `strategy/` — плагины стратегий:
  - `base.py` — контракт: `Strategy` c `on_candle(candles[:i+1]) -> list[Signal]`,
    `on_fill`, `reset`, `warmup_period`; данные-классы `Signal`/`Fill`;
  - реестр `STRATEGY_REGISTRY` в `strategy/__init__.py` (запись — класс
    стратегии или фабрика-функция, напр. `sma_cross_trend`), фабрика
    `create_strategy(name, params)`;
  - новая стратегия: класс от `Strategy`, регистрация в реестре, параметры —
    в `strategy_params` конфига (yaml). Пример в README. Композитные
    стратегии — обёрткой над внутренней (`trend_filter.py: TrendFiltered`
    гейтит только `LONG_ENTRY` по SMA-тренду, выходы не фильтруются).
- `engine/` — бэктест:
  - `broker.py` — симуляция исполнения: комиссия `fee_rate` за сторону,
    проскальзывание `slippage_bps` против нас;
  - `portfolio.py` — кэш, одна long-позиция, `TradeRecord` (pnl нетто по обеим комиссиям);
  - `backtest.py` — событийный цикл: (1) исполнение queued-ордера по open,
    (2) интрабарные стоп/тейк, (3) mark-to-market по close, (4) вызов стратегии.
    **Движок — единственный владелец SL/TP**: дистанция от close сигнальной свечи
    переякоривается на цену исполнения (`active_stop = fill − (ref_close − stop)`).
- `risk.py` — размер позиции: доля equity, округление вниз до `quantity_precision`,
  отсечка по `min_notional` (0 = не торговать).
- `report/` — `metrics.py` (чистые функции: доходность, CAGR, Шарп, просадка,
  метрики сделок, benchmark buy & hold) и `plots.py` (matplotlib, backend Agg).
- `research/` — исследования: `sweep.py` (разворот сетки, срез периода, строки
  с ошибками вместо падений) и `walkforward.py` (планировщик IS→OOS окон,
  lead-in из `warmup_period`, stitched-кривая).
- `cli.py` — команды typer + сохранение артефактов (`reports/last_run`,
  `reports/sweep/last`, `reports/walkforward/last`).

## Инварианты, которые нельзя ломать

- **Никакого look-ahead**: стратегия видит только закрытые свечи до текущей
  включительно (`candles[:i+1]`); сигнал по close свечи `i` исполняется по
  open свечи `i+1`. Любая правка движка/стратегий обязана сохранять это
  (тесты `TestLookAhead`, parity-тесты sweep/walkforward следят).
- **В датасет попадают только закрытые свечи** (downloader и `--update`).
- **Движок — единственный владелец SL/TP**: уровни приходят в `Signal` как
  дистанции от close сигнальной свечи и переякориваются на фактическую цену
  исполнения; стратегия не имеет своего пути выхода по пробою стопа.
- **Рантайм-строки — на английском**: reason-строки сделок (`"sma cross up"`,
  `"stop loss"`, ...), тексты `ValueError`/логов, CLI help в `Annotated[...]`.
  Они хранятся в trades.csv/results.csv и проверяются тестами через
  `pytest.raises(match=...)` — перевод ломает тесты и данные.
- **Одна long-позиция**: пирамидинг не поддерживается, частичные закрытия запрещены.

## Конвенции кода

- Идентификаторы и рантайм-строки — английские; комментарии, docstrings
  (Google-стиль: ключевые слова `Args:`/`Returns:`/`Raises:`/`Attributes:` не
  переводить, содержимое — на русском) и документация — русские.
- В библиотечном коде — `logging`, не `print`; пользовательский вывод CLI —
  `typer.echo`/`rich`.
- pandas 3.x-совместимость: `Timedelta` сравнивать только с `pd.Timedelta`
  (никаких сравнений с `datetime.timedelta` напрямую).
- Новые зависимости — только при реальной необходимости (добавлять в
  `pyproject.toml`, коммитить `uv.lock`).
- Docstrings в Google-стиле.

## Тесты

- Офлайн: сеть мокается (`FakeExchange`/`GapExchange`, `pytest-mock`).
- Точная математика на литералах: посчитанные вручную значения индикаторов,
  PnL, метрик — не менять «примерные» проверки.
- Parity-тесты: sweep и walkforward должны совпадать с прямым прогоном
  движка на тех же данных (`test_single_combination_matches_direct_engine_run`,
  `test_single_window_single_combo_matches_direct_engine_run`).
- Гейты перед коммитом: `uv run pytest` и `uv run ruff check .` — зелёные.

## Артефакты

- `data/` и `reports/` — в `.gitignore` (локальные датасеты и результаты прогонов).
- `uv.lock` — в git (воспроизводимая установка).
