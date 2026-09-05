# trading-bot

Учебный крипто-торговый бот: биржа Bybit (спот), свинг-стратегии на свечах,
backtest-first подход. Цель проекта — научиться алготрейдингу: от сбора данных
и бэктеста до (в перспективе) paper-трейдинга и live-торговли малой суммой.

## Статус

- [x] Этап 0 — каркас проекта, спецификация, зависимости
- [x] Этап 1 — слой данных (загрузка OHLCV с Bybit, Parquet-хранилище, CLI `download`)
- [ ] Этап 2 — индикаторы (SMA, EMA, RSI, ATR)
- [ ] Этап 3 — бэктест-движок (engine, broker, portfolio, risk)
- [ ] Этап 4 — отчёты (метрики, графики)
- [ ] Этап 5+ — стратегии, paper/live

Дизайн проекта: [docs/superpowers/specs/2026-09-05-trading-bot-design.md](docs/superpowers/specs/2026-09-05-trading-bot-design.md)

## Быстрый старт

Требуется [uv](https://docs.astral.sh/uv/) и Python 3.12+.

```bash
uv sync
```

Скачать исторические свечи с Bybit (публичный API, ключи не нужны):

```bash
uv run trading-bot download --symbol BTC/USDT --timeframe 4h --since 2025-08-01
```

Инкрементальное обновление существующего датасета:

```bash
uv run trading-bot download --symbol BTC/USDT --timeframe 4h --since 2025-08-01 --update
```

Данные складываются в Parquet: `data/bybit/BTC_USDT/4h.parquet`.

## Разработка

```bash
uv run pytest          # тесты (офлайн, сеть не нужна)
uv run ruff check .    # линтер
```

## Структура

```
config/            примеры конфигов (backtest.yaml)
data/              скачанные свечи (gitignored)
reports/           отчёты бэктестов (gitignored)
docs/              дизайн-документы
src/trading_bot/
  cli.py           CLI (typer)
  config.py        pydantic-модели конфигурации
  data/            слой данных: exchange, downloader, storage
tests/
```
