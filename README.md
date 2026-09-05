# trading-bot

Учебный крипто-торговый бот: биржа **Bybit** (спот), свинг-стратегии на свечах,
подход **backtest-first** — сначала проверяем идеи на истории, только потом
(в перспективе) paper- и live-торговля. MVP проекта — офлайн-бэктестер с
честной моделью исполнения.

> Это образовательный проект. Ничего здесь — не финансовый совет;
> торговля криптой рискованна, можете потерять деньги.

## Статус и роадмап

- [x] Этап 0 — каркас проекта, спецификация, зависимости
- [x] Этап 1 — слой данных (загрузка OHLCV с Bybit, Parquet-хранилище, CLI `download`)
- [x] Этап 2 — индикаторы (SMA, EMA, RSI, ATR) и стратегия `sma_cross`
- [x] Этап 3 — бэктест-движок (event loop, broker, portfolio, risk) и CLI `backtest`
- [x] Этап 4 — отчёты: метрики прогона, графики эквити/просадки и сделок, CLI `report`
- [ ] Этап 5 — paper-трейдинг на Bybit testnet
- [ ] Этап 6 — live-торговля малой суммой (с жёсткими ограничениями риска)

Дизайн проекта: [docs/superpowers/specs/2026-09-05-trading-bot-design.md](docs/superpowers/specs/2026-09-05-trading-bot-design.md)

## Быстрый старт

Требуется [uv](https://docs.astral.sh/uv/) и Python 3.12+.

```bash
uv sync
```

Дальше три команды: скачать историю → прогнать бэктест → посмотреть отчёт.

**1. Скачать свечи с Bybit** (публичный API, ключи не нужны):

```bash
uv run trading-bot download --symbol BTC/USDT --timeframe 4h --since 2025-01-01
```

Данные ложатся в Parquet: `data/bybit/BTC_USDT/4h.parquet`. Сохраняются только
закрытые свечи — формирующаяся последняя свеча отбрасывается
(свеча закрыта ⟺ `timestamp + timeframe ≤ now`).

Инкрементальное обновление существующего датасета (`--since` не нужен,
если датасет уже есть):

```bash
uv run trading-bot download --symbol BTC/USDT --timeframe 4h --update
```

**2. Прогнать бэктест** (конфиг — `config/backtest.yaml`):

```bash
uv run trading-bot backtest --config config/backtest.yaml
```

Пару и таймфрейм из конфига можно переопределить флагами — данные берутся из
соответствующего датасета:

```bash
uv run trading-bot backtest --config config/backtest.yaml --symbol ETH/USDT --timeframe 1h
```

Артефакты прогона (кривая эквити, сделки, бенчмарк buy & hold, meta) пишутся
в `reports/last_run/`.

**3. Посмотреть отчёт** — таблица метрик + графики:

```bash
uv run trading-bot report
```

Команда читает артефакты из `reports/last_run/`, печатает метрики (доходность,
CAGR, Шарп, просадка, winrate, profit factor, комиссии, ...) и сохраняет
`reports/last_run/equity.png` (эквити + просадка + оверлей buy & hold) и
`reports/last_run/trades.png` (сделки на графике цены). Рядом с основной
таблицей выводится сравнение «Стратегия vs Buy & hold» (доходность, CAGR,
макс. просадка, Шарп) за тот же период. Свечи для графика сделок ищутся
автоматически в `data/`; можно указать файл явно:

```bash
uv run trading-bot report --run-dir reports/last_run --candles data/bybit/BTC_USDT/4h.parquet
```

> Примечание: бенчмарк buy & hold — упрощённая модель (разовая покупка на весь
> капитал по первой свече) без комиссий и проскальзывания, тогда как стратегия
> платит их за каждую сделку. Поэтому сравнение систематически смещено в пользу
> buy & hold — читайте его как оценку снизу для пассивной позиции.

**4. Перебор параметров стратегии (sweep)** — сетка `--param имя=v1,v2,...`
(повторяемый флаг), всё остальное берётся из конфига:

```bash
uv run trading-bot sweep --config config/backtest.yaml \
  --param fast=10,15,20 --param slow=30,50
```

Невалидные комбинации (например `fast >= slow`) не прерывают прогон — они
попадают в результаты с текстом ошибки. Результаты пишутся в
`reports/sweep/last/results.csv` (все комбинации и метрики) и печатаются
таблицей с лучшей комбинацией по доходности. Для больших сеток в таблице
показываются только топ-10, полные результаты — в CSV.

**5. Walk-forward валидация** — честная проверка сетки без заглядывания в
будущее. История делится на окна «ин-семпл (IS) → аут-оф-семпл (OOS)»: на IS
перебирается сетка параметров, лучшая комбинация прогоняется на следующем
OOS-отрезке. Сшитая (stitched) кривая из OOS-частей — оценка того, что
стратегия реально дала бы при периодической переоптимизации:

```bash
uv run trading-bot walkforward --config config/backtest.yaml \
  --param fast=10,20,30 --param slow=50,100 \
  --is-days 180 --oos-days 60 --mode rolling --objective sharpe
```

- `--is-days` / `--oos-days` — длины окон обучения и проверки в днях
  (дефолты 180/60); в rolling-режиме блоки IS→OOS идут встык, следующий IS
  никогда не видит OOS-данные предыдущего окна; в `anchored`-режиме IS растёт
  от начала истории.
- `--objective` — метрика выбора лучшей комбинации на IS: `sharpe` или
  `total_return_pct`.
- `--symbol` / `--timeframe` — переопределение пары/таймфрейма из конфига.

Перед каждым OOS-отрезком движок получает lead-in из `warmup_period` свечей
стратегии и стартует каждое окно с пустого портфеля: позиция может только
открыться за время lead-in (индикаторы «разогреваются» на прошлых данных).
PnL отбрасывается только до `oos_start`: сделка, открытая в lead-in и
закрытая в OOS, вкладывает свою OOS-часть в equity и доходность, но
исключена из trade-метрик — поэтому окно может показать «Сделок: 0» при
ненулевой доходности.

Результаты пишутся в `reports/walkforward/last/`: `results.csv` (окна,
лучшие параметры, IS objective, OOS-метрики), `stitched_equity.parquet`,
`meta.json` и `walkforward.png` (сшитая кривая vs buy & hold за тот же
период). В конце печатается сводка: stitched-доходность/CAGR/Шарп/просадка
против buy & hold и среднее число сделок на окно.

Как читать результат: если stitched-доходность заметно хуже результата того
же sweep на всей истории — сетка на всей истории переобучилась (выбрала то,
что случайно выстрелило в прошлом). Walk-forward ближе к тому, что было бы
на живом графике, но и он — не гарантия: рынки меняются, окон мало.

## Архитектура

Слои изолированы, зависимости направлены сверху вниз (CLI → engine → data):

```
src/trading_bot/
  cli.py           CLI (typer): download, backtest, report, sweep, walkforward
  config.py        pydantic-модель конфига бэктеста (config/backtest.yaml)
  indicators.py    векторные индикаторы: sma, ema, rsi, atr
  risk.py          размер позиции (доля equity, min_notional, округление)
  data/            слой данных: exchange (ccxt/bybit), downloader (пагинация,
                   ретраи, заполнение гэпов), storage (Parquet)
  engine/          бэктест: broker (комиссия+проскальзывание), portfolio
                   (кэш, позиция, TradeRecord), backtest (event loop)
  strategy/        плагины стратегий: base (Signal/Fill/Strategy ABC),
                   sma_cross, реестр по имени
  research/        слой исследований: sweep по сетке параметров,
                   walk-forward валидация (IS→OOS окна, stitched-кривая)
  report/          метрики прогона (metrics.py) и графики (plots.py)
```

Стратегия не знает, кто её вызывает — бэктест-движок или будущий live-движок;
исполнение полностью принадлежит engine и его broker. Подробности — в
[спецификации](docs/superpowers/specs/2026-09-05-trading-bot-design.md).

## Как добавить свою стратегию

Стратегия — класс от `Strategy`: метод `on_candle` получает все свечи
до закрытой включительно и возвращает список `Signal` (намерения, не исполнение).
Пример — моментум за 10 свечей:

```python
# src/trading_bot/strategy/momentum.py
from trading_bot.strategy.base import Signal, SignalKind, Strategy


class MomentumStrategy(Strategy):
    name = "momentum"

    def __init__(self, lookback: int = 10, threshold: float = 0.02) -> None:
        self.lookback = lookback
        self.threshold = threshold

    @property
    def warmup_period(self) -> int:
        return self.lookback

    def on_candle(self, candles) -> list[Signal]:
        close = candles["close"]
        change = close.iloc[-1] / close.iloc[-self.lookback] - 1.0
        if change > self.threshold:
            return [Signal(SignalKind.LONG_ENTRY, reason=f"рост {change:.1%} за {self.lookback} свечей")]
        if change < -self.threshold:
            return [Signal(SignalKind.LONG_EXIT, reason=f"падение {change:.1%}")]
        return []
```

Зарегистрируйте класс в реестре:

```python
# src/trading_bot/strategy/__init__.py
from trading_bot.strategy.momentum import MomentumStrategy

STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "sma_cross": SmaCrossStrategy,
    "momentum": MomentumStrategy,  # новая стратегия
}
```

И включите её в конфиг:

```yaml
# config/backtest.yaml
strategy: momentum
strategy_params:
  lookback: 10
  threshold: 0.02
```

Параметры из `strategy_params` передаются в конструктор как есть. Опционально
можно переопределить `on_fill` (фактические исполнения) и `reset`
(сброс состояния между прогонами).

## Честность бэктеста

Чтобы результатам можно было верить, движок моделирует исполнение консервативно:

- **Нет заглядывания в будущее**: сигнал считается по close свечи `i`,
  исполняется по open свечи `i+1`.
- **Комиссия** — taker за каждую сторону (`fee_rate`, по умолчанию 0.1%).
- **Проскальзывание** — в базисных пунктах против нас: покупка выше, продажа
  ниже (`slippage_bps`).
- **Стопы** принадлежат движку: дистанция стопа, заданная стратегией от
  close сигнальной свечи (`close − atr_mult·ATR`), переносится на
  фактическую цену исполнения — `active_stop = fill − дистанция`
  (для тейка симметрично). Уровни проверяются внутри свечи (`low ≤ stop`);
  гэп через уровень исполняется по open (худшая цена), при одновременном
  касании стопа и тейка приоритет у стопа.
- **Валидация входа**: пустой датасет, NaN в обязательных колонках или
  `high < low` — явная ошибка, а не тихо испорченный прогон.
- **Одна long-позиция**, размер ограничен долей equity (`position_size_pct`)
  и `min_notional` биржи; equity переоценивается по close каждой свечи.
- Ордер, оставшийся неисполненным на последней свече, не считается сделкой.

## Разработка

```bash
uv run pytest          # тесты (офлайн, сеть не нужна)
uv run ruff check .    # линтер
```

## Структура репозитория

```
config/            конфиги (backtest.yaml)
data/              скачанные свечи (gitignored)
reports/           артефакты прогонов и графики (gitignored)
docs/              дизайн-документы
src/trading_bot/   исходный код (см. «Архитектура»)
tests/             тесты, зеркалят структуру src
```

## Дисклеймер

Проект создан для изучения алготрейдинга. Это не финансовый совет,
не инвестиционная рекомендация и не готовый торговый инструмент.
Авторы не несут ответственности за возможные убытки при использовании кода.
