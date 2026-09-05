"""Подмены биржи и фабрики раннера для офлайн-тестов live-контура."""

from __future__ import annotations

import ccxt

from tests.conftest import BASE_MS, HOUR_MS
from trading_bot.config import LiveConfig
from trading_bot.data.exchange import ExchangeClient
from trading_bot.data.storage import CandleStorage
from trading_bot.engine.broker import SimulatedBroker
from trading_bot.live.execution import PaperAdapter
from trading_bot.live.runner import LiveRunner
from trading_bot.live.state import load_or_fresh_state
from trading_bot.strategy import create_strategy

FOUR_HOUR_MS = 4 * HOUR_MS

# Малые периоды donchian: сигналы детерминированы на нескольких свечах.
DONCHIAN_TEST_PARAMS = {
    "entry_period": 2,
    "exit_period": 1,
    "atr_period": 1,
    "atr_mult": 2.0,
}


def donchian_rows(closes: list[float]) -> list[list]:
    """Собрать сырые свечи ``[ts, o, h, l, c, v]`` по списку close (4h с BASE_MS)."""
    rows: list[list] = []
    for i, close in enumerate(closes):
        open_ = closes[i - 1] if i > 0 else close
        high = max(open_, close) + 1.0
        low = min(open_, close) - 1.0
        rows.append([BASE_MS + i * FOUR_HOUR_MS, open_, high, low, close, 10.0])
    return rows


def append_candle(fake: FakeCcxt, closes: list[float]) -> None:
    """Дописать к подмене последнюю свечу списка ``closes`` (новая закрытая свеча)."""
    fake.rows.append(donchian_rows(closes)[-1])


class FakeCcxt:
    """Публичный уровень ccxt: OHLCV и тикер; приватные методы запрещены.

    Каждый приватный вызов фиксируется в ``private_calls`` и роняет тест:
    так paper-инвариант «никаких приватных API» доказывается явно.
    """

    def __init__(self, rows: list[list], ticker_price: float = 100.0) -> None:
        self.rows = list(rows)
        self.ticker_price = ticker_price
        self.ohlcv_calls: list[int | None] = []
        self.private_calls: list[str] = []

    def fetch_ohlcv(self, symbol, timeframe="4h", since=None, limit=1000) -> list[list]:
        self.ohlcv_calls.append(since)
        return [r for r in self.rows if since is None or r[0] >= since][:limit]

    def fetch_ticker(self, symbol) -> dict:
        return {"last": self.ticker_price}

    def create_order(self, *args, **kwargs) -> dict:
        self.private_calls.append("create_order")
        raise AssertionError("private create_order must not be called in paper mode")

    def fetch_order(self, *args, **kwargs) -> dict:
        self.private_calls.append("fetch_order")
        raise AssertionError("private fetch_order must not be called in paper mode")

    def fetch_balance(self, *args, **kwargs) -> dict:
        self.private_calls.append("fetch_balance")
        raise AssertionError("private fetch_balance must not be called in paper mode")


class FailingOrderCcxt(FakeCcxt):
    """Testnet-подмена: публичные данные и баланс работают, ордер отклоняется."""

    def __init__(self, rows, ticker_price=100.0, free_usdt=0.0) -> None:
        super().__init__(rows, ticker_price)
        self.free_usdt = free_usdt
        self.reject = True
        self.created_orders: list[tuple] = []

    def create_order(self, symbol, order_type, side, quantity, *args, **kwargs) -> dict:
        if self.reject:
            self.private_calls.append("create_order")
            raise ccxt.ExchangeError("order rejected by exchange")
        self.created_orders.append((symbol, order_type, side, quantity))
        return {"id": f"ord-{len(self.created_orders)}"}

    def fetch_order(self, order_id, symbol) -> dict:
        return {"id": order_id, "status": "closed"}

    def fetch_balance(self) -> dict:
        return {"free": {"USDT": self.free_usdt}}


class PrivateCcxt(FakeCcxt):
    """Testnet-подмена с рабочими приватными методами и заданными ответами."""

    def __init__(
        self,
        rows,
        ticker_price=100.0,
        order_status: dict | None = None,
        free_usdt: float = 0.0,
        fetch_order_exc: Exception | None = None,
    ) -> None:
        super().__init__(rows, ticker_price)
        self.order_status = order_status or {}
        self.free_usdt = free_usdt
        self.fetch_order_exc = fetch_order_exc
        self.created_orders: list[tuple] = []

    def create_order(self, symbol, order_type, side, quantity, *args, **kwargs) -> dict:
        self.created_orders.append((symbol, order_type, side, quantity))
        return {"id": f"ord-{len(self.created_orders)}"}

    def fetch_order(self, order_id, symbol) -> dict:
        if self.fetch_order_exc is not None:
            raise self.fetch_order_exc
        return {"id": order_id, "status": "closed", **self.order_status}

    def fetch_balance(self) -> dict:
        return {"free": {"USDT": self.free_usdt}}


def make_config(tmp_path, **overrides) -> LiveConfig:
    """Собрать live-конфиг с путями в ``tmp_path`` и тестовыми параметрами donchian."""
    base = {
        "symbol": "BTC/USDT",
        "timeframe": "4h",
        "strategy": "donchian",
        "strategy_params": dict(DONCHIAN_TEST_PARAMS),
        "mode": "paper",
        "start_cash": 10_000.0,
        "data_root": str(tmp_path / "live" / "data"),
        "state_path": str(tmp_path / "live" / "state.json"),
        "kill_switch_path": str(tmp_path / "live" / "STOP"),
        "log_file": str(tmp_path / "live" / "logs" / "live.log"),
    }
    return LiveConfig.model_validate({**base, **overrides})


def build_runner(cfg: LiveConfig, fake, adapter=None) -> LiveRunner:
    """Собрать раннер поверх подмены биржи (state читается с диска, как при рестарте)."""
    client = ExchangeClient()
    client.exchange = fake
    storage = CandleStorage(cfg.data_root)
    state = load_or_fresh_state(cfg.state_path, cfg)
    strategy = create_strategy(cfg.strategy, cfg.strategy_params)
    if adapter is None:
        broker = SimulatedBroker(fee_rate=cfg.fee_rate, slippage_bps=cfg.slippage_bps)
        adapter = PaperAdapter(
            broker,
            price_source=lambda: client.fetch_ticker_last(cfg.symbol),
            equity_source=lambda: state.equity,
        )
    return LiveRunner(cfg, state, strategy, adapter, client, storage)


def make_runner(tmp_path, closes: list[float], ticker_price: float = 100.0, **overrides):
    """Собрать paper-раннер с подменой ccxt; вернуть ``(runner, fake, cfg)``."""
    cfg = make_config(tmp_path, **overrides)
    fake = FakeCcxt(donchian_rows(closes), ticker_price=ticker_price)
    return build_runner(cfg, fake), fake, cfg
