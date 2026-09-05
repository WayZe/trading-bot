"""Тесты цикла live-раннера (офлайн, биржа подменена)."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import BASE_MS, rows_to_df
from tests.live.fakes import (
    DONCHIAN_TEST_PARAMS,
    FOUR_HOUR_MS,
    FailingOrderCcxt,
    append_candle,
    build_runner,
    donchian_rows,
    make_config,
    make_runner,
)
from trading_bot.data.exchange import ExchangeClient
from trading_bot.data.storage import CandleStorage
from trading_bot.engine.backtest import BacktestEngine
from trading_bot.engine.broker import SimulatedBroker
from trading_bot.live.execution import TestnetAdapter as _TestnetAdapter
from trading_bot.live.state import LiveState
from trading_bot.risk import RiskManager
from trading_bot.strategy import create_strategy

FEE_RATE = 0.001
SLIPPAGE_BPS = 5.0
START_CASH = 10_000.0

# Аналитика сценария «6 плоских свечей по 100 + пробойная на 120»:
# канал входа = max(high[4..5]) = 101, close 120 > 101 -> LONG_ENTRY;
# TR пробойной свечи = max(121-99, 121-100, 99-100) = 22, ATR(1) = 22,
# уровень стопа от сигнального close: 120 - 2*22 = 76, дистанция стопа = 44.
FLAT = [100.0] * 6
BREAKOUT_CLOSE = 120.0
STOP_LEVEL_REF = 76.0
STOP_DISTANCE = BREAKOUT_CLOSE - STOP_LEVEL_REF


def _last_candle_iso(n_candles: int) -> str:
    return pd.Timestamp(BASE_MS + (n_candles - 1) * FOUR_HOUR_MS, unit="ms", tz="UTC").isoformat()


class TestEntryCycle:
    def test_entry_signal_opens_position_and_reanchors_stop(self, tmp_path) -> None:
        runner, fake, cfg = make_runner(tmp_path, closes=FLAT, ticker_price=BREAKOUT_CLOSE)
        runner.run_once()  # чистый state: полная догрузка = только прогрев

        assert runner.state.position is None
        assert runner.state.last_candle_ts == _last_candle_iso(len(FLAT))

        append_candle(fake, [*FLAT, BREAKOUT_CLOSE])
        runner.run_once()  # новая закрытая свеча-пробой -> вход

        state = LiveState.load(cfg.state_path)
        assert state.position is not None
        fill_price = BREAKOUT_CLOSE * (1 + SLIPPAGE_BPS / 10_000)
        assert state.position.entry_price == pytest.approx(fill_price)
        # Стоп переякорен на фактическую цену входа (дистанция 44 от close 120).
        assert state.active_stop == pytest.approx(fill_price - STOP_DISTANCE)
        assert state.active_tp is None
        assert state.position.entry_reason == "donchian breakout up"
        assert [t["side"] for t in state.trades] == ["buy"]
        assert state.last_candle_ts == _last_candle_iso(len(FLAT) + 1)

    def test_entry_quantity_follows_risk_manager(self, tmp_path) -> None:
        runner, fake, cfg = make_runner(tmp_path, closes=FLAT, ticker_price=BREAKOUT_CLOSE)
        runner.run_once()
        append_candle(fake, [*FLAT, BREAKOUT_CLOSE])
        runner.run_once()

        expected_qty = math.floor(START_CASH * 0.95 / BREAKOUT_CLOSE * 10**6) / 10**6
        state = LiveState.load(cfg.state_path)
        assert state.position.quantity == pytest.approx(expected_qty)

    def test_exit_signal_closes_position(self, tmp_path) -> None:
        runner, fake, cfg = make_runner(tmp_path, closes=FLAT, ticker_price=BREAKOUT_CLOSE)
        runner.run_once()
        append_candle(fake, [*FLAT, BREAKOUT_CLOSE])
        runner.run_once()

        # Пробой канала выхода вниз (close 95 < min(low[6]) = 99).
        fake.ticker_price = 100.0  # выше стопа: защита не срабатывает
        append_candle(fake, [*FLAT, BREAKOUT_CLOSE, 95.0])
        runner.run_once()

        state = LiveState.load(cfg.state_path)
        assert state.position is None
        assert state.active_stop is None and state.active_tp is None
        assert [t["side"] for t in state.trades] == ["buy", "sell"]
        assert state.trades[-1]["reason"] == "donchian breakdown"
        assert state.last_candle_ts == _last_candle_iso(len(FLAT) + 2)


class TestStopBreach:
    def _open_position(self, tmp_path):
        runner, fake, cfg = make_runner(tmp_path, closes=FLAT, ticker_price=BREAKOUT_CLOSE)
        runner.run_once()
        append_candle(fake, [*FLAT, BREAKOUT_CLOSE])
        runner.run_once()
        return runner, fake, cfg

    def test_stop_breach_by_ticker_sells_immediately_before_candles(self, tmp_path) -> None:
        runner, fake, cfg = self._open_position(tmp_path)
        state = LiveState.load(cfg.state_path)
        stop = state.active_stop

        # Тикер гэпнул под стоп: выход по фактической цене (хуже стопа).
        fake.ticker_price = stop - 5.0
        runner.run_once()

        state = LiveState.load(cfg.state_path)
        assert state.position is None
        assert state.active_stop is None and state.active_tp is None
        sell = state.trades[-1]
        assert sell["side"] == "sell"
        assert sell["reason"] == "stop loss"
        assert sell["price"] == pytest.approx((stop - 5.0) * (1 - SLIPPAGE_BPS / 10_000))

    def test_take_profit_breach_by_ticker_sells(self, tmp_path) -> None:
        runner, fake, cfg = self._open_position(tmp_path)
        runner.state.set_stops(runner.state.active_stop, active_tp=150.0)
        runner.state.save(cfg.state_path)

        fake.ticker_price = 151.0
        runner.run_once()

        state = LiveState.load(cfg.state_path)
        assert state.position is None
        assert state.trades[-1]["reason"] == "take profit"

    def test_equity_tracks_pnl_and_fees(self, tmp_path) -> None:
        runner, fake, cfg = self._open_position(tmp_path)
        state = LiveState.load(cfg.state_path)
        stop = state.active_stop
        quantity = state.position.quantity

        fake.ticker_price = stop
        runner.run_once()

        in_price = BREAKOUT_CLOSE * (1 + SLIPPAGE_BPS / 10_000)
        out_price = stop * (1 - SLIPPAGE_BPS / 10_000)
        expected = START_CASH - quantity * in_price * (1 + FEE_RATE) + quantity * out_price * (
            1 - FEE_RATE
        )
        assert LiveState.load(cfg.state_path).equity == pytest.approx(expected)


class TestRestart:
    def test_restart_does_not_duplicate_orders_and_exits_work(self, tmp_path) -> None:
        runner1, fake, cfg = make_runner(tmp_path, closes=FLAT, ticker_price=BREAKOUT_CLOSE)
        runner1.run_once()
        append_candle(fake, [*FLAT, BREAKOUT_CLOSE])
        fake.ticker_price = BREAKOUT_CLOSE
        runner1.run_once()  # вход

        # «Рестарт»: новый раннер на том же state-файле и хранилище.
        runner2 = build_runner(cfg, fake)
        assert runner2.state.position is not None

        fake.ticker_price = 119.9  # выше стопа: цикл без новых ордеров
        runner2.run_once()
        state = LiveState.load(cfg.state_path)
        assert [t["side"] for t in state.trades] == ["buy"]  # дублей входа нет
        assert state.position is not None

        # Стратегия в новом раннере знает про позицию: пробой выхода -> sell.
        fake.ticker_price = 94.5
        append_candle(fake, [*FLAT, BREAKOUT_CLOSE, 95.0])
        runner2.run_once()

        state = LiveState.load(cfg.state_path)
        assert [t["side"] for t in state.trades] == ["buy", "sell"]
        assert state.position is None

    def test_restart_reprocesses_unsaved_candles_from_storage(self, tmp_path) -> None:
        # Сбой между записью свечи в storage и persist last_candle_ts:
        # storage обгоняет state -> свеча должна обработаться при рестарте.
        runner1, fake, cfg = make_runner(tmp_path, closes=FLAT, ticker_price=BREAKOUT_CLOSE)
        runner1.run_once()
        # Имитируем «упавшую» запись: свеча в storage, но last_candle_ts отстаёт.
        append_candle(fake, [*FLAT, BREAKOUT_CLOSE])
        storage = CandleStorage(cfg.data_root)
        tail = rows_to_df(donchian_rows([*FLAT, BREAKOUT_CLOSE])[-1:])
        merged = storage.append("bybit", cfg.symbol, cfg.timeframe, tail)

        assert len(merged) == len(FLAT) + 1

        runner2 = build_runner(cfg, fake)
        runner2.state.last_candle_ts = _last_candle_iso(len(FLAT))
        fake.ticker_price = BREAKOUT_CLOSE
        runner2.run_once()

        state = LiveState.load(cfg.state_path)
        assert state.position is not None
        assert state.last_candle_ts == _last_candle_iso(len(FLAT) + 1)


class TestKillSwitch:
    def test_kill_switch_blocks_orders_but_processes_candles(self, tmp_path, caplog) -> None:
        runner, fake, cfg = make_runner(tmp_path, closes=FLAT, ticker_price=BREAKOUT_CLOSE)
        runner.run_once()
        kill_path = Path(cfg.kill_switch_path)
        kill_path.parent.mkdir(parents=True, exist_ok=True)
        kill_path.write_text("stop", encoding="utf-8")

        append_candle(fake, [*FLAT, BREAKOUT_CLOSE])
        fake.ticker_price = BREAKOUT_CLOSE
        with caplog.at_level(logging.WARNING, logger="trading_bot.live.runner"):
            runner.run_once()

        state = LiveState.load(cfg.state_path)
        assert state.position is None  # ордер не размещён
        assert state.trades == []
        # Свечи при этом обрабатываются: last_candle дошёл до пробойной свечи.
        assert state.last_candle_ts == _last_candle_iso(len(FLAT) + 1)
        assert "kill switch" in caplog.text.lower()

    def test_kill_switch_blocks_even_protective_exit(self, tmp_path, caplog) -> None:
        runner, fake, cfg = make_runner(tmp_path, closes=FLAT, ticker_price=BREAKOUT_CLOSE)
        runner.run_once()
        append_candle(fake, [*FLAT, BREAKOUT_CLOSE])
        fake.ticker_price = BREAKOUT_CLOSE
        runner.run_once()
        assert LiveState.load(cfg.state_path).position is not None

        Path(cfg.kill_switch_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.kill_switch_path).write_text("stop", encoding="utf-8")
        fake.ticker_price = 10.0  # глубоко под стопом
        with caplog.at_level(logging.WARNING, logger="trading_bot.live.runner"):
            runner.run_once()

        state = LiveState.load(cfg.state_path)
        assert state.position is not None  # защитный выход тоже заблокирован
        assert [t["side"] for t in state.trades] == ["buy"]
        assert "kill switch" in caplog.text.lower()

    def test_removing_kill_switch_resumes_trading(self, tmp_path) -> None:
        runner, fake, cfg = make_runner(tmp_path, closes=FLAT, ticker_price=BREAKOUT_CLOSE)
        runner.run_once()
        kill_path = Path(cfg.kill_switch_path)
        kill_path.parent.mkdir(parents=True, exist_ok=True)
        kill_path.write_text("stop", encoding="utf-8")
        append_candle(fake, [*FLAT, BREAKOUT_CLOSE])
        fake.ticker_price = BREAKOUT_CLOSE
        runner.run_once()
        assert LiveState.load(cfg.state_path).position is None

        # Файл удалён -> торговля возобновляется на следующей закрытой свече.
        kill_path.unlink()
        append_candle(fake, [*FLAT, BREAKOUT_CLOSE, 125.0])
        fake.ticker_price = 125.0
        runner.run_once()

        state = LiveState.load(cfg.state_path)
        assert state.position is not None
        assert [t["side"] for t in state.trades] == ["buy"]


class TestNeedsAttention:
    def test_exchange_order_failure_sets_flag_and_blocks_trading(self, tmp_path, caplog) -> None:
        cfg = make_config(tmp_path)
        fake = FailingOrderCcxt(
            donchian_rows(FLAT), ticker_price=BREAKOUT_CLOSE, free_usdt=START_CASH
        )
        adapter = _TestnetAdapter(build_client(fake), cfg.symbol)
        runner = build_runner(cfg, fake, adapter=adapter)
        runner.run_once()

        append_candle(fake, [*FLAT, BREAKOUT_CLOSE])
        fake.ticker_price = BREAKOUT_CLOSE
        with caplog.at_level(logging.ERROR, logger="trading_bot.live.runner"):
            runner.run_once()

        state = LiveState.load(cfg.state_path)
        assert state.needs_attention is True  # позиция неясна -> флаг
        assert state.position is None
        assert state.trades == []
        assert "needs_attention" in caplog.text

        # Ордер отклонён один раз; теперь биржа «починилась», но торговля
        # стоит до ручного сброса флага, а heartbeat кричит.
        fake.reject = False
        append_candle(fake, [*FLAT, BREAKOUT_CLOSE, 125.0])
        fake.ticker_price = 125.0
        caplog.clear()
        with caplog.at_level(logging.ERROR, logger="trading_bot.live.runner"):
            runner.run_once()

        state = LiveState.load(cfg.state_path)
        assert fake.created_orders == []  # ордера по-прежнему запрещены
        assert state.position is None
        assert "NEEDS ATTENTION" in caplog.text


class TestPaperInvariant:
    def test_paper_cycle_never_calls_private_api(self, tmp_path) -> None:
        runner, fake, cfg = make_runner(tmp_path, closes=FLAT, ticker_price=BREAKOUT_CLOSE)
        runner.run_once()
        append_candle(fake, [*FLAT, BREAKOUT_CLOSE])
        runner.run_once()

        # Полный цикл с входом прошёл без единого приватного вызова биржи.
        assert fake.private_calls == []
        assert LiveState.load(cfg.state_path).position is not None


class TestEngineParity:
    def test_live_stop_exit_matches_backtest_engine(self, tmp_path) -> None:
        closes = [*FLAT, 120.0, 119.0, 70.0]

        engine = BacktestEngine(
            strategy=create_strategy("donchian", DONCHIAN_TEST_PARAMS),
            risk=RiskManager(position_size_pct=0.95, quantity_precision=6, min_notional=5.0),
            broker=SimulatedBroker(fee_rate=FEE_RATE, slippage_bps=SLIPPAGE_BPS),
            start_cash=START_CASH,
        )
        result = engine.run(rows_to_df(donchian_rows(closes)))
        engine_stops = result.trades[result.trades["reason_exit"] == "stop loss"]
        assert len(engine_stops) == 1
        engine_exit_price = float(engine_stops["exit_price"].iloc[0])
        engine_entry_price = float(result.trades["entry_price"].iloc[0])

        # Live: тикер входа = open свечи после сигнальной (120.0, как в движке),
        # тикер выхода = активный стоп (движок исполняет интрабарный стоп по уровню).
        runner, fake, cfg = make_runner(tmp_path, closes=FLAT, ticker_price=120.0)
        runner.run_once()
        append_candle(fake, [*FLAT, 120.0])
        fake.ticker_price = 120.0
        runner.run_once()
        append_candle(fake, [*FLAT, 120.0, 119.0])
        fake.ticker_price = 119.0
        runner.run_once()

        live_stop = LiveState.load(cfg.state_path).active_stop
        fake.ticker_price = live_stop
        runner.run_once()

        state = LiveState.load(cfg.state_path)
        assert state.position is None
        live_exit_price = state.trades[-1]["price"]
        live_entry_price = state.trades[0]["price"]

        # Переякоривание общими функциями (engine.stops) даёт тот же уровень:
        assert live_stop == pytest.approx(engine_entry_price - STOP_DISTANCE)
        assert live_entry_price == pytest.approx(engine_entry_price)
        assert live_exit_price == pytest.approx(engine_exit_price)


def build_client(fake):
    """Обернуть подмену ccxt в ExchangeClient (без реальной сети)."""
    client = ExchangeClient()
    client.exchange = fake
    return client
