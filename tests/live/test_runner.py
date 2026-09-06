"""Тесты цикла live-раннера (офлайн, биржа подменена)."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import ccxt
import pandas as pd
import pytest

from tests.conftest import BASE_MS, rows_to_df
from tests.live.fakes import (
    DONCHIAN_TEST_PARAMS,
    FOUR_HOUR_MS,
    FailingOrderCcxt,
    FakeCcxt,
    PrivateCcxt,
    append_candle,
    build_client,
    build_runner,
    donchian_rows,
    make_config,
    make_runner,
)
from trading_bot.data.storage import CandleStorage
from trading_bot.engine.backtest import BacktestEngine
from trading_bot.engine.broker import SimulatedBroker
from trading_bot.live.execution import TestnetAdapter as _TestnetAdapter
from trading_bot.live.state import LiveState, PositionState
from trading_bot.risk import RiskManager
from trading_bot.strategy import create_strategy
from trading_bot.strategy.base import Signal, SignalKind, Strategy

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

    def test_kill_switch_with_open_position_logs_distance_to_stop(
        self, tmp_path, caplog
    ) -> None:
        runner, fake, cfg = make_runner(tmp_path, closes=FLAT, ticker_price=BREAKOUT_CLOSE)
        runner.run_once()
        append_candle(fake, [*FLAT, BREAKOUT_CLOSE])
        fake.ticker_price = BREAKOUT_CLOSE
        runner.run_once()
        state = LiveState.load(cfg.state_path)
        stop = state.active_stop

        Path(cfg.kill_switch_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.kill_switch_path).write_text("stop", encoding="utf-8")
        ticker = stop + 10.0  # выше стопа, но без защиты STOP-файла
        fake.ticker_price = ticker
        with caplog.at_level(logging.ERROR, logger="trading_bot.live.runner"):
            runner.run_once()

        # Каждый цикл: error с ценой и дистанцией до стопа — позиция без присмотра.
        assert "NOT protected" in caplog.text
        assert f"{ticker:.4f}" in caplog.text
        assert f"{stop:.4f}" in caplog.text
        assert LiveState.load(cfg.state_path).position is not None

    def test_heartbeat_reports_both_switch_flags(self, tmp_path, caplog) -> None:
        runner, fake, cfg = make_runner(tmp_path, closes=FLAT, ticker_price=100.0)
        kill_path = Path(cfg.kill_switch_path)
        kill_path.parent.mkdir(parents=True, exist_ok=True)
        kill_path.write_text("stop", encoding="utf-8")
        with caplog.at_level(logging.INFO, logger="trading_bot.live.runner"):
            runner.run_once()

        assert "kill_switch=active" in caplog.text
        assert "pause=off" in caplog.text

        # Переключаемся на PAUSE: флаги в heartbeat меняются местами.
        kill_path.unlink()
        Path(cfg.pause_switch_path).write_text("pause", encoding="utf-8")
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="trading_bot.live.runner"):
            runner.run_once()

        assert "kill_switch=off" in caplog.text
        assert "pause=active" in caplog.text


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


class TestDuplicateOrderProtection:
    """B1: сбой после отправки ордера не приводит к повторному ордеру.

    NetworkError на create или падение fetch_order после успешного создания
    оставляют факт исполнения неясным: раннер обязан поднять needs_attention
    и НЕ переобрабатывать свечу — иначе следующий цикл поставил бы второй
    ордер по тому же сигналу.
    """

    def _testnet_runner(self, tmp_path, fake):
        cfg = make_config(tmp_path)
        adapter = _TestnetAdapter(build_client(fake), cfg.symbol)
        return build_runner(cfg, fake, adapter=adapter), cfg

    def test_fetch_order_failure_marks_attention_without_second_order(
        self, tmp_path, caplog
    ) -> None:
        fake = PrivateCcxt(
            donchian_rows(FLAT),
            ticker_price=BREAKOUT_CLOSE,
            free_usdt=START_CASH,
            fetch_order_exc=ccxt.NetworkError("connection reset"),
        )
        runner, cfg = self._testnet_runner(tmp_path, fake)
        runner.run_once()  # чистый state: только прогрев

        append_candle(fake, [*FLAT, BREAKOUT_CLOSE])
        fake.ticker_price = BREAKOUT_CLOSE
        with caplog.at_level(logging.ERROR, logger="trading_bot.live.runner"):
            runner.run_once()

        # Ордер ушёл на биржу один раз; статус получить не удалось.
        assert len(fake.created_orders) == 1
        state = LiveState.load(cfg.state_path)
        assert state.needs_attention is True
        assert state.attention_reason is not None
        assert state.position is None
        # Свеча помечена обработанной, несмотря на сбой: дубля не будет.
        assert state.last_candle_ts == _last_candle_iso(len(FLAT) + 1)
        assert "will NOT be retried" in caplog.text

        # Сеть «починилась», но торговля стоит до ручного сброса флага.
        fake.fetch_order_exc = None
        append_candle(fake, [*FLAT, BREAKOUT_CLOSE, 125.0])
        fake.ticker_price = 125.0
        runner.run_once()

        assert len(fake.created_orders) == 1
        assert LiveState.load(cfg.state_path).position is None

    def test_unparseable_status_marks_attention_without_second_order(
        self, tmp_path
    ) -> None:
        # Статус без average/price: ордер создан, результат не распарсился.
        fake = PrivateCcxt(
            donchian_rows(FLAT),
            ticker_price=BREAKOUT_CLOSE,
            free_usdt=START_CASH,
            order_status={"status": "closed"},
        )
        runner, cfg = self._testnet_runner(tmp_path, fake)
        runner.run_once()

        append_candle(fake, [*FLAT, BREAKOUT_CLOSE])
        fake.ticker_price = BREAKOUT_CLOSE
        runner.run_once()

        assert len(fake.created_orders) == 1
        state = LiveState.load(cfg.state_path)
        assert state.needs_attention is True
        assert state.position is None
        assert state.last_candle_ts == _last_candle_iso(len(FLAT) + 1)


class TestCatchUp:
    """M2: догон после простоя исполняет сигнал только последней свечи батча."""

    def _warm_runner(self, tmp_path):
        runner, fake, cfg = make_runner(tmp_path, closes=FLAT, ticker_price=130.0)
        runner.run_once()
        return runner, fake, cfg

    def test_batch_of_three_trades_only_the_latest_signal(self, tmp_path, caplog) -> None:
        runner, fake, cfg = self._warm_runner(tmp_path)
        # За время простоя накопились: пробой (вход), пробой канала вниз
        # (выход), снова пробой вверх. Промежуточные сигналы устарели.
        append_candle(fake, [*FLAT, 125.0])
        append_candle(fake, [*FLAT, 125.0, 96.0])
        append_candle(fake, [*FLAT, 125.0, 96.0, 130.0])
        fake.ticker_price = 130.0
        with caplog.at_level(logging.WARNING, logger="trading_bot.live.runner"):
            runner.run_once()

        state = LiveState.load(cfg.state_path)
        # Без догон-логики были бы сделки buy-sell-buy по устаревшим сигналам;
        # теперь исполнен только сигнал последней свечи — один buy.
        assert [t["side"] for t in state.trades] == ["buy"]
        assert state.position is not None
        assert state.last_candle_ts == _last_candle_iso(len(FLAT) + 3)
        assert "catch-up after downtime" in caplog.text
        assert "2 stale candle(s)" in caplog.text

    def test_single_candle_batch_trades_as_before(self, tmp_path) -> None:
        runner, fake, cfg = self._warm_runner(tmp_path)
        append_candle(fake, [*FLAT, 125.0])
        fake.ticker_price = 125.0
        runner.run_once()

        state = LiveState.load(cfg.state_path)
        assert [t["side"] for t in state.trades] == ["buy"]
        assert state.position is not None
        assert state.last_candle_ts == _last_candle_iso(len(FLAT) + 1)


class TestEntryPersistence:
    """M4: позиция сохраняется сразу после исполнения, а не в конце цикла."""

    def test_crash_after_entry_fill_keeps_position_on_disk(self, tmp_path) -> None:
        runner, fake, cfg = make_runner(tmp_path, closes=FLAT, ticker_price=BREAKOUT_CLOSE)
        runner.run_once()
        append_candle(fake, [*FLAT, BREAKOUT_CLOSE])
        fake.ticker_price = BREAKOUT_CLOSE

        # Крэш сразу после fill: стратегия падает в on_fill, который вызывается
        # ПОСЛЕ сохранения state — позиция обязана уже быть на диске.
        def crash(_fill) -> None:
            raise RuntimeError("process killed right after entry fill")

        runner.strategy.on_fill = crash
        with pytest.raises(RuntimeError, match="process killed"):
            runner._run_cycle()

        state = LiveState.load(cfg.state_path)
        assert state.position is not None
        assert [t["side"] for t in state.trades] == ["buy"]
        assert state.active_stop is not None


class TestEntrySizing:
    """m6: сайзинг входа считается по цене тикера, а не close сигнальной свечи."""

    def test_entry_quantity_uses_ticker_price(self, tmp_path) -> None:
        # Тикер (100) сильно ниже close сигнальной свечи (120).
        runner, fake, cfg = make_runner(tmp_path, closes=FLAT, ticker_price=100.0)
        runner.run_once()
        append_candle(fake, [*FLAT, BREAKOUT_CLOSE])
        runner.run_once()

        expected_qty = math.floor(START_CASH * 0.95 / 100.0 * 10**6) / 10**6
        state = LiveState.load(cfg.state_path)
        assert state.position.quantity == pytest.approx(expected_qty)


class TestPauseSwitch:
    """PAUSE запрещает только входы; сигнальные выходы и стопы работают."""

    def _open_position(self, tmp_path):
        runner, fake, cfg = make_runner(tmp_path, closes=FLAT, ticker_price=BREAKOUT_CLOSE)
        runner.run_once()
        append_candle(fake, [*FLAT, BREAKOUT_CLOSE])
        fake.ticker_price = BREAKOUT_CLOSE
        runner.run_once()
        assert LiveState.load(cfg.state_path).position is not None
        return runner, fake, cfg

    def _create_pause(self, cfg) -> None:
        Path(cfg.pause_switch_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.pause_switch_path).write_text("pause", encoding="utf-8")

    def test_pause_blocks_entry_but_processes_candles(self, tmp_path, caplog) -> None:
        runner, fake, cfg = make_runner(tmp_path, closes=FLAT, ticker_price=BREAKOUT_CLOSE)
        runner.run_once()
        self._create_pause(cfg)

        append_candle(fake, [*FLAT, BREAKOUT_CLOSE])
        fake.ticker_price = BREAKOUT_CLOSE
        with caplog.at_level(logging.WARNING, logger="trading_bot.live.runner"):
            runner.run_once()

        state = LiveState.load(cfg.state_path)
        assert state.position is None
        assert state.trades == []
        # Свечи при PAUSE обрабатываются: заблокирован только вход.
        assert state.last_candle_ts == _last_candle_iso(len(FLAT) + 1)
        assert "pause switch is active" in caplog.text

    def test_pause_allows_signal_exit(self, tmp_path) -> None:
        runner, fake, cfg = self._open_position(tmp_path)
        self._create_pause(cfg)

        fake.ticker_price = 100.0  # выше стопа: выход именно по сигналу
        append_candle(fake, [*FLAT, BREAKOUT_CLOSE, 95.0])
        runner.run_once()

        state = LiveState.load(cfg.state_path)
        assert state.position is None
        assert [t["side"] for t in state.trades] == ["buy", "sell"]
        assert state.trades[-1]["reason"] == "donchian breakdown"

    def test_pause_allows_protective_stop(self, tmp_path) -> None:
        runner, fake, cfg = self._open_position(tmp_path)
        stop = LiveState.load(cfg.state_path).active_stop
        self._create_pause(cfg)

        fake.ticker_price = stop - 5.0
        runner.run_once()

        state = LiveState.load(cfg.state_path)
        assert state.position is None
        assert state.trades[-1]["reason"] == "stop loss"


class TestReconcile:
    """reconcile: по количеству доверяем бирже; чужие монеты — needs_attention."""

    ENTRY_TS = pd.Timestamp(BASE_MS, unit="ms", tz="UTC")

    def _runner_with_position(self, tmp_path, free_base: float):
        cfg = make_config(tmp_path)
        fake = PrivateCcxt(donchian_rows(FLAT), free_base=free_base)
        adapter = _TestnetAdapter(build_client(fake), cfg.symbol)
        runner = build_runner(cfg, fake, adapter=adapter)
        runner.state.position = PositionState(
            quantity=1.0,
            entry_price=100.0,
            entry_ts=self.ENTRY_TS.isoformat(),
            entry_reason="test entry",
        )
        runner.state.save(cfg.state_path)
        return runner, cfg

    def test_matching_position_is_left_alone(self, tmp_path) -> None:
        runner, cfg = self._runner_with_position(tmp_path, free_base=1.0)

        runner.reconcile()

        state = LiveState.load(cfg.state_path)
        assert state.position.quantity == pytest.approx(1.0)
        assert state.needs_attention is False

    def test_mismatch_trusts_the_exchange_and_saves(self, tmp_path) -> None:
        runner, cfg = self._runner_with_position(tmp_path, free_base=0.5)

        runner.reconcile()

        state = LiveState.load(cfg.state_path)
        assert state.position.quantity == pytest.approx(0.5)
        assert state.needs_attention is False

    def test_exchange_coins_without_state_position_raise_attention(self, tmp_path) -> None:
        cfg = make_config(tmp_path)
        fake = PrivateCcxt(donchian_rows(FLAT), free_base=2.0)
        adapter = _TestnetAdapter(build_client(fake), cfg.symbol)
        runner = build_runner(cfg, fake, adapter=adapter)

        runner.reconcile()

        state = LiveState.load(cfg.state_path)
        assert state.needs_attention is True
        assert "without a state position" in state.attention_reason


class TakeProfitStrategy(Strategy):
    """Тестовая стратегия для TP-parity: один вход с тейком close + дистанция."""

    name = "tp_only"

    def __init__(self, tp_distance: float = 10.0) -> None:
        self.tp_distance = tp_distance

    @property
    def warmup_period(self) -> int:
        return len(FLAT)

    def on_candle(self, candles) -> list[Signal]:
        if len(candles) == len(FLAT) + 1:
            close = float(candles["close"].iloc[-1])
            return [
                Signal(
                    SignalKind.LONG_ENTRY,
                    reason="tp entry",
                    take_profit=close + self.tp_distance,
                )
            ]
        return []


class TestTakeProfitParity:
    """TP-parity: live-тейк совпадает с бэктест-движком на том же сценарии."""

    def test_live_take_profit_exit_matches_backtest_engine(self, tmp_path) -> None:
        closes = [*FLAT, 120.0, 119.0, 131.0]

        engine = BacktestEngine(
            strategy=TakeProfitStrategy(tp_distance=10.0),
            risk=RiskManager(position_size_pct=0.95, quantity_precision=6, min_notional=5.0),
            broker=SimulatedBroker(fee_rate=FEE_RATE, slippage_bps=SLIPPAGE_BPS),
            start_cash=START_CASH,
        )
        result = engine.run(rows_to_df(donchian_rows(closes)))
        engine_tps = result.trades[result.trades["reason_exit"] == "take profit"]
        assert len(engine_tps) == 1
        engine_exit_price = float(engine_tps["exit_price"].iloc[0])
        engine_entry_price = float(result.trades["entry_price"].iloc[0])

        # Live: тикер входа = open свечи после сигнальной (120.0, как в движке),
        # тикер выхода = активный тейк-профит (движок исполняет по уровню).
        cfg = make_config(tmp_path)
        fake = FakeCcxt(donchian_rows(FLAT), ticker_price=120.0)
        runner = build_runner(cfg, fake, strategy=TakeProfitStrategy(tp_distance=10.0))
        runner.run_once()
        append_candle(fake, [*FLAT, 120.0])
        fake.ticker_price = 120.0
        runner.run_once()

        live_tp = LiveState.load(cfg.state_path).active_tp

        append_candle(fake, [*FLAT, 120.0, 119.0])
        fake.ticker_price = 119.0
        runner.run_once()
        append_candle(fake, [*FLAT, 120.0, 119.0, 131.0])
        fake.ticker_price = live_tp  # пробой тейка тикером
        runner.run_once()

        state = LiveState.load(cfg.state_path)
        assert state.position is None
        assert state.trades[-1]["reason"] == "take profit"
        live_exit_price = state.trades[-1]["price"]

        # Тейк переякорен на фактическую цену входа той же функцией, что в
        # движке, и исполняется по тому же уровню.
        assert live_tp == pytest.approx(engine_entry_price + 10.0)
        assert live_exit_price == pytest.approx(engine_exit_price)
