"""Тесты Telegram-уведомлений в цикле live-раннера (офлайн, биржа подменена).

Инвариант раздела: сбой отправки уведомления никогда не рвёт торговый цикл —
тест с реальным ``Notifier`` и падающим urlopen доказывает, что сделка
исполняется, как если бы уведомлений не было.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import ccxt

from tests.live.fakes import (
    FailingOrderCcxt,
    FakeCcxt,
    RecordingNotifier,
    append_candle,
    build_client,
    build_runner,
    donchian_rows,
    make_config,
)
from trading_bot.live.execution import TestnetAdapter as _TestnetAdapter
from trading_bot.live.notify import Notifier, NullNotifier
from trading_bot.live.runner import LiveRunner
from trading_bot.live.state import load_or_fresh_state

START_CASH = 10_000.0
SLIPPAGE_BPS = 5.0

FLAT = [100.0] * 6
BREAKOUT_CLOSE = 120.0
# См. аналитику сценария в tests/live/test_runner.py: стоп на 44 ниже close 120.
STOP_DISTANCE = 44.0

TELEGRAM_TOKEN = "555:runner-test-token"


def _ok_response():
    """Подмена HTTP-200 ответа Telegram (context manager, как urlopen)."""

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    return _Response()


def _notify_runner(tmp_path, notifier, closes=FLAT, ticker_price=BREAKOUT_CLOSE, **overrides):
    """Собрать paper-раннер с подменой уведомлений; вернуть (runner, fake, cfg)."""
    cfg = make_config(tmp_path, **overrides)
    fake = FakeCcxt(donchian_rows(closes), ticker_price=ticker_price)
    return build_runner(cfg, fake, notifier=notifier), fake, cfg


def _entry_cycle(tmp_path, notifier, **overrides):
    """Прогнать прогрев + свечу-пробой: после этого позиция открыта."""
    runner, fake, cfg = _notify_runner(tmp_path, notifier, **overrides)
    runner.run_once()  # чистый state: только прогрев
    append_candle(fake, [*FLAT, BREAKOUT_CLOSE])
    fake.ticker_price = BREAKOUT_CLOSE
    runner.run_once()  # вход
    return runner, fake, cfg


class TestStartNotification:
    def test_start_sent_once_per_process(self, tmp_path) -> None:
        notifier = RecordingNotifier()
        runner, _, _ = _notify_runner(tmp_path, notifier, ticker_price=100.0)

        runner.run_once()
        runner.run_once()

        starts = notifier.texts("critical")
        assert len(starts) == 1
        assert "Раннер запущен" in starts[0]
        assert "BTC/USDT 4h" in starts[0]
        assert "donchian" in starts[0]
        assert "режим paper" in starts[0]
        assert "позиция: flat" in starts[0]

    def test_start_reports_open_position_after_restart(self, tmp_path) -> None:
        notifier = RecordingNotifier()
        _, fake, cfg = _entry_cycle(tmp_path, RecordingNotifier())

        # «Рестарт»: новый раннер на том же state — старт с открытой позицией.
        restarted = build_runner(cfg, fake, notifier=notifier)
        restarted.run_once()

        starts = notifier.texts("critical")
        assert len(starts) == 1
        assert "LONG 79.166666 @ 120.0600" in starts[0]


class TestTradeNotifications:
    def test_entry_message_contains_qty_price_and_stop(self, tmp_path) -> None:
        notifier = RecordingNotifier()
        _entry_cycle(tmp_path, notifier)

        trades = notifier.texts("trade")
        assert len(trades) == 1
        fill_price = BREAKOUT_CLOSE * (1 + SLIPPAGE_BPS / 10_000)
        assert "📈 LONG BTC/USDT" in trades[0]
        assert f"79.166666 @ {fill_price:.4f}" in trades[0]
        assert f"стоп @ {fill_price - STOP_DISTANCE:.4f}" in trades[0]
        assert "(donchian breakout up)" in trades[0]

    def test_signal_exit_message_contains_pnl(self, tmp_path) -> None:
        notifier = RecordingNotifier()
        runner, fake, _ = _entry_cycle(tmp_path, notifier)

        fake.ticker_price = 100.0  # выше стопа: это сигнальный выход
        append_candle(fake, [*FLAT, BREAKOUT_CLOSE, 95.0])
        runner.run_once()

        trades = notifier.texts("trade")
        assert len(trades) == 2
        exit_price = 100.0 * (1 - SLIPPAGE_BPS / 10_000)
        entry_price = BREAKOUT_CLOSE * (1 + SLIPPAGE_BPS / 10_000)
        pnl = 79.166666 * (exit_price - entry_price)
        assert "📉 Выход" in trades[1]
        assert f"79.166666 @ {exit_price:.4f}" in trades[1]
        assert f"PnL ~{pnl:+.2f} USDT" in trades[1]
        assert "(donchian breakdown)" in trades[1]

    def test_stop_loss_breach_message(self, tmp_path) -> None:
        notifier = RecordingNotifier()
        runner, fake, _ = _entry_cycle(tmp_path, notifier)
        stop = runner.state.active_stop

        fake.ticker_price = stop - 5.0
        runner.run_once()

        trades = notifier.texts("trade")
        assert len(trades) == 2
        assert "🛑 Стоп-лосс" in trades[1]
        assert "(stop loss)" in trades[1]

    def test_take_profit_breach_message(self, tmp_path) -> None:
        notifier = RecordingNotifier()
        runner, fake, _ = _entry_cycle(tmp_path, notifier)
        runner.state.set_stops(runner.state.active_stop, active_tp=150.0)

        fake.ticker_price = 151.0
        runner.run_once()

        trades = notifier.texts("trade")
        assert "🎯 Тейк-профит" in trades[1]
        assert "(take profit)" in trades[1]


class TestNeedsAttentionNotification:
    def test_unclear_order_sends_one_critical_message(self, tmp_path) -> None:
        notifier = RecordingNotifier()
        cfg = make_config(tmp_path)
        fake = FailingOrderCcxt(
            donchian_rows(FLAT), ticker_price=BREAKOUT_CLOSE, free_usdt=START_CASH
        )
        adapter = _TestnetAdapter(build_client(fake), cfg.symbol)
        runner = build_runner(cfg, fake, adapter=adapter, notifier=notifier)
        runner.run_once()  # прогрев (и стартовое сообщение)

        append_candle(fake, [*FLAT, BREAKOUT_CLOSE])
        fake.ticker_price = BREAKOUT_CLOSE
        runner.run_once()  # ордер отклонён биржей -> needs_attention

        attention = notifier.texts("critical")[1:]  # без стартового сообщения
        assert len(attention) == 1
        assert "⚠️ Требуется внимание" in attention[0]
        assert "failed on the exchange" in attention[0]
        assert "Торговля остановлена до ручного вмешательства" in attention[0]

        # Флаг уже поднят: повторных сообщений на каждом цикле нет.
        fake.reject = False
        append_candle(fake, [*FLAT, BREAKOUT_CLOSE, 125.0])
        fake.ticker_price = 125.0
        runner.run_once()

        assert len(notifier.texts("critical")) == 2  # старт + один attention

    def test_flag_raised_before_start_is_not_reannounced(self, tmp_path) -> None:
        notifier = RecordingNotifier()
        cfg = make_config(tmp_path)
        # Рестарт с уже поднятым флагом: state на диске несёт needs_attention
        # ещё до создания раннера.
        persisted = load_or_fresh_state(cfg.state_path, cfg)
        persisted.mark_needs_attention("pre-existing problem")
        persisted.save(cfg.state_path)
        fake = FakeCcxt(donchian_rows(FLAT), ticker_price=100.0)
        runner = build_runner(cfg, fake, notifier=notifier)

        runner.run_once()

        assert "Требуется внимание" not in "".join(notifier.texts("critical"))


class TestSwitchNotifications:
    def test_stop_transition_notified_once_not_every_cycle(self, tmp_path) -> None:
        notifier = RecordingNotifier()
        runner, _, cfg = _notify_runner(tmp_path, notifier, ticker_price=100.0)
        runner.run_once()  # первый цикл: STOP off

        kill_path = Path(cfg.kill_switch_path)
        kill_path.parent.mkdir(parents=True, exist_ok=True)
        kill_path.write_text("stop", encoding="utf-8")
        runner.run_once()  # переход -> ровно одно сообщение
        runner.run_once()  # STOP всё ещё активен -> без повторов

        switches = notifier.texts("switch")
        assert len(switches) == 1
        assert "STOP активирован" in switches[0]

        kill_path.unlink()
        runner.run_once()  # переход обратно

        switches = notifier.texts("switch")
        assert len(switches) == 2
        assert "STOP снят" in switches[1]

    def test_pause_transition_notified(self, tmp_path) -> None:
        notifier = RecordingNotifier()
        runner, _, cfg = _notify_runner(tmp_path, notifier, ticker_price=100.0)
        runner.run_once()

        pause_path = Path(cfg.pause_switch_path)
        pause_path.parent.mkdir(parents=True, exist_ok=True)
        pause_path.write_text("pause", encoding="utf-8")
        runner.run_once()
        pause_path.unlink()
        runner.run_once()

        switches = notifier.texts("switch")
        assert [s for s in switches if "PAUSE активирован" in s]
        assert [s for s in switches if "PAUSE снят" in s]

    def test_restart_with_existing_switch_does_not_spam(self, tmp_path) -> None:
        notifier = RecordingNotifier()
        runner, fake, cfg = _notify_runner(tmp_path, notifier, ticker_price=100.0)
        runner.run_once()

        kill_path = Path(cfg.kill_switch_path)
        kill_path.parent.mkdir(parents=True, exist_ok=True)
        kill_path.write_text("stop", encoding="utf-8")
        runner.run_once()
        assert len(notifier.texts("switch")) == 1

        # «Рестарт» при существующем STOP: первый цикл — начальное состояние.
        restart_notifier = RecordingNotifier()
        restarted = build_runner(cfg, fake, notifier=restart_notifier)
        restarted.run_once()

        assert restart_notifier.texts("switch") == []


class TestNetworkErrorNotifications:
    def test_network_error_notified_and_throttled(self, tmp_path, mocker) -> None:
        urlopen = mocker.patch(
            "trading_bot.live.notify.urlopen", return_value=_ok_response()
        )
        mocker.patch.object(Notifier, "_now_ts", return_value=1000.0)
        notifier = Notifier(TELEGRAM_TOKEN, "42", error_throttle_minutes=60)

        class FlakyTicker(FakeCcxt):
            def __init__(self, rows, ticker_price=100.0) -> None:
                super().__init__(rows, ticker_price)
                self.ticker_exc: Exception | None = None

            def fetch_ticker(self, symbol) -> dict:
                if self.ticker_exc is not None:
                    raise self.ticker_exc
                return super().fetch_ticker(symbol)

        cfg = make_config(tmp_path)
        fake = FlakyTicker(donchian_rows(FLAT), ticker_price=100.0)
        runner = build_runner(cfg, fake, notifier=notifier)

        runner.run_once()  # прогрев + стартовое сообщение (critical)
        assert urlopen.call_count == 1

        fake.ticker_exc = ccxt.NetworkError("connection reset by peer")
        runner.run_once()  # сетевой сбой -> error-уведомление
        runner.run_once()  # сбой повторился в окне троттлинга -> тишина

        assert urlopen.call_count == 2
        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        assert payload["text"].startswith("⚠️ network error")
        assert "connection reset by peer" in payload["text"]

    def test_notifier_failure_never_breaks_the_cycle(self, tmp_path, mocker) -> None:
        # Реальный Notifier, чей сетевой слой падает: send возвращает False,
        # торговый цикл обязан отработать как обычно (вход исполняется).
        mocker.patch(
            "trading_bot.live.notify.urlopen",
            side_effect=RuntimeError("telegram is down"),
        )
        notifier = Notifier(TELEGRAM_TOKEN, "42")
        runner, fake, cfg = _entry_cycle(tmp_path, notifier)

        state = load_or_fresh_state(cfg.state_path, cfg)
        assert state.position is not None
        assert fake.private_calls == []  # paper-инвариант не задет
        assert runner.state.position is not None


class TestHeartbeatDigest:
    def test_digest_sent_after_heartbeat_hours(self, tmp_path, mocker) -> None:
        notifier = RecordingNotifier()
        clock = mocker.patch.object(LiveRunner, "_now_ts", return_value=1000.0)
        runner, _, _ = _notify_runner(tmp_path, notifier, ticker_price=100.0)

        runner.run_once()
        assert notifier.texts("info") == []  # первый дайджест не сразу после старта

        clock.return_value = 1000.0 + 25 * 3600.0
        runner.run_once()
        runner.run_once()  # сразу повторно — дайджест не дублируется

        digests = notifier.texts("info")
        assert len(digests) == 1
        assert digests[0].startswith("💤 Heartbeat")
        assert "позиция flat" in digests[0]
        assert "equity ~10000.00" in digests[0]
        assert "last candle age" in digests[0]
        assert "STOP off, PAUSE off" in digests[0]

    def test_digest_disabled_with_zero_hours(self, tmp_path, mocker) -> None:
        notifier = RecordingNotifier()
        clock = mocker.patch.object(LiveRunner, "_now_ts", return_value=1000.0)
        runner, _, _ = _notify_runner(
            tmp_path, notifier, ticker_price=100.0, heartbeat_hours=0.0
        )

        clock.return_value = 1000.0 + 10_000 * 3600.0
        runner.run_once()
        runner.run_once()

        assert notifier.texts("info") == []

    def test_digest_reports_open_position(self, tmp_path, mocker) -> None:
        notifier = RecordingNotifier()
        clock = mocker.patch.object(LiveRunner, "_now_ts", return_value=1000.0)
        runner, fake, _ = _entry_cycle(tmp_path, notifier)

        clock.return_value = 1000.0 + 30 * 3600.0
        fake.ticker_price = BREAKOUT_CLOSE
        runner.run_once()

        digest = notifier.texts("info")[0]
        assert "LONG 79.166666 @ 120.0600" in digest

    def test_first_digest_comes_only_after_interval_not_at_start(
        self, tmp_path, mocker
    ) -> None:
        # Частые рестарты не должны штормить дайджестами: таймер стартует
        # со временем старта процесса, первый дайджест — через interval.
        notifier = RecordingNotifier()
        clock = mocker.patch.object(LiveRunner, "_now_ts", return_value=1000.0)
        runner, _, _ = _notify_runner(tmp_path, notifier, ticker_price=100.0)

        for step_hours in (10 / 60, 1.0, 2.0):  # 10 мин, 1 ч, 2 ч — всё меньше 24 ч
            clock.return_value = 1000.0 + step_hours * 3600.0
            runner.run_once()

        assert notifier.texts("info") == []


def test_default_runner_uses_silent_stub(tmp_path) -> None:
    """Без notifier раннер собирается с заглушкой и не шлёт ничего."""
    cfg = make_config(tmp_path)
    fake = FakeCcxt(donchian_rows(FLAT), ticker_price=100.0)
    runner = build_runner(cfg, fake)

    assert isinstance(runner.notifier, NullNotifier)
    assert runner.notifier.enabled is False
    assert runner.notifier.send("ignored", category="critical") is False


def test_notifier_send_errors_are_swallowed_not_raised(mocker, caplog) -> None:
    """Договор раннера: send не бросает исключений даже при полном сбое сети."""
    mocker.patch("trading_bot.live.notify.urlopen", side_effect=OSError("network down"))
    notifier = Notifier(TELEGRAM_TOKEN, "42")

    with caplog.at_level(logging.WARNING, logger="trading_bot.live.notify"):
        assert notifier.send("текст") is False

    assert "network down" in caplog.text
