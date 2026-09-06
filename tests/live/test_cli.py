"""Тесты CLI-команды live (офлайн, биржа подменяется)."""

from __future__ import annotations

import fcntl
import json
import logging

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from tests.live.fakes import FakeCcxt, donchian_rows
from trading_bot.cli import _build_notifier, app
from trading_bot.config import LiveConfig
from trading_bot.data.exchange import ExchangeClient

runner = CliRunner()

LIVE_CONFIG = """\
symbol: BTC/USDT
timeframe: 4h
strategy: donchian
strategy_params:
  entry_period: 2
  exit_period: 1
  atr_period: 1
  atr_mult: 2.0
mode: paper
data_root: data/live
state_path: data/live/state.json
kill_switch_path: data/live/STOP
log_file: logs/live.log
"""

LIVE_CONFIG_TESTNET = LIVE_CONFIG.replace("mode: paper", "mode: testnet")


def _write_config(tmp_path, text: str):
    path = tmp_path / "live.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _patched_client(mocker, fake: FakeCcxt) -> None:
    client = ExchangeClient()
    client.exchange = fake
    mocker.patch("trading_bot.cli.ExchangeClient", return_value=client)


class TestLiveOnce:
    def test_once_smoke_writes_state_candles_and_log(self, tmp_path, monkeypatch, mocker) -> None:
        monkeypatch.chdir(tmp_path)
        config = _write_config(tmp_path, LIVE_CONFIG)
        fake = FakeCcxt(donchian_rows([100.0] * 6), ticker_price=100.0)
        _patched_client(mocker, fake)

        result = runner.invoke(app, ["live", "--config", str(config), "--once"])

        assert result.exit_code == 0, result.output
        assert "Live runner: mode=paper symbol=BTC/USDT timeframe=4h" in result.output
        assert "donchian" in result.output
        assert "Kill switch" in result.output
        assert "Position: flat" in result.output
        # state создан, свечи докачаны в data/live (отдельно от research), лог пишется.
        state_path = tmp_path / "data" / "live" / "state.json"
        candles_path = tmp_path / "data" / "live" / "bybit" / "BTC_USDT" / "4h.parquet"
        assert state_path.exists()
        assert candles_path.exists()
        assert len(pd.read_parquet(candles_path, engine="pyarrow")) == 6
        assert (tmp_path / "logs" / "live.log").exists()

    def test_second_once_does_not_duplicate_candles(self, tmp_path, monkeypatch, mocker) -> None:
        monkeypatch.chdir(tmp_path)
        config = _write_config(tmp_path, LIVE_CONFIG)
        fake = FakeCcxt(donchian_rows([100.0] * 6), ticker_price=100.0)
        _patched_client(mocker, fake)
        first = runner.invoke(app, ["live", "--config", str(config), "--once"])
        assert first.exit_code == 0, first.output

        _patched_client(mocker, fake)  # та же подмена: новых свечей нет
        second = runner.invoke(app, ["live", "--config", str(config), "--once"])

        assert second.exit_code == 0, second.output
        candles_path = tmp_path / "data" / "live" / "bybit" / "BTC_USDT" / "4h.parquet"
        assert len(pd.read_parquet(candles_path, engine="pyarrow")) == 6

    def test_new_candle_between_runs_is_processed_once(self, tmp_path, monkeypatch, mocker) -> None:
        monkeypatch.chdir(tmp_path)
        config = _write_config(tmp_path, LIVE_CONFIG)
        fake = FakeCcxt(donchian_rows([100.0] * 6), ticker_price=120.0)
        _patched_client(mocker, fake)
        first = runner.invoke(app, ["live", "--config", str(config), "--once"])
        assert first.exit_code == 0, first.output

        # Появилась закрытая свеча-пробой: paper-вход по тикеру 120.
        fake.rows.append(donchian_rows([100.0] * 6 + [120.0])[-1])
        _patched_client(mocker, fake)
        second = runner.invoke(app, ["live", "--config", str(config), "--once"])
        assert second.exit_code == 0, second.output

        state = json.loads(
            (tmp_path / "data" / "live" / "state.json").read_text(encoding="utf-8")
        )
        assert state["position"] is not None
        assert len(state["trades"]) == 1


class TestDryRun:
    def test_dry_run_forces_paper_without_keys(
        self, tmp_path, monkeypatch, mocker
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("BYBIT_API_KEY", raising=False)
        monkeypatch.delenv("BYBIT_API_SECRET", raising=False)
        config = _write_config(tmp_path, LIVE_CONFIG_TESTNET)
        fake = FakeCcxt(donchian_rows([100.0] * 6), ticker_price=100.0)
        _patched_client(mocker, fake)

        result = runner.invoke(app, ["live", "--config", str(config), "--once", "--dry-run"])

        assert result.exit_code == 0, result.output
        assert "forcing paper mode" in result.output
        assert "mode=paper" in result.output

    def test_testnet_without_keys_fails_with_hint(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("BYBIT_API_KEY", raising=False)
        monkeypatch.delenv("BYBIT_API_SECRET", raising=False)
        config = _write_config(tmp_path, LIVE_CONFIG_TESTNET)

        result = runner.invoke(app, ["live", "--config", str(config), "--once"])

        assert result.exit_code == 1
        assert "BYBIT_API_KEY" in result.output
        assert "testnet.bybit.com" in result.output


class TestStateGuard:
    def test_state_config_mismatch_fails_cleanly(self, tmp_path, monkeypatch, mocker) -> None:
        monkeypatch.chdir(tmp_path)
        config = _write_config(tmp_path, LIVE_CONFIG)
        fake = FakeCcxt(donchian_rows([100.0] * 6), ticker_price=100.0)
        _patched_client(mocker, fake)
        first = runner.invoke(app, ["live", "--config", str(config), "--once"])
        assert first.exit_code == 0, first.output

        # Тот же state-файл, но параметры стратегии в конфиге изменились.
        changed = _write_config(tmp_path, LIVE_CONFIG.replace("entry_period: 2", "entry_period: 3"))
        _patched_client(mocker, fake)
        result = runner.invoke(app, ["live", "--config", str(changed), "--once"])

        assert result.exit_code == 1
        assert "Ошибка состояния live" in result.output
        assert "move or delete the state file" in result.output

    def test_corrupted_state_fails_cleanly(self, tmp_path, monkeypatch, mocker) -> None:
        monkeypatch.chdir(tmp_path)
        config = _write_config(tmp_path, LIVE_CONFIG)
        state_path = tmp_path / "data" / "live" / "state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("{broken", encoding="utf-8")
        fake = FakeCcxt(donchian_rows([100.0] * 6), ticker_price=100.0)
        _patched_client(mocker, fake)

        result = runner.invoke(app, ["live", "--config", str(config), "--once"])

        assert result.exit_code == 1
        assert "corrupted" in result.output


class TestStateLock:
    def _invoke_once(self, tmp_path, mocker, fake, config):
        _patched_client(mocker, fake)
        return runner.invoke(app, ["live", "--config", str(config), "--once"])

    def test_second_run_refused_while_lock_is_held(
        self, tmp_path, monkeypatch, mocker
    ) -> None:
        monkeypatch.chdir(tmp_path)
        config = _write_config(tmp_path, LIVE_CONFIG)
        fake = FakeCcxt(donchian_rows([100.0] * 6), ticker_price=100.0)

        lock_path = tmp_path / "data" / "live" / "state.json.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self._invoke_once(tmp_path, mocker, fake, config)

        assert result.exit_code == 1
        assert "already running" in result.output
        # Раннер не стартовал: ни баннера, ни state-файла.
        assert "Position:" not in result.output
        assert not (tmp_path / "data" / "live" / "state.json").exists()

    def test_lock_is_released_after_run_allows_rerun(
        self, tmp_path, monkeypatch, mocker
    ) -> None:
        monkeypatch.chdir(tmp_path)
        config = _write_config(tmp_path, LIVE_CONFIG)
        fake = FakeCcxt(donchian_rows([100.0] * 6), ticker_price=100.0)
        first = self._invoke_once(tmp_path, mocker, fake, config)
        assert first.exit_code == 0, first.output

        # Lock-файл остаётся на диске, но блокировка снята: повторный запуск
        # в новом процессе проходит (см. также test_second_once_does_not_duplicate).
        lock_path = tmp_path / "data" / "live" / "state.json.lock"
        assert lock_path.exists()
        with lock_path.open("w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)  # не занимает никто

        _patched_client(mocker, fake)
        second = runner.invoke(app, ["live", "--config", str(config), "--once"])
        assert second.exit_code == 0, second.output


@pytest.mark.parametrize(
    ("bad_field", "bad_value"),
    [("timeframe", "7x"), ("mode", "real"), ("poll_seconds", "0"), ("symbol", "BTCUSDT")],
)
def test_invalid_config_fails_cleanly(tmp_path, bad_field: str, bad_value: str) -> None:
    raw = yaml.safe_load(LIVE_CONFIG)
    raw[bad_field] = bad_value
    config = _write_config(tmp_path, yaml.safe_dump(raw))

    result = runner.invoke(app, ["live", "--config", str(config), "--once"])

    assert result.exit_code != 0


def test_colliding_switch_paths_fail_cleanly(tmp_path, monkeypatch, mocker) -> None:
    monkeypatch.chdir(tmp_path)
    raw = yaml.safe_load(LIVE_CONFIG)
    raw["pause_switch_path"] = raw["kill_switch_path"]
    config = _write_config(tmp_path, yaml.safe_dump(raw))
    fake = FakeCcxt(donchian_rows([100.0] * 6), ticker_price=100.0)
    _patched_client(mocker, fake)

    result = runner.invoke(app, ["live", "--config", str(config), "--once"])

    assert result.exit_code != 0
    assert "must be distinct" in result.output


def test_banner_shows_both_switches(tmp_path, monkeypatch, mocker) -> None:
    monkeypatch.chdir(tmp_path)
    config = _write_config(tmp_path, LIVE_CONFIG)
    fake = FakeCcxt(donchian_rows([100.0] * 6), ticker_price=100.0)
    _patched_client(mocker, fake)

    result = runner.invoke(app, ["live", "--config", str(config), "--once"])

    assert result.exit_code == 0, result.output
    assert "Kill switch (data/live/STOP): off" in result.output
    assert "Pause switch (data/live/PAUSE): off" in result.output


def _ok_telegram_response():
    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    return _Response()


class TestTelegramEnv:
    """Пустые поля конфига дополняются из env; значения не логируются."""

    def test_env_variables_enable_notifications(
        self, tmp_path, monkeypatch, mocker, caplog
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "env-secret-token-777")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
        config = _write_config(tmp_path, LIVE_CONFIG)
        fake = FakeCcxt(donchian_rows([100.0] * 6), ticker_price=100.0)
        _patched_client(mocker, fake)
        urlopen = mocker.patch(
            "trading_bot.live.notify.urlopen", return_value=_ok_telegram_response()
        )

        with caplog.at_level(logging.INFO):
            first = runner.invoke(app, ["live", "--config", str(config), "--once"])

        assert first.exit_code == 0, first.output
        assert "telegram notifications enabled" in caplog.text
        # --once (cron): старт-сообщения и heartbeat нет — в чистый прогревный
        # цикл Telegram вообще не дёргается.
        urlopen.assert_not_called()

        # Свеча-пробой: второй cron-запуск исполняет вход и шлёт trade-уведомление.
        fake.rows.append(donchian_rows([100.0] * 6 + [120.0])[-1])
        fake.ticker_price = 120.0
        _patched_client(mocker, fake)
        second = runner.invoke(app, ["live", "--config", str(config), "--once"])

        assert second.exit_code == 0, second.output
        assert urlopen.call_count == 1
        request = urlopen.call_args.args[0]
        assert "/botenv-secret-token-777/" in request.full_url
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["chat_id"] == "42"
        # Значения не попадают ни в stdout, ни в логи.
        assert "env-secret-token-777" not in second.output
        assert "env-secret-token-777" not in caplog.text

    def test_without_env_notifications_are_disabled(
        self, tmp_path, monkeypatch, mocker, caplog
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        config = _write_config(tmp_path, LIVE_CONFIG)
        fake = FakeCcxt(donchian_rows([100.0] * 6), ticker_price=100.0)
        _patched_client(mocker, fake)
        urlopen = mocker.patch("trading_bot.live.notify.urlopen")

        with caplog.at_level(logging.INFO):
            result = runner.invoke(app, ["live", "--config", str(config), "--once"])

        assert result.exit_code == 0, result.output
        assert "notifications are disabled" in caplog.text
        urlopen.assert_not_called()

    def test_config_fields_take_priority_over_env(self, monkeypatch) -> None:
        # Поля конфига приоритетнее окружения (код остаётся гибким): проверяем
        # сборщика нотификатора напрямую — в --once на прогревном цикле
        # сообщений нет, сетью ходить не из чего.
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "env-token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
        cfg = LiveConfig.model_validate(
            {"telegram_bot_token": "config-token", "telegram_chat_id": "99"}
        )
        notifier = _build_notifier(cfg)

        assert notifier.enabled is True
        assert notifier._bot_token == "config-token"
        assert notifier._chat_id == "99"
