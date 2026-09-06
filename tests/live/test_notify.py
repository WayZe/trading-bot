"""Тесты Telegram-уведомителя (офлайн: urlopen подменяется)."""

from __future__ import annotations

import json
import logging
from urllib.error import HTTPError

import pytest

from trading_bot.live.notify import Notifier, NullNotifier

TOKEN = "123456:ABC-secret-token"
CHAT_ID = "42"

# Тестовый токен и его части не должны просочиться в логи ни в одном сценарии.
TOKEN_PARTS = (TOKEN, "ABC-secret", "123456")


def _sent_payload(mock_urlopen) -> dict:
    """Достать JSON-payload из первого вызова подменённого urlopen."""
    request = mock_urlopen.call_args.args[0]
    return json.loads(request.data.decode("utf-8"))


def _response(status: int = 200):
    """Подмена HTTP-ответа Telegram (context manager, как urlopen)."""

    class _Response:
        def __init__(self) -> None:
            self.status = status

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    return _Response()


class TestEnabledDisabled:
    def test_enabled_only_with_token_and_chat_id(self) -> None:
        assert Notifier(TOKEN, CHAT_ID).enabled is True

    @pytest.mark.parametrize(
        ("token", "chat_id"),
        [(None, None), (TOKEN, None), (None, CHAT_ID), ("", "")],
    )
    def test_disabled_without_token_or_chat_id(self, token, chat_id, caplog) -> None:
        with caplog.at_level(logging.INFO, logger="trading_bot.live.notify"):
            notifier = Notifier(token, chat_id)

        assert notifier.enabled is False
        assert "disabled" in caplog.text
        # Факт выключения логируется, значения — никогда.
        assert "secret" not in caplog.text

    def test_disabled_send_returns_false_without_network(self, mocker) -> None:
        urlopen = mocker.patch("trading_bot.live.notify.urlopen")
        notifier = Notifier(None, None)

        assert notifier.send("привет") is False
        urlopen.assert_not_called()


class TestSend:
    def test_successful_send_posts_json_payload(self, mocker) -> None:
        urlopen = mocker.patch(
            "trading_bot.live.notify.urlopen", return_value=_response(200)
        )
        notifier = Notifier(TOKEN, CHAT_ID)

        assert notifier.send("<b>BTC/USDT</b> & рост") is True

        request = urlopen.call_args.args[0]
        # Токен — в URL, payload — корректный JSON с chat_id/text/parse_mode.
        assert request.full_url == f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        assert request.data == json.dumps(
            {"chat_id": CHAT_ID, "text": "<b>BTC/USDT</b> & рост", "parse_mode": "HTML"}
        ).encode("utf-8")

    @pytest.mark.parametrize("status", [400, 401, 403, 500])
    def test_http_error_returns_false_and_masks_token(self, mocker, caplog, status) -> None:
        mocker.patch(
            "trading_bot.live.notify.urlopen",
            side_effect=HTTPError(
                f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                status,
                "Bad Request",
                {},
                None,
            ),
        )
        notifier = Notifier(TOKEN, CHAT_ID)

        with caplog.at_level(logging.WARNING, logger="trading_bot.live.notify"):
            assert notifier.send("текст") is False

        assert "telegram send failed" in caplog.text
        for part in TOKEN_PARTS:
            assert part not in caplog.text
        assert str(status) in caplog.text

    def test_mask_token_replaces_it_even_inside_urls(self) -> None:
        notifier = Notifier(TOKEN, CHAT_ID)

        masked = notifier._mask_token(f"https://api.telegram.org/bot{TOKEN}/sendMessage")

        assert masked == "https://api.telegram.org/bot***/sendMessage"

    def test_unexpected_exception_returns_false_and_masks_token(self, mocker, caplog) -> None:
        mocker.patch(
            "trading_bot.live.notify.urlopen",
            side_effect=RuntimeError(f"boom at bot{TOKEN}/sendMessage"),
        )
        notifier = Notifier(TOKEN, CHAT_ID)

        with caplog.at_level(logging.WARNING, logger="trading_bot.live.notify"):
            assert notifier.send("текст") is False

        assert "boom" in caplog.text
        for part in TOKEN_PARTS:
            assert part not in caplog.text

    def test_oversized_text_is_truncated_to_telegram_limit(self, mocker) -> None:
        urlopen = mocker.patch(
            "trading_bot.live.notify.urlopen", return_value=_response(200)
        )
        notifier = Notifier(TOKEN, CHAT_ID)

        assert notifier.send("x" * 10_000) is True
        assert len(_sent_payload(urlopen)["text"]) == 4000


class TestThrottling:
    def _notifier(self) -> Notifier:
        return Notifier(TOKEN, CHAT_ID, error_throttle_minutes=60)

    def test_error_category_is_throttled_within_window(self, mocker) -> None:
        urlopen = mocker.patch(
            "trading_bot.live.notify.urlopen", return_value=_response(200)
        )
        notifier = self._notifier()
        mocker.patch.object(Notifier, "_now_ts", return_value=1000.0)

        assert notifier.send("сбой 1", category="error") is True
        assert urlopen.call_count == 1

        # Второй сбой в том же окне отбрасывается без сетевого запроса.
        assert notifier.send("сбой 2", category="error") is False
        assert urlopen.call_count == 1

        # Окно вышло — сообщение снова уходит.
        mocker.patch.object(Notifier, "_now_ts", return_value=1000.0 + 3600.0)
        assert notifier.send("сбой 3", category="error") is True
        assert urlopen.call_count == 2

    def test_error_throttle_counts_attempts_not_successes(self, mocker) -> None:
        # Неудачная попытка тоже занимает окно: упавшая сеть не должна
        # превращаться в шторм запросов с таймаутом каждый цикл.
        urlopen = mocker.patch(
            "trading_bot.live.notify.urlopen", side_effect=RuntimeError("down")
        )
        notifier = self._notifier()
        mocker.patch.object(Notifier, "_now_ts", return_value=1000.0)

        assert notifier.send("сбой 1", category="error") is False
        assert notifier.send("сбой 2", category="error") is False
        assert urlopen.call_count == 1

    def test_critical_trade_switch_and_info_are_never_throttled(self, mocker) -> None:
        urlopen = mocker.patch(
            "trading_bot.live.notify.urlopen", return_value=_response(200)
        )
        notifier = self._notifier()
        mocker.patch.object(Notifier, "_now_ts", return_value=1000.0)

        for category in ("critical", "trade", "switch", "info"):
            assert notifier.send("a", category=category) is True
            assert notifier.send("b", category=category) is True
        assert urlopen.call_count == 8

    def test_zero_throttle_minutes_sends_every_error(self, mocker) -> None:
        urlopen = mocker.patch(
            "trading_bot.live.notify.urlopen", return_value=_response(200)
        )
        notifier = Notifier(TOKEN, CHAT_ID, error_throttle_minutes=0)
        mocker.patch.object(Notifier, "_now_ts", return_value=1000.0)

        assert notifier.send("сбой 1", category="error") is True
        assert notifier.send("сбой 2", category="error") is True
        assert urlopen.call_count == 2


class TestNullNotifier:
    def test_send_is_a_silent_no_op(self, mocker) -> None:
        urlopen = mocker.patch("trading_bot.live.notify.urlopen")
        notifier = NullNotifier()

        assert notifier.enabled is False
        assert notifier.send("что угодно", category="critical") is False
        urlopen.assert_not_called()
