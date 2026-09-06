"""Telegram-уведомления live-раннера (stdlib urllib, без новых зависимостей).

Инвариант: сбой отправки **никогда** не рвёт торговый цикл — :meth:`Notifier.send`
глотает любые исключения (и не-200 ответы), логирует warning (токен при этом
маскируется и в лог не попадает никогда) и возвращает ``False``. Раннер вызывает
``send`` напрямую, без дополнительного try/except.

Категории сообщений:

- ``critical`` — старт раннера, ``needs_attention`` (без троттлинга);
- ``trade`` — входы/выходы, стоп/тейк по тикеру (без троттлинга);
- ``switch`` — переходы STOP/PAUSE (без троттлинга);
- ``error`` — сетевые/данные сбои; троттлинг: не чаще раза в
  ``error_throttle_minutes`` (первые всегда уходят);
- ``info`` — heartbeat-дайджест (без троттлинга; ритм задаёт сам раннер).
"""

from __future__ import annotations

import html
import json
import logging
import time
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# Telegram отклоняет сообщения длиннее 4096 символов; страхуемся с запасом.
_MAX_TEXT_LENGTH = 4000
_SEND_TIMEOUT_SECONDS = 10.0

# Категория с троттлингом (сетевые/данные сбои): повторные подряд — глушатся.
_THROTTLED_CATEGORY = "error"


class Notifier:
    """Отправка сообщений в Telegram через Bot API (sendMessage).

    Args:
        bot_token: токен бота от @BotFather; вместе с ``chat_id`` включает
            уведомления (оба ``None``/пустые — уведомления выключены).
        chat_id: идентификатор чата/канала получателя.
        error_throttle_minutes: минимальный интервал между сообщениями
            категории ``error``, минуты.
    """

    def __init__(
        self,
        bot_token: str | None,
        chat_id: str | None,
        *,
        error_throttle_minutes: int = 60,
    ) -> None:
        self.enabled = bool(bot_token) and bool(chat_id)
        self._error_throttle_seconds = max(error_throttle_minutes, 0) * 60
        self._chat_id = chat_id
        self._bot_token = bot_token or ""
        self._last_error_sent_at: float | None = None
        if self.enabled:
            logger.info("telegram notifications enabled")
        else:
            logger.info(
                "telegram notifications are disabled: no bot token/chat id configured"
            )

    def send(self, text: str, *, category: str = "info") -> bool:
        """Отправить текст в Telegram; вернуть ``True`` при успехе.

        Любая ошибка (сеть, не-200, невалидный токен) — warning в лог с
        замаскированным токеном и ``False``; исключения наружу не проходят.
        Категория ``error`` троттлится: в пределах окна повторные сообщения
        отбрасываются без сетевого запроса.

        Args:
            text: текст сообщения (HTML-разметка допустима; ужимается до лимита).
            category: категория для троттлинга (``error`` — единственная
                троттлируемая).

        Returns:
            ``True``, если Telegram принял сообщение, иначе ``False``.
        """
        if not self.enabled:
            return False
        if self._is_throttled(category):
            logger.debug("telegram %s notification throttled (repeated within window)", category)
            return False
        return self._post(text)

    # ------------------------------------------------------------------
    # Внутреннее
    # ------------------------------------------------------------------

    def _is_throttled(self, category: str) -> bool:
        """Проверить (и при отправке обновить) окно троттлинга категории ``error``."""
        if category != _THROTTLED_CATEGORY:
            return False
        now = self._now_ts()
        last = self._last_error_sent_at
        if last is not None and now - last < self._error_throttle_seconds:
            return True
        self._last_error_sent_at = now
        return False

    def _post(self, text: str) -> bool:
        """Выполнить sendMessage; все исключения глотаются с маскировкой токена."""
        url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        payload = json.dumps(
            {"chat_id": self._chat_id, "text": text[:_MAX_TEXT_LENGTH], "parse_mode": "HTML"}
        ).encode("utf-8")
        request = Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=_SEND_TIMEOUT_SECONDS) as response:
                if response.status != 200:
                    logger.warning(
                        "telegram send failed: HTTP %s", response.status
                    )
                    return False
            return True
        except Exception as error:  # сбой уведомления не должен выходить наружу
            logger.warning(
                "telegram send failed: %s", self._mask_token(str(error))
            )
            return False

    def _mask_token(self, text: str) -> str:
        """Заменить токен бота на ``***`` (URL исключений содержит его целиком)."""
        if not self._bot_token:
            return text
        return text.replace(self._bot_token, "***")

    @staticmethod
    def _now_ts() -> float:
        """Текущее время (секунды); отдельная точка для замены в тестах."""
        return time.time()


class NullNotifier:
    """Заглушка «уведомления выключены»: ничего не отправляет, ничего не стоит.

    Имеет тот же интерфейс ``send``/``enabled``, что :class:`Notifier`, поэтому
    раннер принимает любой из них — отдельных ``if`` в торговом цикле не нужно.
    """

    enabled = False

    def send(self, text: str, *, category: str = "info") -> bool:
        """Ничего не делать; всегда ``False``.

        Args:
            text: игнорируется.
            category: игнорируется.

        Returns:
            Всегда ``False``.
        """
        return False


def escape(text: str) -> str:
    """Экранировать динамическую часть сообщения для HTML parse_mode."""
    return html.escape(str(text))
