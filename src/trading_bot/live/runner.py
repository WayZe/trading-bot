"""Цикл live/paper-торговли: тикеры, закрытые свечи, ордера, рестарты.

Один цикл :meth:`LiveRunner.run_once`:

1. Тикер: если позиция открыта и цена пробила активный стоп/тейк —
   немедленный market-sell (гэп через уровень исполняется по фактической
   цене — хуже стопа, как консервативно и в бэктест-движке).
   Ограничение: уровни проверяются только по тикеру, раз в ``poll_seconds`` —
   «фитиль с восстановлением» внутри закрытой свечи live-раннер не видит
   (бэктест-движок видит свечу целиком); для 4h-свечей рекомендуется
   ``poll_seconds`` не больше 60.
2. Свечи: догрузка закрытых свечей с биржи (пагинация, ретраи, отбрасывание
   открытых — всё это делает :class:`HistoryDownloader`), новая закрытая
   свеча идёт в Parquet-хранилище и в стратегию; сигналы исполняются
   адаптером (paper — симуляция по тикеру, testnet — реальные ордера).
   Уровни стоп/тейк входа переякориваются на фактическую цену исполнения
   теми же функциями, что и в бэктест-движке (``engine.stops``); сайзинг
   входа считается по цене тикера, а не по close сигнальной свечи.
   Догон после простоя: если накопилось несколько закрытых свечей, сигнал
   исполняется только для последней — устаревшие сигналы промежуточных
   свечей по текущей цене не торгуются (стратегия их получает как прогрев).
3. Heartbeat и персистентность: состояние сохраняется сразу после каждого
   исполнения и после каждой обработанной свечи — крэш посреди цикла не
   теряет ни позицию, ни историю.

Уведомления Telegram (опционально, ``live/notify.py``): старт процесса,
входы/выходы и защитные стоп/тейк, подъём ``needs_attention``, переходы
STOP/PAUSE, троттлируемые сетевые сбои и суточный heartbeat-дайджест.
Сбой отправки никогда не рвёт торговый цикл: ``Notifier.send`` глотает
любые исключения сам (токен при этом не попадает в лог).

Гарантия от дубля ордера (testnet): любая ошибка исполнения, оставляющая
неясным факт или результат размещения ордера (NetworkError на создание,
сбой после успешного создания — упал fetch_order, статус не распарсился),
поднимает ``needs_attention`` и останавливает торговлю до ручного разбора;
сигнал при этом считается потреблённым — свеча помечается обработанной и
повторного ордера по тому же сигналу не будет.

Защитные механизмы: STOP (``kill_switch_path``) — запрещены ВСЕ новые
ордера, включая защитные выходы (позиция остаётся без присмотра, что
кричит в лог каждый цикл); PAUSE (``pause_switch_path``) — запрещены
только новые входы, сигнальные выходы и защитные стоп/тейк работают;
флаг ``needs_attention`` (после непонятного состояния ордера в testnet
торговля стоит до ручного разбора); сетевые ошибки не убивают цикл,
а логируются и пропускаются.

Первая догрузка истории (чистый state) — только прогрев: исторические
сигналы устарели, торговать по ним по текущей цене нельзя; торговля
начинается со свечей, закрывшихся уже при работающем раннере.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import ccxt
import pandas as pd

from trading_bot.config import LiveConfig
from trading_bot.data.downloader import EXCHANGE_ID, HistoryDownloader
from trading_bot.data.storage import CandleStorage
from trading_bot.engine.stops import (
    REASON_STOP_LOSS,
    REASON_TAKE_PROFIT,
    reanchor_stop_below,
    reanchor_take_profit_above,
)
from trading_bot.live.execution import ExecutionAdapter, FillResult, OrderStateUncertain
from trading_bot.live.notify import Notifier, NullNotifier, escape
from trading_bot.live.state import LiveState
from trading_bot.risk import RiskManager
from trading_bot.strategy.base import Fill, Signal, SignalKind, Strategy

logger = logging.getLogger(__name__)

# Полная догрузка при чистом состоянии: совпадает с диапазоном research-датасетов.
BACKFILL_SINCE = "2023-01-01"

# Подписи защитных выходов в Telegram-уведомлениях (reason-строки — английские).
_EXIT_LABELS = {
    REASON_STOP_LOSS: "🛑 Стоп-лосс",
    REASON_TAKE_PROFIT: "🎯 Тейк-профит",
}


class LiveRunner:
    """Исполняет стратегию на закрытых свечах Bybit и переживает рестарты.

    Args:
        config: live-конфиг (символ, таймфрейм, лимиты, пути).
        state: персистентное состояние (позиция, стопы, последняя свеча).
        strategy: экземпляр стратегии из реестра (уже с параметрами).
        adapter: адаптер исполнения (paper или testnet).
        exchange_client: обёртка ccxt (публичные данные; приватные вызовы
            выполняет только testnet-адаптер).
        storage: Parquet-хранилище live-свечей (корень из конфига).
        notifier: отправитель Telegram-уведомлений; ``None`` — заглушка
            :class:`NullNotifier` (цикл работает как раньше, ничего не шлёт).
            Сбой отправки не рвёт торговый цикл: ``send`` глотает всё сам.
    """

    def __init__(
        self,
        config: LiveConfig,
        state: LiveState,
        strategy: Strategy,
        adapter: ExecutionAdapter,
        exchange_client: object,
        storage: CandleStorage,
        notifier: Notifier | None = None,
    ) -> None:
        self.config = config
        self.state = state
        self.strategy = strategy
        self.adapter = adapter
        self.exchange_client = exchange_client
        self.storage = storage
        self.notifier = notifier if notifier is not None else NullNotifier()
        # Память уведомлений: старт (один раз за процесс), предыдущие состояния
        # свитчей (первый цикл — начальное состояние, без уведомлений), флаг
        # attention (рестарт с уже поднятым флагом не спамит повторно) и таймер
        # heartbeat-дайджеста (первый — через heartbeat_hours после старта).
        self._start_notified = False
        self._attention_notified = state.needs_attention
        self._stop_active: bool | None = None
        self._pause_active: bool | None = None
        self._last_heartbeat_sent = self._now_ts()
        self.risk = RiskManager(
            position_size_pct=config.position_size_pct,
            quantity_precision=config.quantity_precision,
            min_notional=config.min_notional,
        )
        self.downloader = HistoryDownloader(exchange_client)
        self._warmup_strategy()

    def run_once(self) -> None:
        """Выполнить один цикл; неудача любого шага не прерывает работу раннера.

        Сетевые ошибки логируются как warning (цикл повторится на следующем
        тике), любые другие исключения — как error с traceback; раннер
        никогда не падает из-за одной неудачи. Уведомления: сообщение о
        старте процесса — один раз на первом цикле; сетевой сбой — в
        Telegram (категория ``error``, троттлинг); поднятый за цикл флаг
        ``needs_attention`` — критическое сообщение (только сам переход).
        """
        try:
            self._notify_start_once()
            self._run_cycle()
        except ccxt.NetworkError as error:
            logger.warning("network error; skipping this cycle: %s", error)
            self.notifier.send(
                f"⚠️ network error: {escape(_short(error))}", category="error"
            )
        except Exception:
            logger.exception("unexpected error in live cycle; continuing on next tick")
        self._notify_attention_if_raised()

    def run_forever(self) -> None:
        """Основной цикл с опросом каждые ``poll_seconds``; Ctrl+C — плавный выход."""
        logger.info(
            "live loop started: mode=%s %s %s strategy=%s",
            self.config.mode,
            self.config.symbol,
            self.config.timeframe,
            self.config.strategy,
        )
        try:
            while True:
                self.run_once()
                time.sleep(self.config.poll_seconds)
        except KeyboardInterrupt:
            logger.info("interrupted: shutting down live loop")

    def reconcile(self) -> None:
        """Сверить позицию в state с балансом биржи (testnet).

        По количеству доверяем бирже: расхождение логируется warning'ом и
        state приводится к фактическому балансу базовой валюты. Если биржа
        держит валюту, которой нет в state, цена входа неизвестна — стопы
        пересчитать не от чего, выставляется ``needs_attention``.
        """
        base_currency = self.config.symbol.split("/")[0]
        exchange_qty = self.exchange_client.fetch_free_balance(base_currency)
        state_qty = 0.0 if self.state.position is None else self.state.position.quantity
        if math.isclose(exchange_qty, state_qty, abs_tol=1e-9):
            logger.info(
                "reconcile: state and exchange agree (%.6f %s)", exchange_qty, base_currency
            )
            return
        logger.warning(
            "reconcile: state position %.6f != exchange balance %.6f %s; "
            "trusting the exchange",
            state_qty,
            exchange_qty,
            base_currency,
        )
        if self.state.position is None:
            if exchange_qty > 0:
                logger.error(
                    "reconcile: exchange holds %.6f %s without a state position; "
                    "entry price is unknown, marking needs_attention",
                    exchange_qty,
                    base_currency,
                )
                self.state.mark_needs_attention(
                    f"exchange holds {exchange_qty:.6f} {base_currency} without "
                    "a state position: entry price and stops are unknown, "
                    "inspect the exchange manually"
                )
        elif exchange_qty <= 0:
            logger.warning(
                "reconcile: position is missing on the exchange; "
                "clearing state position and stops"
            )
            self.state.position = None
            self.state.clear_stops()
        else:
            self.state.position.quantity = exchange_qty
        self.state.save(self.config.state_path)

    def kill_switch_active(self) -> bool:
        """True, пока существует файл kill-switch (новые ордера запрещены)."""
        return Path(self.config.kill_switch_path).exists()

    def pause_switch_active(self) -> bool:
        """True, пока существует файл pause-switch (новые входы запрещены)."""
        return Path(self.config.pause_switch_path).exists()

    # ------------------------------------------------------------------
    # Внутренний цикл
    # ------------------------------------------------------------------

    def _run_cycle(self) -> None:
        """Один цикл: свитчи → тикер → защита позиции → свечи → heartbeat."""
        self._check_switches()
        ticker_price = self.exchange_client.fetch_ticker_last(self.config.symbol)
        if self.kill_switch_active():
            logger.warning(
                "kill switch is active at %s: all new orders are forbidden this cycle",
                self.config.kill_switch_path,
            )
            self._log_unprotected_position(ticker_price)
        self._protect_position(ticker_price)
        candles = self._sync_candles()
        self._process_new_candles(candles, ticker_price)
        self._heartbeat(ticker_price)
        self._notify_heartbeat()

    def _check_switches(self) -> None:
        """Уведомить о переходах STOP/PAUSE (только об изменении состояния).

        Первый цикл после старта фиксирует начальное состояние без
        уведомлений — рестарт с существующим свитчем не спамит повтором.
        """
        stop_active = self.kill_switch_active()
        pause_active = self.pause_switch_active()
        if self._stop_active is not None:
            if stop_active != self._stop_active:
                self.notifier.send(
                    "🛑 STOP активирован: любые новые ордера запрещены "
                    "(открытая позиция остаётся без защиты)."
                    if stop_active
                    else "✅ STOP снят: ордера снова разрешены.",
                    category="switch",
                )
            if pause_active != self._pause_active:
                self.notifier.send(
                    "⏸ PAUSE активирован: новые входы запрещены, выходы "
                    "и защитные стоп/тейк работают."
                    if pause_active
                    else "▶️ PAUSE снят: новые входы снова разрешены.",
                    category="switch",
                )
        self._stop_active = stop_active
        self._pause_active = pause_active

    def _log_unprotected_position(self, ticker_price: float) -> None:
        """STOP с открытой позицией: error каждый цикл — цена и дистанция до стопа.

        Kill-switch запрещает и защитный выход, поэтому позиция остаётся без
        присмотра; человек должен видеть, насколько цена близка к стопу,
        чтобы снять свитч или закрыть позицию вручную.
        """
        stop = self.state.active_stop
        if self.state.position is None or stop is None:
            return
        logger.error(
            "kill switch is active with an open position: ticker %.4f, stop %.4f "
            "(distance %.4f); the position is NOT protected while the switch exists",
            ticker_price,
            stop,
            ticker_price - stop,
        )

    def _protect_position(self, ticker_price: float) -> None:
        """Проверить открытую позицию по тикеру: пробой стопа/тейка → market-sell."""
        if self.state.position is None:
            return
        if (
            self.state.active_stop is not None
            and ticker_price <= self.state.active_stop
        ):
            exit_reason = REASON_STOP_LOSS
        elif (
            self.state.active_tp is not None
            and ticker_price >= self.state.active_tp
        ):
            exit_reason = REASON_TAKE_PROFIT
        else:
            return
        logger.warning(
            "ticker %.4f breached %s level: exiting position immediately "
            "(gap fills at the actual price)",
            ticker_price,
            exit_reason,
        )
        self._close_position(exit_reason)

    def _close_position(self, reason: str) -> None:
        """Продать позицию целиком, сбросить стопы и сохранить состояние."""
        position = self.state.position
        if position is None:
            return
        fill = self._place_order("sell", position.quantity, reason)
        if fill is None:
            return
        self.state.apply_fill(fill)
        if self.state.position is None:
            self.state.clear_stops()
        else:
            # Частичное исполнение (продано меньше, чем просили): apply_fill
            # уже уменьшил позицию и поднял needs_attention — стопы остаются.
            logger.error(
                "partial %s fill: %.6f of %.6f sold @ %.4f, %.6f remains; "
                "trading is paused for manual review",
                reason,
                fill.quantity,
                position.quantity,
                fill.price,
                self.state.position.quantity,
            )
        self.state.save(self.config.state_path)
        self.strategy.on_fill(self._as_strategy_fill(fill))
        logger.info(
            "exit filled (%s): %.6f @ %.4f (fee %.4f)", reason, fill.quantity, fill.price, fill.fee
        )
        self._notify_exit(_EXIT_LABELS.get(reason, "📉 Выход"), position, fill)

    def _sync_candles(self) -> pd.DataFrame:
        """Догрузить закрытые свечи с биржи в live-хранилище и вернуть датасет.

        Существующий датасет обновляется инкрементально (downloader сам
        перезапрашивает последнюю свечу — она могла быть записана открытой).
        Чистое состояние: полная догрузка с ``BACKFILL_SINCE``.
        """
        existing = self.storage.load(EXCHANGE_ID, self.config.symbol, self.config.timeframe)
        if existing is not None and not existing.empty:
            return self.downloader.update(self.config.symbol, self.config.timeframe, self.storage)
        logger.info(
            "no stored live dataset for %s %s: full backfill since %s",
            self.config.symbol,
            self.config.timeframe,
            BACKFILL_SINCE,
        )
        candles = self.downloader.download(
            self.config.symbol, self.config.timeframe, since=BACKFILL_SINCE
        )
        self.storage.save(EXCHANGE_ID, self.config.symbol, self.config.timeframe, candles)
        return candles

    def _process_new_candles(self, candles: pd.DataFrame, ticker_price: float) -> None:
        """Провести новые закрытые свечи через стратегию и исполнить сигналы.

        Чистый state: вся догруженная история — только прогрев (исторические
        сигналы устарели и по текущей цене не исполняются). Рестарт/простой:
        каждая новая закрытая свеча обрабатывается по очереди, состояние
        сохраняется после каждой — сбой посреди батча не теряет историю.
        Если накопилось несколько свечей, торговый сигнал исполняется только
        для последней: сигналы промежуточных свечей к текущему моменту
        устарели, они остаются стратегией как прогрев (warning с числом
        прогретых свечей). Батч из одной свечи торгуется как раньше.
        """
        last_ts = self.state.last_candle_timestamp()
        if last_ts is None:
            if candles.empty:
                return
            self.strategy.on_candle(candles)
            self.state.set_last_candle(candles["timestamp"].iloc[-1])
            self.state.save(self.config.state_path)
            logger.info(
                "initial backfill done: %d closed candle(s) warmed up; "
                "trading starts from the next closed candle",
                len(candles),
            )
            return
        if candles.empty:
            return
        new_candles = candles[candles["timestamp"] > last_ts]
        if len(new_candles) > 1:
            logger.warning(
                "catch-up after downtime: %d new candle(s) accumulated; executing "
                "signals only for the latest, %d stale candle(s) are strategy "
                "warmup without order execution",
                len(new_candles),
                len(new_candles) - 1,
            )
        for index, row in enumerate(new_candles.itertuples(index=False)):
            candle_ts = pd.Timestamp(row.timestamp)
            ref_close = float(row.close)
            history = candles[candles["timestamp"] <= candle_ts]
            signals = self.strategy.on_candle(history)
            if index == len(new_candles) - 1:
                # Только последняя свеча батча торгуема (или батч из одной).
                for signal in signals:
                    self._handle_signal(signal, ref_close, ticker_price)
            self.state.set_last_candle(candle_ts)
            self.state.save(self.config.state_path)

    def _handle_signal(
        self, signal: Signal, ref_close: float, ticker_price: float
    ) -> None:
        """Исполнить один сигнал стратегии на текущем рынке.

        Сайзинг входа считается по цене тикера: фактическое исполнение ближе
        к текущей рыночной цене, чем к close сигнальной свечи. Дистанции
        стоп/тейк, напротив, заданы стратегией от close сигнальной свечи и
        переносятся на цену исполнения общими функциями ``engine.stops``.
        Позиция и стопы сохраняются в state сразу после исполнения входа:
        крэш до конца цикла не должен «потерять» открытую позицию.
        """
        if signal.kind is SignalKind.LONG_ENTRY:
            if self.state.position is not None:
                logger.debug("LONG_ENTRY ignored: position already open")
                return
            equity = self.adapter.fetch_equity()
            quantity = self.risk.position_quantity(equity, ticker_price)
            if quantity <= 0.0:
                logger.warning(
                    "LONG_ENTRY skipped: quantity below min_notional "
                    "(equity %.2f, ticker %.2f)",
                    equity,
                    ticker_price,
                )
                return
            stop_distance = (
                None if signal.stop_loss is None else ref_close - signal.stop_loss
            )
            tp_distance = (
                None if signal.take_profit is None else signal.take_profit - ref_close
            )
            fill = self._place_order(
                "buy", quantity, signal.reason, stop_distance, tp_distance
            )
            if fill is None:
                return
            self.state.apply_fill(fill)
            # Переякоривание — те же функции, что в бэктест-движке (engine.stops):
            # дистанция от close сигнальной свечи переносится на цену исполнения.
            self.state.set_stops(
                reanchor_stop_below(signal.stop_loss, ref_close, fill.price),
                reanchor_take_profit_above(signal.take_profit, ref_close, fill.price),
            )
            self.state.save(self.config.state_path)
            self.strategy.on_fill(self._as_strategy_fill(fill))
            logger.info(
                "entry filled (%s): %.6f @ %.4f, stop=%s tp=%s",
                signal.reason,
                fill.quantity,
                fill.price,
                self.state.active_stop,
                self.state.active_tp,
            )
            stop_text = (
                "нет"
                if self.state.active_stop is None
                else f"{self.state.active_stop:.4f}"
            )
            self.notifier.send(
                f"📈 LONG {escape(self.config.symbol)}: "
                f"{fill.quantity:.6f} @ {fill.price:.4f}, стоп @ {stop_text} "
                f"({escape(signal.reason)})",
                category="trade",
            )
        elif signal.kind is SignalKind.LONG_EXIT:
            if self.state.position is None:
                logger.debug("LONG_EXIT ignored: no open position")
                return
            position = self.state.position
            fill = self._place_order("sell", self.state.position.quantity, signal.reason)
            if fill is None:
                return
            self.state.apply_fill(fill)
            if self.state.position is None:
                self.state.clear_stops()
            else:
                # Частичное исполнение: apply_fill поднял needs_attention,
                # стопы при остатке позиции сохраняются.
                logger.error(
                    "partial %s fill: %.6f of %.6f sold @ %.4f, %.6f remains; "
                    "trading is paused for manual review",
                    signal.reason,
                    fill.quantity,
                    self.state.position.quantity + fill.quantity,
                    fill.price,
                    self.state.position.quantity,
                )
            self.state.save(self.config.state_path)
            self.strategy.on_fill(self._as_strategy_fill(fill))
            logger.info(
                "exit filled (%s): %.6f @ %.4f (fee %.4f)",
                signal.reason,
                fill.quantity,
                fill.price,
                fill.fee,
            )
            self._notify_exit("📉 Выход", position, fill)
        else:
            logger.warning("unknown signal kind %r ignored", signal.kind)

    def _place_order(
        self,
        side: str,
        quantity: float,
        reason: str,
        stop_distance: float | None = None,
        tp_distance: float | None = None,
    ) -> FillResult | None:
        """Единственная точка размещения ордеров: свитчи, флаг внимания, ошибки.

        Свитчи: STOP запрещает любые ордера; PAUSE — только входы (``side ==
        "buy"``), выходы и защитные стоп/тейк проходят. ``needs_attention``
        блокирует всё до ручного сброса флага (все варианты — warning/error
        в лог).

        Гарантия от дубля ордера (B1): ошибка исполнения, оставляющая неясным
        факт или результат размещения (``OrderStateUncertain`` от testnet-
        адаптера, отклонение биржей), поднимает ``needs_attention`` с текстом
        причины и возвращает ``None`` — исключение не уходит в общий цикл,
        свеча помечается обработанной и повторного ордера по тому же сигналу
        не будет. Ошибка, гарантированно оставляющая ордер несозданным
        (валидация ccxt до отправки), повторится на следующем цикле безопасно.
        """
        if side == "buy" and self.pause_switch_active():
            logger.warning(
                "entry order (%s) blocked: pause switch is active at %s "
                "(exits are still allowed)",
                reason,
                self.config.pause_switch_path,
            )
            return None
        if self.kill_switch_active():
            logger.warning(
                "order %s (%s) blocked: kill switch is active", side, reason
            )
            return None
        if self.state.needs_attention:
            logger.error(
                "order %s (%s) blocked: needs_attention flag is set; "
                "resolve the state and clear the flag in %s",
                side,
                reason,
                self.config.state_path,
            )
            return None
        try:
            return self.adapter.execute(
                side, quantity, reason, stop_distance, tp_distance
            )
        except OrderStateUncertain as error:
            logger.error(
                "order %s (%s) is in an unclear state: %s; marking needs_attention, "
                "the signal is consumed and will NOT be retried (duplicate protection)",
                side,
                reason,
                error,
            )
            self.state.mark_needs_attention(str(error))
            self.state.save(self.config.state_path)
            return None
        except ccxt.ExchangeError:
            logger.exception(
                "order %s (%s) failed on the exchange: position state is unclear, "
                "marking needs_attention; the signal is consumed and will NOT be "
                "retried",
                side,
                reason,
            )
            self.state.mark_needs_attention(
                f"{side} order ({reason}) failed on the exchange: verify whether "
                "it was placed and check open orders before resuming"
            )
            self.state.save(self.config.state_path)
            return None

    def _heartbeat(self, ticker_price: float) -> None:
        """Строка наблюдаемости в каждом цикле: режим, позиция, цена, свитчи."""
        if self.state.needs_attention:
            logger.error(
                "NEEDS ATTENTION: trading is paused after an unclear order state%s; "
                "resolve it and clear the flag in %s",
                (
                    f" ({self.state.attention_reason})"
                    if self.state.attention_reason
                    else ""
                ),
                self.config.state_path,
            )
        position = self.state.position
        last_ts = self.state.last_candle_timestamp()
        candle_age = (
            "n/a"
            if last_ts is None
            else str(pd.Timestamp(self._now_ms(), unit="ms", tz="UTC") - last_ts)
        )
        logger.info(
            "heartbeat: mode=%s position=%s last_price=%.4f candle_age=%s "
            "kill_switch=%s pause=%s",
            self.config.mode,
            "flat"
            if position is None
            else f"{position.quantity:.6f} @ {position.entry_price:.4f}",
            ticker_price,
            candle_age,
            "active" if self.kill_switch_active() else "off",
            "active" if self.pause_switch_active() else "off",
        )

    # ------------------------------------------------------------------
    # Telegram-уведомления (сбои send не рвут цикл: send глотает всё сам)
    # ------------------------------------------------------------------

    def _notify_start_once(self) -> None:
        """Сообщить о старте процесса — один раз, на первом цикле раннера."""
        if self._start_notified:
            return
        self._start_notified = True
        position = self.state.position
        position_text = (
            "flat"
            if position is None
            else f"LONG {position.quantity:.6f} @ {position.entry_price:.4f}"
        )
        self.notifier.send(
            f"🟢 Раннер запущен: {escape(self.config.symbol)} "
            f"{self.config.timeframe}, {escape(self.config.strategy)}, "
            f"режим {self.config.mode}, позиция: {position_text}",
            category="critical",
        )

    def _notify_attention_if_raised(self) -> None:
        """Сообщить о подъёме ``needs_attention`` — только о самом переходе.

        Флаг, уже поднятый на момент старта (рестарт с неразобранным
        состоянием), повторного сообщения не порождает.
        """
        if not self.state.needs_attention or self._attention_notified:
            return
        self._attention_notified = True
        reason = escape(self.state.attention_reason or "unknown reason")
        self.notifier.send(
            f"⚠️ Требуется внимание: {reason}. "
            "Торговля остановлена до ручного вмешательства.",
            category="critical",
        )

    def _notify_exit(self, label: str, position, fill: FillResult) -> None:
        """Сообщить о выходе из позиции: цена, объём и приблизительный PnL.

        PnL считается от цены входа state (без учёта комиссий) по фактической
        цене исполнения fill — в paper точно, в testnet по фактическим ценам.

        Args:
            label: подпись события («📉 Выход», «🛑 Стоп-лосс», «🎯 Тейк-профит»).
            position: позиция до выхода (источник цены входа).
            fill: факт исполнения продажи.
        """
        pnl = fill.quantity * (fill.price - position.entry_price)
        self.notifier.send(
            f"{label}: {fill.quantity:.6f} @ {fill.price:.4f}, "
            f"PnL ~{pnl:+.2f} USDT ({escape(fill.reason)})",
            category="trade",
        )

    def _notify_heartbeat(self) -> None:
        """Ежедневный дайджест-heartbeat (``heartbeat_hours``, 0 — выключен).

        Первый дайджест уходит через ``heartbeat_hours`` после старта процесса
        (таймер инициализируется временем старта), дальше — раз в период.
        """
        interval_seconds = self.config.heartbeat_hours * 3600.0
        if interval_seconds <= 0.0:
            return
        now = self._now_ts()
        if now - self._last_heartbeat_sent < interval_seconds:
            return
        self._last_heartbeat_sent = now
        position = self.state.position
        position_text = (
            "flat"
            if position is None
            else f"LONG {position.quantity:.6f} @ {position.entry_price:.4f}"
        )
        last_ts = self.state.last_candle_timestamp()
        if last_ts is None:
            candle_age = "n/a"
        else:
            age_hours = (
                pd.Timestamp(self._now_ms(), unit="ms", tz="UTC") - last_ts
            ).total_seconds() / 3600.0
            candle_age = f"{age_hours:.1f}h"
        self.notifier.send(
            f"💤 Heartbeat: позиция {position_text}, "
            f"equity {self._equity_text()}, last candle age {candle_age}, "
            f"STOP {'on' if self.kill_switch_active() else 'off'}, "
            f"PAUSE {'on' if self.pause_switch_active() else 'off'}",
            category="info",
        )

    def _equity_text(self) -> str:
        """Текст эквити для дайджеста; недоступное эквити — ``n/a``.

        В paper эквити ведётся в state; в testnet берётся с биржи (приватный
        вызов — только в testnet), сбой запроса не рвёт дайджест.
        """
        if self.config.mode == "testnet":
            try:
                return f"~{self.adapter.fetch_equity():.2f}"
            except Exception:  # дайджест важнее одного неудавшегося запроса
                logger.warning("heartbeat: failed to fetch equity", exc_info=True)
                return "n/a"
        return f"~{self.state.equity:.2f}"

    def _warmup_strategy(self) -> None:
        """Сбросить стратегию и прогреть её на истории из хранилища.

        После рестарта стратегия должна знать, что позиция открыта, — иначе
        стратегии, отслеживающие позицию (например, donchian), перестанут
        выдавать выходные сигналы. Синхронизация — синтетическим on_fill.
        """
        self.strategy.reset()
        history = self.storage.load(EXCHANGE_ID, self.config.symbol, self.config.timeframe)
        if history is not None and not history.empty:
            cutoff = self.state.last_candle_timestamp()
            prefix = history if cutoff is None else history[history["timestamp"] <= cutoff]
            if not prefix.empty:
                # Сигналы прогрева игнорируются: эти свечи уже обработаны.
                self.strategy.on_candle(prefix)
        if self.state.position is not None:
            position = self.state.position
            self.strategy.on_fill(
                Fill(
                    side="buy",
                    price=position.entry_price,
                    quantity=position.quantity,
                    timestamp=pd.Timestamp(position.entry_ts),
                    fee=0.0,
                    reason=position.entry_reason,
                )
            )

    @staticmethod
    def _as_strategy_fill(fill: FillResult) -> Fill:
        """Перевести :class:`FillResult` адаптера в :class:`Fill` стратегии."""
        return Fill(
            side=fill.side,
            price=fill.price,
            quantity=fill.quantity,
            timestamp=fill.ts,
            fee=fill.fee,
            reason=fill.reason,
        )

    @staticmethod
    def _now_ms() -> int:
        """Текущее время UTC в миллисекундах от эпохи (для heartbeat)."""
        return int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)

    @staticmethod
    def _now_ts() -> float:
        """Текущее время в секундах (таймер heartbeat-дайджеста; точка для тестов)."""
        return time.time()


def _short(text: object, limit: int = 200) -> str:
    """Урезать длинный текст ошибки до ``limit`` символов (сообщения без простыней)."""
    return str(text)[:limit]
