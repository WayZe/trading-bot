"""Цикл live/paper-торговли: тикеры, закрытые свечи, ордера, рестарты.

Один цикл :meth:`LiveRunner.run_once`:

1. Тикер: если позиция открыта и цена пробила активный стоп/тейк —
   немедленный market-sell (гэп через уровень исполняется по фактической
   цене — хуже стопа, как консервативно и в бэктест-движке).
2. Свечи: догрузка закрытых свечей с биржи (пагинация, ретраи, отбрасывание
   открытых — всё это делает :class:`HistoryDownloader`), новая закрытая
   свеча идёт в Parquet-хранилище и в стратегию; сигналы исполняются
   адаптером (paper — симуляция по тикеру, testnet — реальные ордера).
   Уровни стоп/тейк входа переякориваются на фактическую цену исполнения
   теми же функциями, что и в бэктест-движке (``engine.stops``).
3. Heartbeat и персистентность: состояние сохраняется после каждой
   обработанной свечи и каждого исполнения — рестарт посреди батча не
   теряет историю.

Защитные механизмы: kill-switch (файл существует → любые новые ордера
запрещены), флаг ``needs_attention`` (после непонятного состояния ордера
в testnet торговля стоит до ручного разбора), сетевые ошибки не убивают
цикл, а логируются и пропускаются.

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
from trading_bot.live.execution import ExecutionAdapter, FillResult
from trading_bot.live.state import LiveState
from trading_bot.risk import RiskManager
from trading_bot.strategy.base import Fill, Signal, SignalKind, Strategy

logger = logging.getLogger(__name__)

# Полная догрузка при чистом состоянии: совпадает с диапазоном research-датасетов.
BACKFILL_SINCE = "2023-01-01"


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
    """

    def __init__(
        self,
        config: LiveConfig,
        state: LiveState,
        strategy: Strategy,
        adapter: ExecutionAdapter,
        exchange_client: object,
        storage: CandleStorage,
    ) -> None:
        self.config = config
        self.state = state
        self.strategy = strategy
        self.adapter = adapter
        self.exchange_client = exchange_client
        self.storage = storage
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
        никогда не падает из-за одной неудачи.
        """
        try:
            self._run_cycle()
        except ccxt.NetworkError as error:
            logger.warning("network error; skipping this cycle: %s", error)
        except Exception:
            logger.exception("unexpected error in live cycle; continuing on next tick")

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
                self.state.mark_needs_attention()
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

    # ------------------------------------------------------------------
    # Внутренний цикл
    # ------------------------------------------------------------------

    def _run_cycle(self) -> None:
        """Один цикл: тикер → защита позиции → свечи → heartbeat."""
        if self.kill_switch_active():
            logger.warning(
                "kill switch is active at %s: all new orders are forbidden this cycle",
                self.config.kill_switch_path,
            )
        ticker_price = self.exchange_client.fetch_ticker_last(self.config.symbol)
        self._protect_position(ticker_price)
        candles = self._sync_candles()
        self._process_new_candles(candles)
        self._heartbeat(ticker_price)

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
        assert position is not None  # вызывается только при открытой позиции
        fill = self._place_order("sell", position.quantity, reason)
        if fill is None:
            return
        self.state.apply_fill(fill)
        self.state.clear_stops()
        self.strategy.on_fill(self._as_strategy_fill(fill))
        self.state.save(self.config.state_path)
        logger.info(
            "exit filled (%s): %.6f @ %.4f (fee %.4f)", reason, fill.quantity, fill.price, fill.fee
        )

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

    def _process_new_candles(self, candles: pd.DataFrame) -> None:
        """Провести новые закрытые свечи через стратегию и исполнить сигналы.

        Чистый state: вся догруженная история — только прогрев (исторические
        сигналы устарели и по текущей цене не исполняются). Рестарт:
        каждая новая закрытая свеча обрабатывается по очереди, состояние
        сохраняется после каждой — сбой посреди батча не теряет историю.
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
        for row in new_candles.itertuples(index=False):
            candle_ts = pd.Timestamp(row.timestamp)
            ref_close = float(row.close)
            history = candles[candles["timestamp"] <= candle_ts]
            for signal in self.strategy.on_candle(history):
                self._handle_signal(signal, ref_close)
            self.state.set_last_candle(candle_ts)
            self.state.save(self.config.state_path)

    def _handle_signal(self, signal: Signal, ref_close: float) -> None:
        """Исполнить один сигнал стратегии на текущем рынке."""
        if signal.kind is SignalKind.LONG_ENTRY:
            if self.state.position is not None:
                logger.debug("LONG_ENTRY ignored: position already open")
                return
            equity = self.adapter.fetch_equity()
            quantity = self.risk.position_quantity(equity, ref_close)
            if quantity <= 0.0:
                logger.warning(
                    "LONG_ENTRY skipped: quantity below min_notional "
                    "(equity %.2f, close %.2f)",
                    equity,
                    ref_close,
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
            self.strategy.on_fill(self._as_strategy_fill(fill))
            logger.info(
                "entry filled (%s): %.6f @ %.4f, stop=%s tp=%s",
                signal.reason,
                fill.quantity,
                fill.price,
                self.state.active_stop,
                self.state.active_tp,
            )
        elif signal.kind is SignalKind.LONG_EXIT:
            if self.state.position is None:
                logger.debug("LONG_EXIT ignored: no open position")
                return
            fill = self._place_order("sell", self.state.position.quantity, signal.reason)
            if fill is None:
                return
            self.state.apply_fill(fill)
            self.state.clear_stops()
            self.strategy.on_fill(self._as_strategy_fill(fill))
            logger.info(
                "exit filled (%s): %.6f @ %.4f (fee %.4f)",
                signal.reason,
                fill.quantity,
                fill.price,
                fill.fee,
            )
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
        """Единственная точка размещения ордеров: kill-switch, флаг внимания, ошибки.

        Ордер не размещается, пока активен kill-switch или выставлен
        ``needs_attention`` (оба варианта — warning в лог). Ошибка биржи при
        размещении (testnet) означает неясное состояние позиции: error-лог,
        ``needs_attention``, persist — и продолжение без ордера.
        """
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
        except ccxt.ExchangeError:
            logger.exception(
                "order %s (%s) failed on the exchange: position state is unclear, "
                "marking needs_attention",
                side,
                reason,
            )
            self.state.mark_needs_attention()
            self.state.save(self.config.state_path)
            return None

    def _heartbeat(self, ticker_price: float) -> None:
        """Строка наблюдаемости в каждом цикле: режим, позиция, цена, возраст свечи."""
        if self.state.needs_attention:
            logger.error(
                "NEEDS ATTENTION: trading is paused after an unclear order state; "
                "resolve it and clear the flag in %s",
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
            "heartbeat: mode=%s position=%s last_price=%.4f candle_age=%s kill_switch=%s",
            self.config.mode,
            "flat"
            if position is None
            else f"{position.quantity:.6f} @ {position.entry_price:.4f}",
            ticker_price,
            candle_age,
            "active" if self.kill_switch_active() else "off",
        )

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
