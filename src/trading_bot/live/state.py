"""Персистентное состояние live-раннера.

Состояние переживает рестарты: JSON-файл (``LiveConfig.state_path``)
хранит зафиксированные при старте параметры контура, открытую позицию,
активные стоп/тейк-уровни, последнюю обработанную закрытую свечу, equity
paper-режима и список исполненных сделок. Запись атомарная (tmp + rename),
чтобы сбой процесса не оставлял побитый файл.

Инвариант загрузки: отсутствующий файл — это чистый старт (свежее состояние
из конфига), а вот битый JSON — явная ошибка: молча перезаписать состояние
означало бы потерять позицию, поэтому раннер отказывается стартовать.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from trading_bot.config import LiveConfig
from trading_bot.live.execution import FillResult

logger = logging.getLogger(__name__)

STATE_VERSION = 1


@dataclass
class PositionState:
    """Открытая long-позиция в персистентном состоянии.

    Attributes:
        quantity: объём позиции в базовой валюте.
        entry_price: фактическая средняя цена входа.
        entry_ts: момент входа (ISO-строка с таймзоной).
        entry_reason: reason-строка сигнала, открывшего позицию.
    """

    quantity: float
    entry_price: float
    entry_ts: str
    entry_reason: str


@dataclass
class LiveState:
    """Всё, что раннер обязан помнить между циклами и рестартами.

    Attributes:
        version: версия схемы состояния (несовместимая версия — явная ошибка).
        mode: режим исполнения, зафиксированный при старте (``paper``|``testnet``).
        symbol: торговая пара на момент старта.
        timeframe: таймфрейм свечей на момент старта.
        strategy: имя стратегии на момент старта.
        strategy_params: параметры стратегии на момент старта.
        position: открытая позиция или ``None``.
        active_stop: абсолютный уровень стопа (после переякоривания на цену
            входа) или ``None``.
        active_tp: абсолютный уровень тейк-профита или ``None``.
        last_candle_ts: ISO-метка последней *обработанной* закрытой свечи.
        equity: капитал paper-режима (кэш); в testnet не используется —
            эквити берётся с биржи.
        trades: список исполненных сделок (side, price, quantity, ts, reason, fee).
        needs_attention: флаг «состояние неясно» (например, ордер в testnet
            не дошёл или его статус неизвестен); торгуем блокируется до
            ручного разбора.
    """

    version: int = STATE_VERSION
    mode: str = "paper"
    symbol: str = "BTC/USDT"
    timeframe: str = "4h"
    strategy: str = ""
    strategy_params: dict = field(default_factory=dict)
    position: PositionState | None = None
    active_stop: float | None = None
    active_tp: float | None = None
    last_candle_ts: str | None = None
    equity: float = 0.0
    trades: list[dict] = field(default_factory=list)
    needs_attention: bool = False

    def apply_fill(self, fill: FillResult) -> None:
        """Применить исполнение к позиции, equity и списку сделок.

        Покупка открывает позицию (пирамидинг не поддерживается), продажа
        закрывает её целиком (частичные закрытия запрещены). Equity — кэш:
        покупка уменьшает на объём и комиссию, продажа увеличивает на выручку
        минус комиссию.
        """
        record = {
            "side": fill.side,
            "price": fill.price,
            "quantity": fill.quantity,
            "ts": fill.ts.isoformat(),
            "reason": fill.reason,
            "fee": fill.fee,
        }
        self.trades.append(record)
        if fill.side == "buy":
            if self.position is not None:
                raise RuntimeError("position already open: pyramiding is not supported")
            self.equity -= fill.quantity * fill.price + fill.fee
            self.position = PositionState(
                quantity=fill.quantity,
                entry_price=fill.price,
                entry_ts=fill.ts.isoformat(),
                entry_reason=fill.reason,
            )
        elif fill.side == "sell":
            if self.position is None:
                raise RuntimeError("no position to sell")
            self.equity += fill.quantity * fill.price - fill.fee
            self.position = None
        else:
            raise ValueError(f"fill side must be 'buy' or 'sell', got {fill.side!r}")

    def set_stops(self, active_stop: float | None, active_tp: float | None) -> None:
        """Установить активные уровни стоп/тейк (после переякоривания на цену входа)."""
        self.active_stop = active_stop
        self.active_tp = active_tp

    def clear_stops(self) -> None:
        """Сбросить активные уровни (после закрытия позиции)."""
        self.active_stop = None
        self.active_tp = None

    def set_last_candle(self, ts: pd.Timestamp) -> None:
        """Запомнить последнюю обработанную закрытую свечу (ISO с таймзоной)."""
        self.last_candle_ts = pd.Timestamp(ts).isoformat()

    def last_candle_timestamp(self) -> pd.Timestamp | None:
        """Вернуть метку последней обработанной свечи или ``None`` для чистого старта."""
        if self.last_candle_ts is None:
            return None
        return pd.Timestamp(self.last_candle_ts)

    def mark_needs_attention(self) -> None:
        """Выставить флаг «требуется внимание человека» (торговля блокируется)."""
        self.needs_attention = True

    def ensure_matches_config(self, config: LiveConfig) -> None:
        """Проверить, что зафиксированное состояние соответствует текущему конфигу.

        Raises:
            ValueError: если mode/symbol/timeframe/strategy/strategy_params
                не совпадают — продолжать на чужом состоянии опасно; файл
                состояния нужно перенести или удалить (чистый старт).
        """
        mismatches = []
        if self.mode != config.mode:
            mismatches.append(f"mode {self.mode!r} != {config.mode!r}")
        if self.symbol != config.symbol:
            mismatches.append(f"symbol {self.symbol!r} != {config.symbol!r}")
        if self.timeframe != config.timeframe:
            mismatches.append(f"timeframe {self.timeframe!r} != {config.timeframe!r}")
        if self.strategy != config.strategy:
            mismatches.append(f"strategy {self.strategy!r} != {config.strategy!r}")
        if self.strategy_params != config.strategy_params:
            mismatches.append(
                f"strategy_params {self.strategy_params!r} != "
                f"{config.strategy_params!r}"
            )
        if mismatches:
            raise ValueError(
                f"live state {config.state_path!r} does not match the config "
                f"({'; '.join(mismatches)}); move or delete the state file to start fresh"
            )

    def save(self, path: Path | str) -> None:
        """Атомарно записать состояние (JSON: tmp-файл + rename)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.parent / (path.name + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(asdict(self), file, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

    @classmethod
    def load(cls, path: Path | str) -> LiveState | None:
        """Загрузить состояние из JSON; ``None`` — файла ещё нет (чистый старт).

        Raises:
            ValueError: если файл битый (не JSON / не объект / чужая версия
                схемы) — молча перезаписывать состояние с открытой позицией
                нельзя.
        """
        path = Path(path)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as file:
                raw = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"live state file {path} is corrupted: {error}; "
                "inspect and fix or remove it manually — refusing to start over it"
            ) from error
        if not isinstance(raw, dict):
            raise ValueError(
                f"live state file {path} must contain a JSON object, got "
                f"{type(raw).__name__}"
            )
        version = raw.get("version")
        if version != STATE_VERSION:
            raise ValueError(
                f"live state file {path} has unsupported schema version "
                f"{version!r} (expected {STATE_VERSION})"
            )
        position = raw.pop("position", None)
        state = cls(**raw)
        state.position = None if position is None else PositionState(**position)
        return state


def fresh_state(config: LiveConfig) -> LiveState:
    """Собрать свежее состояние для чистого старта из live-конфига.

    Стратегия и её параметры фиксируются при старте; equity paper-режима —
    ``start_cash``, в testnet equity не используется (берётся с биржи).
    """
    return LiveState(
        mode=config.mode,
        symbol=config.symbol,
        timeframe=config.timeframe,
        strategy=config.strategy,
        strategy_params=dict(config.strategy_params),
        equity=config.start_cash if config.mode == "paper" else 0.0,
    )


def load_or_fresh_state(path: Path | str, config: LiveConfig) -> LiveState:
    """Загрузить состояние с диска или собрать свежее для чистого старта."""
    state = LiveState.load(path)
    if state is None:
        logger.info("no live state at %s: starting fresh", path)
        return fresh_state(config)
    logger.info(
        "loaded live state from %s (position=%s, last_candle=%s)",
        path,
        "open" if state.position else "flat",
        state.last_candle_ts,
    )
    return state
