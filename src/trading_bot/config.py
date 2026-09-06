"""Pydantic-модели конфигурации и загрузка YAML."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# Таймфрейм в стиле ccxt: одна или несколько цифр, затем m/h/d (напр. 15m, 4h, 1d).
_TIMEFRAME_PATTERN = re.compile(r"^(\d+)([mhd])$")
_TIMEFRAME_UNIT_MS = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
_MAX_TIMEFRAME_MS = 86_400_000  # 1d


def _validated_timeframe(value: str) -> str:
    """Проверить строку таймфрейма (общая для бэктест- и live-конфигов).

    Raises:
        ValueError: если таймфрейм — не ``<count><m|h|d>`` в стиле ccxt
            длительностью до 1d.
    """
    match = _TIMEFRAME_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(
            f"timeframe must be '<count><m|h|d>' in ccxt style, "
            f"e.g. '15m', '4h', '1d'; got {value!r}"
        )
    count, unit = int(match.group(1)), match.group(2)
    if count * _TIMEFRAME_UNIT_MS[unit] > _MAX_TIMEFRAME_MS:
        raise ValueError(f"timeframe must not exceed 1d; got {value!r}")
    return value


class BacktestConfig(BaseModel):
    """Конфигурация одного запуска бэктеста.

    Attributes:
        symbol: торговая пара в формате ccxt, напр. ``"BTC/USDT"``.
        timeframe: таймфрейм свечи, напр. ``"15m"``, ``"4h"``, ``"1d"``.
        start: дата начала бэктеста, ``YYYY-MM-DD`` (включительно).
        end: дата конца бэктеста, ``YYYY-MM-DD`` (не включительно); ``None``
            означает «до последней доступной свечи».
        start_cash: начальный капитал в котируемой валюте.
        fee_rate: taker-комиссия биржи за одну сторону сделки, доля (0.001 = 0.1%).
        slippage_bps: проскальзывание исполнения в базисных пунктах (5 bps = 0.05%).
        position_size_pct: доля капитала, выделяемая под новую позицию.
        quantity_precision: число десятичных знаков, до которых округляется вниз
            объём входа.
        min_notional: минимальная стоимость ордера в котируемой валюте.
        strategy: имя плагина стратегии, зарегистрированного в ``trading_bot.strategy``.
        strategy_params: свободные параметры, передаваемые плагину стратегии.
    """

    symbol: str = "BTC/USDT"
    timeframe: str = "4h"
    start: str
    end: str | None = None
    start_cash: float = Field(default=10_000.0, gt=0.0)
    fee_rate: float = Field(default=0.001, ge=0.0, le=0.1)
    slippage_bps: float = Field(default=5.0, ge=0.0)
    position_size_pct: float = Field(default=0.95, gt=0.0, le=1.0)
    quantity_precision: int = Field(default=6, ge=0, le=10)
    min_notional: float = Field(default=5.0, ge=0.0)
    strategy: str = "sma_cross"
    strategy_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timeframe")
    @classmethod
    def _validate_timeframe(cls, value: str) -> str:
        """Проверить, что таймфрейм — ``<count><m|h|d>`` в стиле ccxt длительностью до 1d."""
        return _validated_timeframe(value)

    @field_validator("start", "end")
    @classmethod
    def _validate_iso_date(cls, value: str | None) -> str | None:
        """Проверить, что поля дат имеют формат ``YYYY-MM-DD``."""
        if value is not None:
            date.fromisoformat(value)
        return value

    @model_validator(mode="after")
    def _validate_period(self) -> BacktestConfig:
        """Проверить, что дата конца не раньше даты начала."""
        if self.end is not None and self.end < self.start:
            raise ValueError(f"end ({self.end}) must not be before start ({self.start})")
        return self


def load_config(path: Path | str) -> BacktestConfig:
    """Загрузить :class:`BacktestConfig` из YAML-файла."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config file {path} must contain a YAML mapping")
    return BacktestConfig.model_validate(raw)


class LiveConfig(BaseModel):
    """Конфигурация live/paper-раннера (``config/live.yaml``).

    Attributes:
        symbol: торговая пара в формате ccxt, напр. ``"BTC/USDT"``.
        timeframe: таймфрейм закрытых свечей, напр. ``"15m"``, ``"4h"``, ``"1d"``.
        strategy: имя плагина стратегии из ``trading_bot.strategy``.
        strategy_params: параметры стратегии; фиксируются в state при старте.
        mode: режим исполнения — ``"paper"`` (симуляция по рыночной цене,
            ключи не нужны) или ``"testnet"`` (реальные ордера на bybit testnet).
        poll_seconds: период опроса биржи в основном цикле, секунды.
        fee_rate: taker-комиссия за сторону сделки, доля (используется в paper).
        slippage_bps: проскальзывание в базисных пунктах (используется в paper).
        position_size_pct: доля капитала под новую позицию.
        start_cash: стартовый капитал paper-режима в котируемой валюте.
        quantity_precision: число знаков, до которых округляется вниз объём входа.
        min_notional: минимальная стоимость ордера в котируемой валюте.
        data_root: корень Parquet-хранилища live-свечей (отдельно от research).
        state_path: путь к JSON-файлу персистентного состояния раннера.
        kill_switch_path: путь файла-STOP-свитча; пока файл существует, любые
            новые ордера запрещены (тикеры/свечи продолжают обрабатываться).
        pause_switch_path: путь файла-PAUSE-свитча; пока файл существует,
            запрещены только новые входы — сигнальные выходы и защитные
            стоп/тейк исполняются как обычно.
        log_file: путь к rotating-логу раннера.
        telegram_bot_token: токен Telegram-бота для уведомлений; ``None`` —
            взять из env ``TELEGRAM_BOT_TOKEN`` (значения не логируются никогда).
        telegram_chat_id: chat id получателя уведомлений; ``None`` — взять из
            env ``TELEGRAM_CHAT_ID``.
        heartbeat_hours: период heartbeat-дайджеста в Telegram, часы;
            ``0`` — дайджест выключен.
        error_throttle_minutes: минимальный интервал между уведомлениями
            категории «error» (сетевые/данные сбои), минуты.
    """

    symbol: str = "BTC/USDT"
    timeframe: str = "4h"
    strategy: str = "donchian_trend"
    # Параметры из устойчивого ядра walk-forward исследования donchian_trend;
    # в MVP переоптимизация в рантайме не делается — смена параметров ручная
    # (правка конфига, перезапуск с чистым state).
    strategy_params: dict[str, Any] = Field(
        default_factory=lambda: {
            "entry_period": 40,
            "exit_period": 10,
            "atr_period": 14,
            "atr_mult": 2.0,
            "trend_period": 100,
        }
    )
    mode: Literal["paper", "testnet"] = "paper"
    poll_seconds: int = Field(default=60, gt=0)
    fee_rate: float = Field(default=0.001, ge=0.0, le=0.1)
    slippage_bps: float = Field(default=5.0, ge=0.0)
    position_size_pct: float = Field(default=0.95, gt=0.0, le=1.0)
    start_cash: float = Field(default=10_000.0, gt=0.0)
    quantity_precision: int = Field(default=6, ge=0, le=10)
    min_notional: float = Field(default=5.0, ge=0.0)
    data_root: str = "data/live"
    state_path: str = "data/live/state.json"
    kill_switch_path: str = "data/live/STOP"
    pause_switch_path: str = "data/live/PAUSE"
    log_file: str = "logs/live.log"
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    heartbeat_hours: float = Field(default=24.0, ge=0.0)
    error_throttle_minutes: int = Field(default=60, ge=0)

    @field_validator("timeframe")
    @classmethod
    def _validate_timeframe(cls, value: str) -> str:
        """Проверить, что таймфрейм — ``<count><m|h|d>`` в стиле ccxt длительностью до 1d."""
        return _validated_timeframe(value)

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, value: str) -> str:
        """Проверить, что символ — непустая пара вида ``BASE/QUOTE``."""
        base, slash, quote = value.partition("/")
        if not slash or not base or not quote:
            raise ValueError(
                f"symbol must be a ccxt pair like 'BTC/USDT', got {value!r}"
            )
        return value

    @field_validator("strategy")
    @classmethod
    def _validate_strategy(cls, value: str) -> str:
        """Проверить, что имя стратегии непустое (существование проверит реестр)."""
        if not value.strip():
            raise ValueError("strategy must be a non-empty name")
        return value

    @model_validator(mode="after")
    def _validate_switch_paths(self) -> LiveConfig:
        """Проверить, что пути state/STOP/PAUSE не совпадают друг с другом.

        Совпадающие пути делают свитчи бессмысленными или опасными: файл,
        блокирующий только входы (PAUSE), не должен оказаться тем же файлом,
        что полный STOP, а оба — тем же файлом, что state (запись состояния
        стёрла бы свитч и наоборот).

        Raises:
            ValueError: если любые два из путей ``state_path``,
                ``kill_switch_path`` и ``pause_switch_path`` совпадают.
        """
        paths = {
            "state_path": Path(self.state_path),
            "kill_switch_path": Path(self.kill_switch_path),
            "pause_switch_path": Path(self.pause_switch_path),
        }
        seen: dict[Path, str] = {}
        collisions: list[str] = []
        for name, path in paths.items():
            first = seen.get(path)
            if first is not None:
                collisions.append(f"{first} == {name} ({path})")
            else:
                seen[path] = name
        if collisions:
            raise ValueError(
                "live switch paths must be distinct: " + "; ".join(collisions)
            )
        return self


def load_live_config(path: Path | str) -> LiveConfig:
    """Загрузить :class:`LiveConfig` из YAML-файла."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config file {path} must contain a YAML mapping")
    return LiveConfig.model_validate(raw)
