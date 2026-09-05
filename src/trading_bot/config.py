"""Pydantic-модели конфигурации и загрузка YAML."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# Таймфрейм в стиле ccxt: одна или несколько цифр, затем m/h/d (напр. 15m, 4h, 1d).
_TIMEFRAME_PATTERN = re.compile(r"^(\d+)([mhd])$")
_TIMEFRAME_UNIT_MS = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
_MAX_TIMEFRAME_MS = 86_400_000  # 1d


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
