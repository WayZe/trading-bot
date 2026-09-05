"""Тесты персистентного состояния live-раннера."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from trading_bot.config import LiveConfig
from trading_bot.live.execution import FillResult
from trading_bot.live.state import (
    STATE_VERSION,
    LiveState,
    PositionState,
    fresh_state,
    load_or_fresh_state,
)

TS = pd.Timestamp("2025-08-01T04:00:00Z")


def _fill(side: str, price: float, quantity: float = 2.0, fee: float = 0.24) -> FillResult:
    return FillResult(
        side=side, quantity=quantity, price=price, fee=fee, ts=TS, reason="test reason"
    )


class TestSaveLoad:
    def test_roundtrip_preserves_all_fields(self, tmp_path) -> None:
        state = LiveState(
            mode="paper",
            symbol="BTC/USDT",
            timeframe="4h",
            strategy="donchian",
            strategy_params={"entry_period": 2},
            position=PositionState(
                quantity=1.5,
                entry_price=101.0,
                entry_ts=TS.isoformat(),
                entry_reason="breakout",
            ),
            active_stop=95.0,
            active_tp=120.0,
            last_candle_ts=TS.isoformat(),
            equity=8_000.0,
            trades=[{"side": "buy", "price": 101.0, "quantity": 1.5}],
            needs_attention=False,
        )
        path = tmp_path / "nested" / "state.json"

        state.save(path)
        loaded = LiveState.load(path)

        assert loaded == state

    def test_missing_file_returns_none(self, tmp_path) -> None:
        assert LiveState.load(tmp_path / "absent.json") is None

    def test_corrupted_json_raises_instead_of_overwriting(self, tmp_path) -> None:
        path = tmp_path / "state.json"
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(ValueError, match="corrupted"):
            LiveState.load(path)
        # Битый файл не был перезаписан.
        assert path.read_text(encoding="utf-8") == "{not json"

    def test_non_object_json_raises(self, tmp_path) -> None:
        path = tmp_path / "state.json"
        path.write_text("[1, 2]", encoding="utf-8")

        with pytest.raises(ValueError, match="JSON object"):
            LiveState.load(path)

    def test_unsupported_version_raises(self, tmp_path) -> None:
        path = tmp_path / "state.json"
        path.write_text(json.dumps({"version": STATE_VERSION + 100}), encoding="utf-8")

        with pytest.raises(ValueError, match="schema version"):
            LiveState.load(path)

    def test_save_is_atomic_no_tmp_leftovers(self, tmp_path) -> None:
        path = tmp_path / "state.json"
        LiveState().save(path)
        LiveState().save(path)

        assert path.exists()
        assert not path.with_name(path.name + ".tmp").exists()


class TestApplyFill:
    def test_buy_opens_position_and_reduces_equity(self) -> None:
        state = LiveState(equity=10_000.0)

        state.apply_fill(_fill("buy", price=100.0, quantity=2.0, fee=0.2))

        assert state.position is not None
        assert state.position.quantity == 2.0
        assert state.position.entry_price == 100.0
        assert state.position.entry_ts == TS.isoformat()
        assert state.position.entry_reason == "test reason"
        assert state.equity == pytest.approx(10_000.0 - 200.0 - 0.2)
        assert state.trades == [
            {
                "side": "buy",
                "price": 100.0,
                "quantity": 2.0,
                "ts": TS.isoformat(),
                "reason": "test reason",
                "fee": 0.2,
            }
        ]

    def test_sell_closes_position_and_adds_proceeds(self) -> None:
        state = LiveState(equity=1_000.0)
        state.apply_fill(_fill("buy", price=100.0, quantity=2.0, fee=0.2))

        state.apply_fill(_fill("sell", price=110.0, quantity=2.0, fee=0.22))

        assert state.position is None
        # 1000 - (200 + 0.2) + (220 - 0.22) = 1019.58
        assert state.equity == pytest.approx(1_019.58)
        assert len(state.trades) == 2

    def test_pyramiding_is_rejected(self) -> None:
        state = LiveState()
        state.apply_fill(_fill("buy", price=100.0))

        with pytest.raises(RuntimeError, match="pyramiding"):
            state.apply_fill(_fill("buy", price=100.0))

    def test_sell_without_position_is_rejected(self) -> None:
        with pytest.raises(RuntimeError, match="no position"):
            LiveState().apply_fill(_fill("sell", price=100.0))

    def test_unknown_side_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="side"):
            LiveState().apply_fill(_fill("hold", price=100.0))


class TestTransitions:
    def test_stops_set_and_cleared(self) -> None:
        state = LiveState()
        state.set_stops(95.0, 120.0)
        assert (state.active_stop, state.active_tp) == (95.0, 120.0)
        state.clear_stops()
        assert state.active_stop is None and state.active_tp is None

    def test_last_candle_roundtrip(self) -> None:
        state = LiveState()
        assert state.last_candle_timestamp() is None
        state.set_last_candle(TS)
        assert state.last_candle_timestamp() == TS
        assert state.last_candle_ts == TS.isoformat()


class TestConfigConsistency:
    def test_matching_config_passes(self, tmp_path) -> None:
        config = LiveConfig(strategy_params={"a": 1})
        state = fresh_state(config)

        state.ensure_matches_config(config)  # не бросает

    def test_mismatched_config_raises_with_hint(self, tmp_path) -> None:
        config = LiveConfig()
        state = fresh_state(config)

        other = LiveConfig(strategy_params={"entry_period": 55})
        with pytest.raises(ValueError, match="move or delete the state file"):
            state.ensure_matches_config(other)


class TestFreshState:
    def test_paper_equity_is_start_cash(self) -> None:
        config = LiveConfig(mode="paper", start_cash=7_500.0)
        state = fresh_state(config)

        assert state.mode == "paper"
        assert state.equity == 7_500.0
        assert state.position is None
        assert state.strategy == "donchian_trend"
        assert state.strategy_params == config.strategy_params

    def test_testnet_equity_is_zero(self) -> None:
        state = fresh_state(LiveConfig(mode="testnet"))

        assert state.equity == 0.0

    def test_load_or_fresh_creates_fresh_on_missing_file(self, tmp_path) -> None:
        config = LiveConfig(mode="paper", start_cash=1_234.0)

        state = load_or_fresh_state(tmp_path / "state.json", config)

        assert state.equity == 1_234.0
