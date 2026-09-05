"""Общие фикстуры и хелперы для тестов."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

# 2025-08-01T00:00:00Z в миллисекундах от эпохи.
BASE_MS = int(datetime(2025, 8, 1, tzinfo=UTC).timestamp() * 1000)
HOUR_MS = 3_600_000


def make_candles(
    n: int, start_ms: int = BASE_MS, tf_ms: int = HOUR_MS, base_price: float = 100.0
) -> list[list]:
    """Сгенерировать ``n`` валидных сырых строк свечей ``[ts_ms, o, h, l, c, v]``."""
    rows: list[list] = []
    price = base_price
    for i in range(n):
        ts = start_ms + i * tf_ms
        open_ = price
        close = price + 1.0
        high = max(open_, close) + 2.0
        low = min(open_, close) - 2.0
        rows.append([ts, open_, high, low, close, 10.0])
        price = close
    return rows


def rows_to_df(rows: list[list]) -> pd.DataFrame:
    """Перевести сырые строки свечей во фрейм без сортировки и дедупликации."""
    df = pd.DataFrame(
        rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    if rows:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).astype(
            "datetime64[ms, UTC]"
        )
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype("float64")
    return df
