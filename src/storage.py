from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import duckdb
import pandas as pd

from .config import SILVER_DIR, DUCKDB_PATH, ET


def _glob_for_symbol(base: Path, symbol: str) -> str:
    return str(base / f"symbol={symbol}" / "date=*/**.parquet")


def symbols_available() -> List[str]:
    """List symbols with any data in silver."""
    if not SILVER_DIR.exists():
        return []
    syms = []
    for p in SILVER_DIR.glob("symbol=*"):
        if p.is_dir():
            syms.append(p.name.split("=", 1)[1])
    return sorted(syms)


def load_silver(symbols: Iterable[str], start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Load silver bars for symbols.
    Returns DataFrame with columns: ts (tz-aware New York time), open, high, low, close, volume, symbol.
    Time filtering is applied in pandas to avoid tz mismatches.
    """
    symbols = list(symbols)
    if not symbols:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume", "symbol"])  # empty

    con = duckdb.connect(str(DUCKDB_PATH))
    try:
        selects = [f"SELECT * FROM read_parquet('{_glob_for_symbol(SILVER_DIR, s)}')" for s in symbols]
        unioned = " UNION ALL ".join(selects)
        q = f"SELECT * FROM ({unioned}) ORDER BY symbol, ts"
        df = con.sql(q).df()
    finally:
        con.close()

    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["ts"])  # preserve tz if present
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize(ET)
    else:
        df["ts"] = df["ts"].dt.tz_convert(ET)

    if start is not None:
        start = pd.to_datetime(start)
        start = start.tz_localize(ET) if start.tzinfo is None else start.tz_convert(ET)
        df = df[df["ts"] >= start]
    if end is not None:
        end = pd.to_datetime(end)
        end = end.tz_localize(ET) if end.tzinfo is None else end.tz_convert(ET)
        df = df[df["ts"] <= end]

    return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def coverage_report() -> pd.DataFrame:
    """Return first/last timestamp and count per symbol from silver (times in New York tz)."""
    syms = symbols_available()
    if not syms:
        return pd.DataFrame(columns=["symbol", "first_ts", "last_ts", "rows"])

    con = duckdb.connect(str(DUCKDB_PATH))
    try:
        selects = [
            f"SELECT '{s}' AS symbol, min(ts) AS first_ts, max(ts) AS last_ts, count(*) AS rows FROM read_parquet('{_glob_for_symbol(SILVER_DIR, s)}')"
            for s in syms
        ]
        q = " UNION ALL ".join(selects)
        df = con.sql(q).df()
    finally:
        con.close()

    if df.empty:
        return df
    df["first_ts"] = pd.to_datetime(df["first_ts"])
    df["last_ts"] = pd.to_datetime(df["last_ts"])
    df["first_ts"] = df["first_ts"].dt.tz_localize(ET) if df["first_ts"].dt.tz is None else df["first_ts"].dt.tz_convert(ET)
    df["last_ts"] = df["last_ts"].dt.tz_localize(ET) if df["last_ts"].dt.tz is None else df["last_ts"].dt.tz_convert(ET)
    return df.sort_values("symbol").reset_index(drop=True)

def gaps_report(symbol: str, max_days: int = 30) -> pd.DataFrame:
    """Scan for gaps (>1 minute between consecutive bars) over the last `max_days` days in New York time.
    Returns a DataFrame with columns: start_ts, end_ts, gap_minutes.
    Note: DST transitions can appear as large gaps by design when using local time.
    """
    end = pd.Timestamp.now(tz=ET)
    start = end - pd.Timedelta(days=int(max_days))
    df = load_silver([symbol])
    if not df.empty:
        df = df[(df["ts"] >= start) & (df["ts"] <= end)]
    if df.empty:
        return pd.DataFrame(columns=["start_ts", "end_ts", "gap_minutes"])  # empty

    df = df.sort_values("ts").reset_index(drop=True)
    deltas = df["ts"].diff().fillna(pd.Timedelta(0))
    gap_mask = deltas > pd.Timedelta(minutes=1)
    starts = df.loc[gap_mask, "ts"].reset_index(drop=True)
    prevs = df["ts"].shift(1)[gap_mask].reset_index(drop=True)

    out = pd.DataFrame({
        "start_ts": prevs,
        "end_ts": starts,
    }).dropna()
    if out.empty:
        return pd.DataFrame(columns=["start_ts", "end_ts", "gap_minutes"])  # none

    out["start_ts"] = pd.to_datetime(out["start_ts"])
    out["end_ts"] = pd.to_datetime(out["end_ts"])
    out["gap_minutes"] = ((out["end_ts"] - out["start_ts"]) / pd.Timedelta(minutes=1)).astype(int)
    return out.sort_values("start_ts").reset_index(drop=True)