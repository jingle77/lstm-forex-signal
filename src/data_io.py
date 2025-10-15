from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import duckdb
import numpy as np
import pandas as pd
import requests

from .config import (
    BRONZE_DIR,
    SILVER_DIR,
    DUCKDB_PATH,
    FMP_1MIN_URL_TEMPLATE,
    UTC,
)


class SimpleRateLimiter:
    """Minimal sleep-based rate limiter enforcing a min interval between calls."""

    def __init__(self, min_interval_seconds: float = 0.25) -> None:
        self.min_interval = float(min_interval_seconds)
        self._last: float = 0.0

    def wait(self) -> None:
        now = time.time()
        if self._last > 0:
            elapsed = now - self._last
            to_sleep = self.min_interval - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)
        self._last = time.time()


# -----------------------------
# Fetching & Bronze appends
# -----------------------------

def _fetch_fmp_1m(symbol: str, api_key: str) -> pd.DataFrame:
    """Fetch 1m bars from FMP for a symbol.

    Returns a DataFrame with columns: ts (UTC tz-aware), open, high, low, close, volume, symbol.
    Sorted ascending by ts.
    """
    url = FMP_1MIN_URL_TEMPLATE.format(symbol=symbol, api_key=api_key)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        data = []

    # FMP returns most-recent-first; normalize
    records = []
    for row in data:
        try:
            ts = pd.to_datetime(row.get("date"), utc=True)
            if ts.tzinfo is None:
                ts = ts.tz_localize(timezone.utc)
            records.append({
                "ts": ts,
                "open": float(row.get("open")),
                "high": float(row.get("high")),
                "low": float(row.get("low")),
                "close": float(row.get("close")),
                "volume": float(row.get("volume", 0.0)),
                "symbol": symbol,
            })
        except Exception:
            # Skip malformed rows
            continue

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df

    df = df.sort_values("ts").reset_index(drop=True)
    return df


def _latest_ts_in_bronze(symbol: str) -> Optional[pd.Timestamp]:
    """Return the latest ts present in bronze for symbol (UTC tz-aware)."""
    sym_path = BRONZE_DIR / f"symbol={symbol}"
    if not sym_path.exists():
        return None
    con = duckdb.connect(str(DUCKDB_PATH))
    try:
        q = f"""
            SELECT max(ts) AS max_ts
            FROM read_parquet('{str(sym_path)}/date=*/**.parquet')
        """
        res = con.sql(q).df()
        val = res.loc[0, "max_ts"]
        if pd.isna(val):
            return None
        ts = pd.to_datetime(val, utc=True)
        if ts.tzinfo is None:
            ts = ts.tz_localize(UTC)
        return ts
    finally:
        con.close()


def _append_bronze(symbol: str, bars: pd.DataFrame) -> int:
    """Append bars into bronze partitioned as bronze/1min/symbol=SY/date=YYYY-MM-DD/*.parquet.
    Returns number of rows written.
    """
    if bars.empty:
        return 0

    # Ensure schema & UTC
    bars = bars.copy()
    bars["ts"] = pd.to_datetime(bars["ts"], utc=True)
    bars["symbol"] = symbol

    # Partition by UTC date
    bars["date"] = bars["ts"].dt.strftime("%Y-%m-%d")

    written = 0
    for date_str, chunk in bars.groupby("date"):
        out_dir = BRONZE_DIR / f"symbol={symbol}" / f"date={date_str}"
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"bars_{int(time.time()*1e6)}.parquet"
        out_path = out_dir / fname
        chunk.drop(columns=["date"], inplace=True)
        chunk.to_parquet(out_path, engine="pyarrow", index=False)
        written += len(chunk)
    return written


# -----------------------------
# Silver rebuild
# -----------------------------

def rebuild_silver_for_symbol(symbol: str, drop_existing: bool = False) -> Dict[str, int]:
    """Dedupe on (symbol, ts), sort by ts, write partitioned Parquet under silver/.
    Returns stats dict with before/after counts.
    """
    sym_bronze = BRONZE_DIR / f"symbol={symbol}"
    if not sym_bronze.exists():
        return {"before": 0, "after": 0}

    sym_silver = SILVER_DIR / f"symbol={symbol}"
    if drop_existing and sym_silver.exists():
        # remove silver partition directory entirely
        for p in sym_silver.rglob("*.parquet"):
            try:
                p.unlink()
            except Exception:
                pass
        for d in sorted(sym_silver.rglob("*"), reverse=True):
            if d.is_dir():
                try:
                    d.rmdir()
                except Exception:
                    pass

    con = duckdb.connect(str(DUCKDB_PATH))
    try:
        q = f"""
            WITH ds AS (
                SELECT *
                FROM read_parquet('{str(sym_bronze)}/date=*/**.parquet')
            ), dedup AS (
                SELECT *,
                       ROW_NUMBER() OVER (PARTITION BY symbol, ts ORDER BY ts) AS rn
                FROM ds
            )
            SELECT * FROM dedup WHERE rn = 1 ORDER BY ts
        """
        df = con.sql(q).df()
    finally:
        con.close()

    if df.empty:
        return {"before": 0, "after": 0}

    before = len(df)

    # Ensure UTC tz
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # Partition by date
    df["date"] = df["ts"].dt.strftime("%Y-%m-%d")

    # Overwrite silver partitions for dates present
    for date_str, chunk in df.groupby("date"):
        out_dir = SILVER_DIR / f"symbol={symbol}" / f"date={date_str}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # remove existing files in this date partition
        for old in out_dir.glob("*.parquet"):
            try:
                old.unlink()
            except Exception:
                pass
        chunk = chunk.drop(columns=["date"], errors="ignore")
        chunk.to_parquet(out_dir / "part-00000.parquet", engine="pyarrow", index=False)

    after = sum(len(g) for _, g in df.groupby("date"))
    return {"before": before, "after": after}


# -----------------------------
# Incremental update orchestration
# -----------------------------

def run_incremental_update(
    pairs: Iterable[str],
    limiter: SimpleRateLimiter,
    dedupe_to_silver: bool = True,
    drop_existing_silver: bool = False,
) -> Dict[str, Dict[str, int]]:
    """Fetch latest 1m bars for each pair, append new rows to bronze, optionally rebuild silver.

    Returns per-pair stats dict: {
      symbol: {"fetched": int, "new": int, "bronze_appended": int, "silver_before": int, "silver_after": int}
    }
    """
    from .config import get_fmp_api_key

    api_key = get_fmp_api_key()
    results: Dict[str, Dict[str, int]] = {}

    for symbol in pairs:
        limiter.wait()
        df = _fetch_fmp_1m(symbol, api_key)
        fetched = len(df)

        latest = _latest_ts_in_bronze(symbol)
        if latest is not None and not df.empty:
            df = df[df["ts"] > latest]
        new_rows = len(df)

        appended = _append_bronze(symbol, df) if new_rows > 0 else 0

        silver_before = silver_after = 0
        if dedupe_to_silver and (new_rows > 0 or drop_existing_silver):
            stats = rebuild_silver_for_symbol(symbol, drop_existing=drop_existing_silver)
            silver_before, silver_after = stats.get("before", 0), stats.get("after", 0)

        results[symbol] = {
            "fetched": fetched,
            "new": new_rows,
            "bronze_appended": appended,
            "silver_before": silver_before,
            "silver_after": silver_after,
        }

    return results