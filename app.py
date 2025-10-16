from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    DEFAULT_PAIRS,
    BRONZE_DIR,
    SILVER_DIR,
    MODELS_DIR,
    UTC,
    AppConfig,
    get_fmp_api_key,
)
from src.data_io import SimpleRateLimiter, run_incremental_update
from src.storage import symbols_available, load_silver, coverage_report, gaps_report
from src.lstm import train_pair, save_lstm_bundle, forecast_pair

st.set_page_config(page_title="FX LSTM Research App", layout="wide")

# -------------------------
# Helpers
# -------------------------

def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=UTC)


def _ensure_api_key():
    try:
        return get_fmp_api_key()
    except Exception as e:
        st.warning("Set your FMP_API_KEY in a .env file at the repo root.")
        st.stop()


def _default_pairs() -> List[str]:
    syms = symbols_available()
    return syms if syms else DEFAULT_PAIRS


def _list_model_dirs() -> Dict[str, List[Path]]:
    """Return {symbol: [model_dir_paths_sorted_by_trained_at]}.
    Sorting uses meta.json['trained_at'] when available; otherwise folder mtime.
    """
    out: Dict[str, List[Path]] = {}
    if not MODELS_DIR.exists():
        return out

    def _trained_at_or_mtime(p: Path):
        meta_p = p / "meta.json"
        if meta_p.exists():
            try:
                with open(meta_p, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                ta = meta.get("trained_at")
                if ta:
                    return pd.to_datetime(ta, utc=True)
            except Exception:
                pass
        # fallback: filesystem mtime
        return pd.to_datetime(p.stat().st_mtime, unit="s", utc=True)

    for sym_dir in MODELS_DIR.iterdir():
        if not sym_dir.is_dir():
            continue
        symbol = sym_dir.name
        cand = [p for p in sym_dir.iterdir() if p.is_dir() and (p / "meta.json").exists()]
        # sort by trained_at (oldest→newest)
        cand.sort(key=_trained_at_or_mtime)
        out[symbol] = cand
    return out
    for sym_dir in MODELS_DIR.iterdir():
        if not sym_dir.is_dir():
            continue
        symbol = sym_dir.name
        cand = [p for p in sym_dir.iterdir() if p.is_dir() and (p / "meta.json").exists()]
        cand.sort()  # lexicographic; timestamp suffix ensures latest last
        out[symbol] = cand
    return out


def _latest_model_dir_for_symbol(symbol: str, model_dirs: Dict[str, List[Path]]) -> Optional[Path]:
    dirs = model_dirs.get(symbol, [])
    return dirs[-1] if dirs else None


def _load_meta(path: Path) -> Dict:
    with open(path / "meta.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _load_model(path: Path):
    import tensorflow as tf
    # Prefer model.h5; if not present, try model.keras; load without compiling to avoid deserialization issues
    h5 = path / "model.h5"
    kf = path / "model.keras"
    if h5.exists():
        return tf.keras.models.load_model(h5, compile=False)
    elif kf.exists():
        return tf.keras.models.load_model(kf, compile=False)
    else:
        raise FileNotFoundError(f"No model file found in {path}")


def _format_ts(ts: pd.Timestamp) -> str:
    ts = pd.to_datetime(ts)
    if ts.tzinfo is None:
        from src.config import ET
        ts = ts.tz_localize(ET)
    return ts.strftime("%Y-%m-%d %H:%M:%S %Z")


# -------------------------
# UI Tabs
# -------------------------

tab_data, tab_train, tab_signals, tab_about = st.tabs([
    "Data Import & Refresh",
    "Model Training (LSTM)",
    "Signals",
    "About / Help",
])

# =========================
# Data Import & Refresh
# =========================
with tab_data:
    st.header("Data Import & Refresh")
    st.caption("Incremental ingestion → bronze/, rebuild deduped silver/ with UTC timestamps.")

    pairs = st.multiselect(
        "Currency Pairs",
        options=_default_pairs(),
        default=_default_pairs(),
        key="data_pairs",
    )

    min_interval = st.number_input(
        "Min seconds between API calls",
        min_value=0.0,
        value=0.25,
        step=0.05,
        key="rate_min_interval",
    )

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("Fetch latest & Rebuild silver", key="fetch_latest"):
            _ensure_api_key()
            limiter = SimpleRateLimiter(min_interval_seconds=float(min_interval))
            with st.spinner("Fetching from FMP and updating bronze/silver..."):
                stats = run_incremental_update(pairs, limiter, dedupe_to_silver=True)
            st.success("Update complete.")
            st.dataframe(pd.DataFrame(stats).T)

    st.subheader("Coverage (silver)")
    cov = coverage_report()
    if cov.empty:
        st.info("No silver data yet. Fetch to populate.")
    else:
        st.dataframe(cov)

    st.subheader("Scan gaps (last 30 days)")
    symbol_for_gaps = st.selectbox("Symbol", options=pairs or _default_pairs(), key="gap_symbol")
    gaps = gaps_report(symbol_for_gaps, max_days=30)
    if gaps.empty:
        st.info("No gaps > 1 minute found in the last 30 days.")
    else:
        st.dataframe(gaps)

# =========================
# Model Training (LSTM)
# =========================
with tab_train:
    st.header("Model Training (LSTM)")

    pairs_train = st.multiselect(
        "Pairs to train",
        options=_default_pairs(),
        default=_default_pairs(),
        key="train_pairs",
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        L = st.number_input("Lookback window L (minutes)", min_value=30, max_value=240, value=60, step=5, key="train_L")
    with col2:
        H = st.number_input("Prediction horizon H (minutes)", min_value=5, max_value=60, value=15, step=5, key="train_H")
    with col3:
        epochs = st.number_input("Epochs", min_value=5, max_value=200, value=25, step=5, key="train_epochs")
    with col4:
        batch_size = st.number_input("Batch size", min_value=32, max_value=2048, value=256, step=32, key="train_batch")

    train_days = st.number_input("Use last N days of silver data", min_value=1, max_value=14, value=3, step=1, key="train_days")

    if st.button("Train models for selected pairs", key="btn_train"):
        all_metrics = []
        pbar = st.progress(0, text="Training...")
        total = len(pairs_train)
        for i, sym in enumerate(pairs_train, start=1):
            # Load recent silver bars
            end = _utc_now()
            start = end - pd.Timedelta(days=int(train_days))
            df = load_silver([sym], start=start, end=end)
            if df.empty or len(df) < (L + H + 10):
                st.info(f"{sym}: Not enough data to train (need > L+H). Skipping.")
                pbar.progress(i / total, text=f"{i}/{total}")
                continue

            try:
                res = train_pair(df_symbol=df[df["symbol"] == sym], L=int(L), H=int(H), epochs=int(epochs), batch_size=int(batch_size))
            except Exception as e:
                st.warning(f"{sym}: Training error: {e}")
                pbar.progress(i / total, text=f"{i}/{total}")
                continue

            # Save bundle
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_dir = MODELS_DIR / sym / f"lstm_L{int(L)}_H{int(H)}_{timestamp}"
            save_lstm_bundle(res.model, out_dir, res.meta)

            row = {
                "symbol": sym,
                "L": int(L),
                "H": int(H),
                "val_mse": res.metrics["val_mse"],
                "val_mae": res.metrics["val_mae"],
                "model_dir": str(out_dir),
            }
            all_metrics.append(row)
            pbar.progress(i / total, text=f"{i}/{total}")

        if all_metrics:
            st.success("Training complete.")
            st.dataframe(pd.DataFrame(all_metrics))
        else:
            st.info("No models trained.")

# =========================
# Signals
# =========================
with tab_signals:
    st.header("Signals")

    # Optional refresh to rescan the models directory after training
    if st.button("Rescan models", key="sig_rescan"):
        st.rerun()

    model_dirs = _list_model_dirs()
    syms_available = sorted(model_dirs.keys())
    if not syms_available:
        st.info("No trained models found. Train some in the previous tab.")
    else:
        # Choose symbols to display
        syms_select = st.multiselect(
            "Select pairs",
            options=syms_available,
            default=syms_available,
            key="sig_pairs",
        )

        # Toggle: auto-pick latest model per symbol vs manual selection
        use_latest = st.checkbox(
            "Use latest trained model per symbol", value=True, key="sig_use_latest"
        )

        selected_models: Dict[str, Path] = {}
        for sym in syms_select:
            latest = _latest_model_dir_for_symbol(sym, model_dirs)
            choices = model_dirs.get(sym, [])

            if use_latest:
                if latest is not None:
                    st.caption(f"{sym}: using latest model `{latest.name}`")
                    selected_models[sym] = latest
                else:
                    st.info(f"{sym}: No models found.")
            else:
                if not choices:
                    st.info(f"{sym}: No models found.")
                    continue
                label_map = {str(p): p.name for p in choices}
                default_idx = choices.index(latest) if (latest and latest in choices) else 0
                chosen = st.selectbox(
                    f"Model for {sym}",
                    options=[str(p) for p in choices],
                    index=default_idx,
                    format_func=lambda s: label_map.get(s, s),
                    key=f"model_{sym}",
                )
                selected_models[sym] = Path(chosen)

        # Single forecast trigger button (unique key scoped to Signals tab)
        do_forecast = st.button("Generate forecasts", key="sig_btn_forecast_main")

        if do_forecast:
            cols = st.columns(2)
            for i, sym in enumerate(syms_select):
                with cols[i % 2]:
                    mdir = selected_models.get(sym)
                    if mdir is None:
                        st.info(f"{sym}: No model selected or available.")
                        continue

                    issues = []
                    if not (mdir / "meta.json").exists():
                        issues.append("Missing meta.json")
                        meta = {}
                    else:
                        meta = _load_meta(mdir)

                    if not issues:
                        L = int(meta.get("L"))
                        H = int(meta.get("H"))
                        scaler = meta.get("scaler", {})

                        # Load available silver and build last L window up to last available bar
                        df = load_silver([sym])
                        df = df[df["symbol"] == sym].sort_values("ts")
                        if df.empty or len(df) < L:
                            issues.append(f"Not enough data for lookback window L={L}")

                    if issues:
                        st.info(f"{sym}: " + "; ".join(issues) + ".")
                        continue

                    last_ts = df["ts"].max()
                    df_up_to_last = df[df["ts"] <= last_ts]
                    last_window = df_up_to_last.tail(L)["close"].values.reshape(-1, 1)

                    # Forecast
                    model = _load_model(mdir)
                    yhat = forecast_pair(model, last_window=last_window, H=H, scaler_info=scaler)

                    # Prepare chart data (default lookback = model's L minutes)
                    chart_lookback_start = last_ts - pd.Timedelta(minutes=int(L))
                    hist = df[df["ts"] >= chart_lookback_start][["ts", "close"]].copy()
                    hist.rename(columns={"close": "value"}, inplace=True)
                    hist["segment"] = "history"

                    fut_times = [last_ts + pd.Timedelta(minutes=i) for i in range(1, H + 1)]
                    fut = pd.DataFrame({"ts": fut_times, "value": yhat, "segment": "forecast"})

                    plot_df = pd.concat([hist, fut], ignore_index=True)
                    plot_df["value"] = plot_df["value"].astype(float)
                    plot_df = plot_df.sort_values("ts")

                    color_scale = alt.Scale(domain=["history", "forecast"], range=["#4C78A8", "#F58518"])
                    line = (
                        alt.Chart(plot_df)
                        .mark_line(size=2)
                        .encode(
                            x=alt.X("ts:T", title="Time (ET)"),
                            y=alt.Y("value:Q", title=f"{sym} Close", scale=alt.Scale(zero=False)),
                            color=alt.Color("segment:N", scale=color_scale, title="Segment"),
                            tooltip=[alt.Tooltip("ts:T", title="Time"), alt.Tooltip("value:Q", title="Price", format=".6f"), "segment:N"],
                        )
                    )
                    now_rule = alt.Chart(pd.DataFrame({"ts": [last_ts]})).mark_rule(strokeDash=[6, 4]).encode(x="ts:T")
                    chart = (line + now_rule).properties(width=900, height=300, title=f"{sym}: Lookback (L={L}) & {H}-min Forecast")

                    st.altair_chart(chart, use_container_width=True)

                    # Metrics table
                    latest_actual = float(hist["value"].iloc[-1]) if not hist.empty else float(df["close"].iloc[-1])
                    predicted_endpoint = float(yhat[-1]) if len(yhat) else np.nan

                    meta_metrics = meta.get("metrics", {})
                    tbl = pd.DataFrame([
                        {
                            "symbol": sym,
                            "L": L,
                            "H": H,
                            "latest_actual": latest_actual,
                            "predicted_endpoint": predicted_endpoint,
                            "val_mse": meta_metrics.get("val_mse"),
                            "val_mae": meta_metrics.get("val_mae"),
                            "model_dir": str(mdir),
                        }
                    ])
                    st.dataframe(tbl)

with tab_about:
    st.header("About / Help")
    st.markdown(
        """
        **Directories**

        - `data/bronze/1min/symbol=PAIR/date=YYYY-MM-DD/*.parquet` — raw incremental appends from FMP.
        - `data/silver/1min/symbol=PAIR/date=YYYY-MM-DD/*.parquet` — deduped, sorted per-symbol.
        - `models/PAIR/lstm_L{L}_H{H}_YYYYMMDD_HHMMSS/` — saved Keras model + `meta.json`.

        **.env**

        Create `.env` at the repo root with `FMP_API_KEY=...` (Financial Modeling Prep key).

        **L and H**

        - `L` (lookback window, minutes): how many past minutes feed the LSTM.
        - `H` (horizon, minutes): how many minutes ahead to forecast.

        All timestamps are stored and displayed in **UTC**. For debugging or local views, convert for display only.
        """
    )