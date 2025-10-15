from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class TrainResult:
    model: keras.Model
    scaler_mean: float
    scaler_std: float
    metrics: Dict[str, float]
    meta: Dict


def make_supervised(df: pd.DataFrame, L: int, H: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding windows over a *single symbol* df with columns including 'ts' and 'close'.

    Returns (X_idx, X_all, y_all) where X_idx are the end indices of each window (to allow time-based splitting),
    X_all shape (n_samples, L, 1), y_all shape (n_samples, H) with H-step future closes.
    Note: scaling should be applied outside using train-only stats to avoid leakage.
    """
    df = df.sort_values("ts").reset_index(drop=True)
    series = df["close"].astype(float).values
    n = len(series)
    N = n - L - H + 1
    if N <= 0:
        return np.array([]), np.empty((0, L, 1)), np.empty((0, H))

    X_idx = np.arange(L, L + N)  # window end index for each sample
    X = np.zeros((N, L, 1), dtype=np.float32)
    y = np.zeros((N, H), dtype=np.float32)
    for i in range(N):
        start = i
        end = i + L
        X[i, :, 0] = series[start:end]
        y[i, :] = series[end:end + H]
    return X_idx, X, y


def build_lstm_model(L: int, num_features: int, H: int) -> keras.Model:
    """Keras Sequential: (optional Masking) → LSTM(64) → Dense(H)."""
    model = keras.Sequential([
        layers.Input(shape=(L, num_features)),
        # layers.Masking(mask_value=0.0),  # optional, generally not needed if no padding
        layers.LSTM(64, return_sequences=False),
        layers.Dense(H),
    ])
    model.compile(optimizer=keras.optimizers.Adam(), loss="mse", metrics=["mae"])
    return model


def train_pair(
    df_symbol: pd.DataFrame,
    L: int,
    H: int,
    epochs: int = 25,
    batch_size: int = 256,
    val_frac: float = 0.2,
    random_state: int = 42,
) -> TrainResult:
    """Chronological split, standardize on train-only, train LSTM to predict H-step close path."""
    rng = np.random.default_rng(random_state)

    # Build supervised dataset
    X_idx, X_raw, y_raw = make_supervised(df_symbol, L, H)
    if X_raw.shape[0] == 0:
        raise ValueError("Not enough data to build supervised dataset. Increase history window.")

    # Chronological split by sample end index
    n_samples = X_raw.shape[0]
    split_idx = int((1 - val_frac) * n_samples)

    # Fit scaler on train-only raw closes (flatten windows)
    closes_series = df_symbol.sort_values("ts")["close"].astype(float).values
    train_series_end = L + split_idx  # index in the original series where train samples end
    scaler = StandardScaler()
    scaler.fit(closes_series[:train_series_end].reshape(-1, 1))

    # Apply scaler to all X and y
    def scale_arr(arr: np.ndarray) -> np.ndarray:
        shp = arr.shape
        flat = arr.reshape(-1, 1)
        scaled = scaler.transform(flat).reshape(shp)
        return scaled

    X_scaled = scale_arr(X_raw)
    y_scaled = scale_arr(y_raw)

    X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]

    model = build_lstm_model(L=L, num_features=1, H=H)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    hist = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[es],
    )

    # Evaluate on validation set using restored best weights
    val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)

    # Date span meta
    dmin = pd.to_datetime(df_symbol["ts"].min(), utc=True)
    dmax = pd.to_datetime(df_symbol["ts"].max(), utc=True)

    meta = {
        "symbol": str(df_symbol["symbol"].iloc[0]) if "symbol" in df_symbol.columns and len(df_symbol) else "",
        "L": int(L),
        "H": int(H),
        "features": ["close"],
        "scaler": {"mean": float(scaler.mean_[0]), "std": float(scaler.scale_[0])},
        "metrics": {"val_mse": float(val_loss), "val_mae": float(val_mae)},
        "date_span": {"start": dmin.isoformat(), "end": dmax.isoformat()},
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    return TrainResult(
        model=model,
        scaler_mean=float(scaler.mean_[0]),
        scaler_std=float(scaler.scale_[0]),
        metrics={"val_mse": float(val_loss), "val_mae": float(val_mae)},
        meta=meta,
    )


def save_lstm_bundle(model: keras.Model, out_dir: Path, meta: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save model as H5 for portability
    model_path = out_dir / "model.h5"
    # Save without optimizer to keep bundle lean and avoid deserializing training state on load
    model.save(model_path, include_optimizer=False)
    # Save metadata
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def forecast_pair(model: keras.Model, last_window: np.ndarray, H: int, scaler_info: Dict[str, float]) -> np.ndarray:
    """Forecast H steps from the last Lx1 window of *raw* closes using the trained model that outputs H.

    last_window: shape (L, 1) in *original* price scale.
    Returns np.ndarray length H in original price scale.
    """
    mean = float(scaler_info.get("mean"))
    std = float(scaler_info.get("std"))
    if std == 0:
        std = 1e-9

    # Scale window
    z = (last_window - mean) / std
    z = z.astype(np.float32)[None, ...]  # (1, L, 1)

    # Model outputs H scaled values; inverse transform
    y_scaled = model.predict(z, verbose=0)[0]  # (H,)
    y = (y_scaled * std) + mean
    return y.astype(float)