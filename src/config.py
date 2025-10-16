from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import os

from dotenv import load_dotenv

# --- Directories ---
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze" / "1min"
SILVER_DIR = DATA_DIR / "silver" / "1min"
MODELS_DIR = REPO_ROOT / "models"
DUCKDB_PATH = DATA_DIR / "warehouse.duckdb"  # used for ad-hoc queries

# Ensure base dirs exist
for p in (BRONZE_DIR, SILVER_DIR, MODELS_DIR, DATA_DIR):
    p.mkdir(parents=True, exist_ok=True)

# --- Defaults ---
DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]

# FMP 1-minute historical FX bars
FMP_1MIN_URL_TEMPLATE = "https://financialmodelingprep.com/api/v3/historical-chart/1min/{symbol}?apikey={api_key}"

# Minimal interval between HTTP calls, in seconds (UI can override)
DEFAULT_MIN_CALL_INTERVAL_SECONDS = 0.25

# Preferred storage/display timezone (as requested): America/New_York (handles EST/EDT)
UTC = timezone.utc
ET = ZoneInfo("America/New_York")


def load_env() -> None:
    """Load .env if present (idempotent)."""
    load_dotenv(override=False)


def get_fmp_api_key() -> str:
    """Return FMP API key from env (.env supported). Raises if missing."""
    load_env()
    api_key = os.getenv("FMP_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "FMP_API_KEY is not set. Create a .env file at repo root with 'FMP_API_KEY=...'."
        )
    return api_key


@dataclass(frozen=True)
class AppConfig:
    """Container for configurable knobs the UI may override."""
    min_call_interval_seconds: float = DEFAULT_MIN_CALL_INTERVAL_SECONDS