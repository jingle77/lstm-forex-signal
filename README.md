# lstm-forex-signal
A lightweight, local-first research tool for minute-level FX forecasting. It ingests 1-minute bars from Financial Modeling Prep (FMP), maintains a bronze to silver data lake in Parquet, trains per-pair LSTM models (lookback L minutes → forecast H minutes), and visualizes history vs. forecast in Streamlit.

---
## Features
- **Bronze/Silver Layering:** Raw appends to data/bronze/; deduped, time-sorted Parquet in data/silver/.
- **Fast storage & SQL:** Parquet + DuckDB for quick, columnar queries.
- **Per-pair LSTM:** Small, portable Keras model (LSTM(64) → Dense(H)), early stopping, train-only scaling.
- **Signals viz:** History (last L minutes) + H-minute forecast, per symbol. Two-column layout.
- **Model management:** Saved under models/<SYMBOL>/lstm_L{L}_H{H}_YYYYMMDD_HHMMSS/ with model.h5 + meta.json.
- **Timezone:** Timestamps stored and displayed in America/New_York (EST/EDT).

---
## Data Sources
- **1-Minute Interval Forex Chart:** https://financialmodelingprep.com/stable/historical-chart/1min?symbol=EURUSD&apikey=<API_KEY>
- **Important Note:** Access may vary depending on API access

---
## Setup
**Prereqs:** Python 3.12 (Codespaces/Linux OK), FMP API key.

1. Create venv & install deps
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Add you FMP API Key
```bash
echo "FMP_API_KEY=YOUR_KEY" > .env
```

3. Run the App
```bash
streamlit run app.py
```
