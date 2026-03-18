# Macro Event Impact Tracker: MarketPulse

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Data Pipeline](https://img.shields.io/badge/Data-Pandas-orange?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Visualization](https://img.shields.io/badge/DataViz-Matplotlib-green?logo=plotly&logoColor=white)](https://matplotlib.org/)

## Overview
**Macro Event Impact Tracker** is an analytical toolkit designed to cross-reference macroeconomic data surprises with multitype asset performance. The framework evaluates the reaction of SPY (Equities), EURUSD=X (FX), TLT (Rates), and ^VIX (Vol) to highly anticipated news releases like the CPI, NFP, ISM PMI, and FOMC Rate Decisions. 

This project demonstrates expertise in **Financial API Orchestration**, **Statistical Impact Modeling**, and **Production-Quality Data Visualization**.

---

## Core Features
- **Macro Surprise Engine (FRED API)**: Connects to the St. Louis Fed API to retrieve historic indicator expectations, actuals, and release times, transforming outputs into rolling-window normalized "Surprise Z-scores".
- **Dynamic Asset Scraper (yfinance)**: Intraday (1h, 4h) and Interday (1d, 1w) interval pricing data retrieved on-the-fly referencing event timestamps.
- **Cross-Asset Returns Heatmap**: Generates a conditional median performance surface analyzing the expected short-term directional skew of Assets vs. Positive Surprises, complete with T-Test significance highlighting.
- **Cumulative Alpha Backtest**: Simulates an "Event Day Only" SPY cumulative drift relative to a stochastic randomized "Non-Event Day" background model to highlight macro-driven market volatility.
- **Resilient Fallback Modes**: Implements a highly robust `DEMO_MODE` containing seed-generated market walks and synthetic macro distributions if live API keys/limits are exhausted.

---

## Tech Stack & Architecture

### Languages & Tools
- **Language**: Python 3.12+
- **Data Engineering**: Pandas, NumPy, SciPy
- **APIs**: yfinance, fredapi
- **Visualization**: Matplotlib, Seaborn

### Project Structure
```text
MarketPulse/
├── data/               
│   └── cache/          # Local CSV cache storage resolving heavy API traffic
├── output/             # Generated dashboards (e.g. macro_impact_dashboard.png)
├── macro_tracker.py    # Main script: Orchestrates fetching, merging, and plotting
├── requirements.txt    # Project dependencies
└── README.md           # This project documentation
```

---

## Installation & Setup

1. **Clone the Repository**
```bash
git clone https://github.com/yourname/MarketPulse.git
cd MarketPulse
```

2. **Configure Virtual Environment**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Install Dependencies**
```powershell
pip install -r requirements.txt
```

4. **Environment Variables (Optional)**
```powershell
# Required for live FRED queries. If omitted, the script automatically defaults to DEMO_MODE.
$env:FRED_API_KEY="your_fed_api_key_here"
```

---

## Running the Pipeline

### Generating the Dashboard
To execute the data-fetch protocols, align timeseries logic, and compile the final high-density, 6-panel dark mode dashboard:
```powershell
python macro_tracker.py
```

### Visual Output
The pipeline generates:
1. Console readout using `tabulate` for immediate CLI inspection of Hit Rates and Mean Returns.
2. A publish-ready `output/macro_impact_dashboard.png` highlighting:
    - Event Timeline Strip-charts
    - Regression Confidence Scatters
    - Reaction Distributions
    - Volatility Regime Factor Bars
    - A custom 5-column stylistic Matplotlib Summary Table

---

## Key Learnings & Future Scope
- **Data Synchronization**: Overcame challenges with harmonizing high-granularity intraday price timelines directly against disjointed periodic macroeconomic publication schedules.
- **Visual Grid Management**: Built sophisticated nested Matplotlib `GridSpec` layouts handling multi-axial rendering to synthesize large scopes of impact dimensions efficiently.
- **Future Improvements**:
  - Integration of tick-level order book depth data.
  - Expanding the indicator universe beyond US-Centric inputs (e.g. ECB Rate Decisions, BoJ Target Rates).
  - Implementation of an automated emailing / cron-job framework publishing weekly dashboard refreshes.
