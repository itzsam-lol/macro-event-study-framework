# Macro Event Impact Tracker

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Data Pipeline](https://img.shields.io/badge/Data-Pandas-orange?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Visualization](https://img.shields.io/badge/DataViz-Matplotlib-green?logo=plotly&logoColor=white)](https://matplotlib.org/)

A comprehensive study framework to track and analyze the impact of macroeconomic events on various asset classes. This tool provides quantitative insights into how markets react to surprises in FOMC, CPI, NFP, and PMI data.

## Key Findings

*   **FOMC Lead Signal**: The study identifies the FOMC rate surprise as the strongest predictive signal in the dataset ($R^2 = 0.050, \beta = 0.303$). This indicates a strong, economically meaningful link between policy surprises and next-day equity variance.
*   **Bond Reaction Pattern**: TLT (Long Bonds) exhibits a distinct "buy the rumor, sell the news" behavior, rallying on the event day (+0.15% T=0d) before consistently selling off in the following session (-0.29% T+1d).
*   **Volatility Compression**: Positive macro surprises lead to sustained VIX compression that intensifies over time, reaching its sharpest decline at the 1-week horizon (-1.63%), suggesting that "good news" keeps fear suppressed well beyond the event window.
*   **Regime-Dependent Resilience**: In High Volatility environments, FOMC decisions act as a reliable vol-compressor. When the market is already fearful, policy clarity effectively reduces uncertainty and risk premiums.

## Features
- Real-time data fetching from FRED and Yahoo Finance
- Cross-asset reaction analysis across multiple time horizons (T=0d, T+1d, T+2d, T+1w)
- Dynamic dashboard generation with 7 key analysis panels
- Statistical significance testing for event reactions
- Cached data management for efficient re-runs

## Tech Stack
- **Core**: Python 3.9+, `pandas`, `numpy`, `scipy`
- **Data**: `fredapi` (FRED Release Data), `yfinance` (Asset Prices)
- **Visualization**: `matplotlib` (Tailored Dark Design), `tabulate` (Terminal Metrics)

## Usage
1. Set your `FRED_API_KEY` environment variable.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the study: `python macro_tracker.py`.

Outputs and cached data are stored in `./output/` and `./data/cache/`.

## License
MIT

---
*Built with FRED + yfinance | github.com/itzsam-lol*
