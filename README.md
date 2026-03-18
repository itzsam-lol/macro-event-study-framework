# Macro Event Impact Tracker

An institutional-grade Python framework for studying cross-asset reactions to macroeconomic event surprises (FOMC, CPI, NFP, PMI) using FRED and Yahoo Finance data.

## 📊 Key Findings (3-Year Study)

*   **FOMC Lead Signal**: The study identifies the FOMC rate surprise as the strongest predictive signal in the dataset ($R^2 = 0.136, \beta = 0.495$). Over 13% of next-day SPY variance is explained by the magnitude of the rate surprise alone.
*   **Bond Rally Pattern**: TLT (Long Bonds) exhibits a distinct "buy the rumor, sell the news" behavior, rallying on the event day (+0.15% T=0d) before consistently selling off in the following session (-0.29% T+1d).
*   **Volatility Compression**: Positive macro surprises lead to sustained VIX compression that intensifies over time, reaching its sharpest decline at the 1-week horizon (-1.63%), suggesting that "good news" keeps fear suppressed well beyond the event window.
*   **Regime-Dependent Resilience**: In High Volatility environments, FOMC decisions act as a reliable vol-compressor. While PMI surprises spike VIX by +3.5% in "Low Vol" regimes, FOMC's impact in "High Vol" is consistently negative, as policy clarity resolves market uncertainty.

## 🛠️ Tech Stack
*   **Core**: Python 3.9+, `pandas`, `numpy`, `scipy`
*   **Data**: `fredapi` (FRED Release Data), `yfinance` (Asset Prices)
*   **Visualization**: `matplotlib` (Tailored Dark Design), `tabulate` (Terminal Metrics)

## 🚀 Quick Start

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set API Key**:
Set your `FRED_API_KEY` as an environment variable.

3. **Run Study**:
```bash
python macro_tracker.py
```

Outputs are saved to `./output/macro_impact_dashboard.png`.

---
*Built with FRED + yfinance | github.com/itzsam-lol*
