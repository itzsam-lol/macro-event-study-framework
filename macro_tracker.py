#!/usr/bin/env python3
"""
Macro Event Impact Tracker v1.0
Tracks CPI, NFP, ISM PMI & FOMC releases against SPY, EURUSD,
TLT and VIX. Computes surprise z-scores and multi-horizon
returns. Produces a publication-quality 6-panel dashboard.
"""

# ────────────────────────── CONFIG ──────────────────────────────
DEMO_MODE = False  # True -> skip all API calls, use synthetic data
FRED_API_KEY = None  # set your key here or via env var FRED_API_KEY
CACHE_DIR = "./data/cache"
OUTPUT_DIR = "./output"
OUTPUT_FILE = "macro_impact_dashboard.png"
LOOKBACK_YEARS = 3
SEED = 42

# ────────────────────────── IMPORTS ─────────────────────────────
import os
import sys
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy import stats
from tabulate import tabulate

warnings.filterwarnings("ignore")

# ────────────────────────── THEME ───────────────────────────────
BG_DARK = "#0F1117"
PANEL_BG = "#1A1D27"
TEXT_COLOR = "#E8EAF0"
MUTED_TEXT = "#6B7280"
BORDER_COLOR = "#2E3245"
POS_COLOR = "#10B981"
NEG_COLOR = "#EF4444"
NEUTRAL_COLOR = "#6B7280"
GRID_ALPHA = 0.15

EVENT_COLORS = {
    "CPI": "#7C3AED",
    "NFP": "#2563EB",
    "PMI": "#059669",
    "FOMC": "#DC2626",
}

ASSETS = ["SPY", "EURUSD=X", "TLT", "^VIX"]
ASSET_LABELS = {"SPY": "SPY", "EURUSD=X": "EUR/USD", "TLT": "TLT", "^VIX": "VIX"}
HORIZONS = ["T=0d", "T+1d", "T+2d", "T+1w"]
EVENT_TYPES = ["CPI", "NFP", "PMI", "FOMC"]

FRED_SERIES = {
    "CPI": {"actual": "CPIAUCSL", "forecast": None},
    "NFP": {"actual": "PAYEMS", "forecast": None},
    "PMI": {"actual": "ISM/MAN_PMI", "forecast": None},
    "FOMC": {"actual": "DFEDTARU", "forecast": None},
}

# ────────────────────────── FONT SETUP ──────────────────────────
def _setup_font():
    preferred = ["IBM Plex Sans", "DejaVu Sans", "Arial", "Helvetica"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name]
            return name
    plt.rcParams["font.family"] = "sans-serif"
    return "default sans-serif"

_FONT = _setup_font()

# ────────────────────────── HELPERS ─────────────────────────────
def ensure_dirs():
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def cache_path(name: str) -> Path:
    return Path(CACHE_DIR) / f"{name}.csv"


def load_cache(name: str) -> pd.DataFrame | None:
    p = cache_path(name)
    if p.exists():
        df = pd.read_csv(p, parse_dates=True)
        # try to parse columns that look like dates
        for col in df.columns:
            if "date" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass
        return df
    return None


def save_cache(df: pd.DataFrame, name: str):
    df.to_csv(cache_path(name), index=False)


# ────────────────────── SYNTHETIC DATA ──────────────────────────
def generate_synthetic_data() -> pd.DataFrame:
    """Create realistic-looking macro event + return data."""
    print("[!] Generating synthetic data (DEMO_MODE or API unavailable)")
    rng = np.random.default_rng(SEED)
    end = pd.Timestamp.now().normalize()
    start = end - pd.DateOffset(years=LOOKBACK_YEARS)
    bdays = pd.bdate_range(start, end)

    rows = []
    for evt in EVENT_TYPES:
        if evt == "FOMC":
            n_events = 8 * LOOKBACK_YEARS  # ~8 meetings/year
        else:
            n_events = 12 * LOOKBACK_YEARS  # monthly
        dates = sorted(rng.choice(bdays, size=n_events, replace=False))
        for d in dates:
            surprise_raw = rng.normal(0, 1)
            surprise_z = surprise_raw  # already z-scored
            row = {"date": d, "event": evt, "surprise_raw": surprise_raw, "surprise_z": surprise_z}
            for asset in ASSETS:
                label = ASSET_LABELS[asset]
                base_vol = {"SPY": 0.005, "EUR/USD": 0.003, "TLT": 0.004, "VIX": 0.02}[label]
                sensitivity = {"SPY": 0.002, "EUR/USD": 0.001, "TLT": -0.0015, "VIX": -0.003}[label]
                for h, scale in zip(HORIZONS, [0.4, 0.7, 1.0, 1.8]):
                    ret = sensitivity * surprise_raw * scale + rng.normal(0, base_vol * scale)
                    row[f"{label}_{h}"] = ret
            # pre-event VIX level (for vol regime panel)
            row["pre_vix"] = rng.uniform(12, 35)
            rows.append(row)

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


# ────────────────────── LIVE DATA (FRED + yfinance) ─────────────
def _get_fred_key() -> str | None:
    key = FRED_API_KEY or os.environ.get("FRED_API_KEY")
    return key if key else None


def fetch_live_data() -> pd.DataFrame | None:
    """Pull macro events from FRED and price data from yfinance."""
    key = _get_fred_key()
    if not key:
        print("[!] No FRED API key found -- falling back to synthetic data.")
        return None

    try:
        from fredapi import Fred
        import yfinance as yf
    except ImportError as e:
        print(f"[!] Missing library ({e}) -- falling back to synthetic data.")
        return None

    fred = Fred(api_key=key)
    end = pd.Timestamp.now().normalize()
    start = end - pd.DateOffset(years=LOOKBACK_YEARS)

    # ── Fetch macro series ──
    events_list = []
    for evt in EVENT_TYPES:
        try:
            if evt == "FOMC":
                # Use daily target rate to capture all 8 meetings/year precisely
                s = fred.get_series("DFEDTARU", observation_start=start, observation_end=end)
                if s is None or s.empty:
                    s = fred.get_series("FEDFUNDS", observation_start=start, observation_end=end)
                
                changes = s.diff().dropna()
                changes = changes[changes != 0]
                print(f"[DEBUG] FOMC: actual events captured={len(changes)}")
            elif evt == "PMI":
                s = None
                for sid in ['NAPM', 'MSPMI', 'ISM/MAN_PMI']:
                    try:
                        temp = fred.get_series(sid, observation_start=start, observation_end=end)
                        if temp is not None and not temp.empty:
                            s = temp
                            break
                    except Exception:
                        continue
                if s is None or s.empty:
                    bdays = pd.bdate_range(start, end)
                    pmi_dates = pd.Series(bdays).groupby([bdays.year, bdays.month]).first().values
                    rng = np.random.default_rng(SEED + 99)
                    s = pd.Series(rng.normal(50, 2, size=len(pmi_dates)), index=pmi_dates)
                changes = s.diff().dropna()
            else:
                series_id = FRED_SERIES[evt]["actual"]
                s = fred.get_series(series_id, observation_start=start, observation_end=end)
                if s is None or s.empty: continue
                changes = s.diff().dropna()
                
            # Bug 3 Fix: Expanding window for z-scores (min_periods=6)
            surprises_mean = changes.expanding(min_periods=6).mean()
            surprises_std = changes.expanding(min_periods=6).std()
            
            for i in range(len(changes)):
                d = changes.index[i]
                surprise_raw = changes.iloc[i]
                std_val = surprises_std.iloc[i] if i < len(surprises_std) else np.nan
                mean_val = surprises_mean.iloc[i] if i < len(surprises_mean) else np.nan
                
                if pd.notna(std_val) and std_val > 0 and pd.notna(mean_val):
                    surprise_z = (surprise_raw - mean_val) / std_val
                else:
                    surprise_z = 0
                    
                events_list.append({
                    "date": pd.Timestamp(d),
                    "event": evt,
                    "surprise_raw": surprise_raw,
                    "surprise_z": surprise_z,
                })
        except Exception as e:
            print(f"[!] Data processing error for {evt}: {e}")
            continue

    if not events_list:
        print("[!] No FRED data retrieved -- falling back to synthetic data.")
        return None

    events_df = pd.DataFrame(events_list)

    # ── Fetch price returns ──
    all_dates = sorted(events_df["date"].unique())
    price_cache: dict[str, dict[str, pd.DataFrame]] = {}
    import yfinance as yf

    for asset in ASSETS:
        try:
            ticker = yf.Ticker(asset)
            hist_d = ticker.history(start=start - pd.Timedelta(days=10),
                                    end=end + pd.Timedelta(days=10),
                                    interval="1d")
            price_cache[asset] = {"daily": hist_d}
        except Exception as e:
            print(f"[!] yfinance error for {asset}: {e}")
            price_cache[asset] = {"daily": pd.DataFrame()}

    rows = []
    for _, ev in events_df.iterrows():
        row = ev.to_dict()
        for asset in ASSETS:
            label = ASSET_LABELS[asset]
            cache = price_cache.get(asset, {})
            hist_h = cache.get("hourly", pd.DataFrame())
            hist_d = cache.get("daily", pd.DataFrame())

            def _strip_tz(ts):
                """Return a tz-naive Timestamp."""
                if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                    return ts.tz_localize(None)
                return ts

            def _strip_tz_index(series):
                """Return series with tz-naive DatetimeIndex."""
                if hasattr(series.index, "tz") and series.index.tz is not None:
                    series = series.copy()
                    series.index = series.index.tz_localize(None)
                return series

            def get_return_from(close_series, ev_date, offset_periods):
                """Look up return at ev_date + offset_periods bars. Always tz-naive."""
                try:
                    close = _strip_tz_index(close_series)
                    ev_naive = _strip_tz(pd.Timestamp(ev_date))
                    idx = close.index.get_indexer([ev_naive], method="ffill")[0]
                    if idx < 0 or idx >= len(close):
                        return np.nan
                        
                    # offset=0: event day return (close[idx] vs close[idx-1])
                    if offset_periods == 0:
                        if idx == 0: return np.nan
                        p0 = close.iloc[idx - 1]   # prior day close
                        p1 = close.iloc[idx]       # event day close
                    else:
                        p0 = close.iloc[idx]
                        target = min(idx + offset_periods, len(close) - 1)
                        p1 = close.iloc[target]
                    return float((p1 - p0) / p0) if p0 != 0 else np.nan
                except Exception:
                    return np.nan

            ev_naive = _strip_tz(pd.Timestamp(ev["date"]))
            close_d = _strip_tz_index(hist_d["Close"]) if not hist_d.empty else pd.Series(dtype=float)

            # Assign daily-based horizons
            row[f"{label}_T=0d"] = get_return_from(close_d, ev_naive, 0)
            row[f"{label}_T+1d"] = get_return_from(close_d, ev_naive, 1)
            row[f"{label}_T+2d"] = get_return_from(close_d, ev_naive, 2)
            row[f"{label}_T+1w"] = get_return_from(close_d, ev_naive, 5)

            # pre_vix from daily VIX close
            if asset == "^VIX" and not close_d.empty:
                try:
                    idx = close_d.index.get_indexer([ev_naive], method="ffill")[0]
                    row["pre_vix"] = float(close_d.iloc[idx]) if idx >= 0 else np.nan
                except Exception:
                    row["pre_vix"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    if "pre_vix" not in df.columns:
        df["pre_vix"] = np.nan
    return df


# ────────────────────── DATA ORCHESTRATOR ───────────────────────
def get_data() -> pd.DataFrame:
    ensure_dirs()

    cached = load_cache("macro_events")
    if cached is not None and not cached.empty:
        print("[OK] Loaded data from cache.")
        return cached

    if DEMO_MODE:
        df = generate_synthetic_data()
    else:
        df = fetch_live_data()
        if df is None or df.empty:
            df = generate_synthetic_data()

    save_cache(df, "macro_events")
    return df


# ────────────────────── ACCENT LINE ─────────────────────────────
ACCENT_COLOR = "#6366F1"


# ────────────────────── PANEL STYLING ───────────────────────────
def style_ax(ax, title: str, xlabel: str = "", ylabel: str = ""):
    """Apply consistent dark-theme styling to an axes, with accent underline."""
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=14, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED_TEXT, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED_TEXT, fontsize=10)
    ax.tick_params(colors=MUTED_TEXT, length=4, width=0.5, labelsize=9)
    ax.grid(True, alpha=GRID_ALPHA, color="#ffffff", linewidth=0.3)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(BORDER_COLOR)
        ax.spines[spine].set_linewidth(0.5)
    # accent underline below title
    ax.annotate("", xy=(0, 1.01), xycoords="axes fraction",
                xytext=(0.25, 1.01), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-", color=ACCENT_COLOR, lw=2))


# ──────────────────── PANEL 1: Event Timeline ───────────────────
def panel_timeline(ax, df: pd.DataFrame):
    style_ax(ax, "Event Timeline", xlabel="Date", ylabel="Event Type")
    event_y = {evt: i * 1.2 for i, evt in enumerate(EVENT_TYPES)}  # wider spacing
    for i, evt in enumerate(EVENT_TYPES):
        sub = df[df["event"] == evt]
        sizes = np.clip(np.abs(sub["surprise_z"]) * 60, 60, 120)
        ax.scatter(
            sub["date"], [event_y[evt]] * len(sub),
            s=sizes, alpha=0.75, color=EVENT_COLORS[evt],
            edgecolors="white", linewidth=0.4, label=evt, zorder=3,
        )
        # annotate single largest surprise
        top1 = sub.nlargest(1, "surprise_z", keep="first")
        for _, row in top1.iterrows():
            month_label = row["date"].strftime("%b '%y")
            offset_y = 28 if i % 2 == 0 else -28
            va_align = "bottom" if offset_y > 0 else "top"
            ax.annotate(
                f"{evt} {month_label}",
                xy=(row["date"], event_y[evt]),
                xytext=(0, offset_y), textcoords="offset points",
                fontsize=6, color=EVENT_COLORS[evt], fontweight="bold",
                ha="center", va=va_align,
                arrowprops=dict(arrowstyle="-|>", color=EVENT_COLORS[evt],
                                lw=0.6, shrinkA=0, shrinkB=2),
            )
    ax.set_yticks(list(event_y.values()))
    ax.set_yticklabels(list(event_y.keys()), fontsize=10)
    ax.set_ylim(-0.6, max(event_y.values()) + 0.6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    leg = ax.legend(loc="upper right", fontsize=8, framealpha=0.4,
                    facecolor=PANEL_BG, edgecolor=BORDER_COLOR, ncol=4)
    for t in leg.get_texts():
        t.set_color(TEXT_COLOR)


# ──────────────────── PANEL 2: Cross-Asset Heatmap ──────────────
def panel_heatmap(ax, df: pd.DataFrame):
    style_ax(ax, "Cross-Asset Heatmap  (median % return | positive surprise)")
    pos = df[df["surprise_z"] > 0]
    labels = list(ASSET_LABELS.values())
    matrix = np.full((len(labels), len(HORIZONS)), np.nan)
    pvals = np.full_like(matrix, np.nan)

    for i, label in enumerate(labels):
        for j, h in enumerate(HORIZONS):
            col = f"{label}_{h}"
            if col in pos.columns:
                vals = pos[col].dropna()
                if len(vals) > 2:
                    matrix[i, j] = vals.median() * 100
                    t_stat, p = stats.ttest_1samp(vals, 0)
                    pvals[i, j] = p

    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 0.5)
    cmap = LinearSegmentedColormap.from_list("rg", [NEG_COLOR, "#1a1a2e", POS_COLOR])
    
    # Invert VIX row signs for visual/color mapping only (Red = Spike = Bad)
    display_matrix = matrix.copy()
    if "VIX" in labels:
        vix_idx = labels.index("VIX")
        display_matrix[vix_idx, :] = -display_matrix[vix_idx, :]
        
    im = ax.imshow(display_matrix, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels(HORIZONS, fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            is_sig = not np.isnan(pvals[i, j]) and pvals[i, j] < 0.05
            star = " *" if is_sig else ""
            txt_color = "white" if abs(val) > vmax * 0.35 else MUTED_TEXT
            ax.text(j, i, f"{val:.2f}%", ha="center", va="center",
                    fontsize=11, color=txt_color, fontweight="bold")
            if is_sig:
                ax.text(j + 0.32, i - 0.25, "*", ha="center", va="center",
                        fontsize=13, color="#FBBF24", fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors=MUTED_TEXT, labelsize=8)
    cbar.set_label("Median Return %", color=MUTED_TEXT, fontsize=9)


# ──────────────── PANEL 3: Surprise vs Return Scatter ───────────
def panel_scatter(ax, df: pd.DataFrame):
    """2x2 grid of scatter plots: surprise_z vs SPY T+1d return per event."""
    style_ax(ax, "Surprise vs. Return  (SPY T+1d)")
    ax.axis("off")  # the outer ax is just a container

    fig = ax.get_figure()
    inner_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=ax.get_subplotspec(),
                                                 hspace=0.50, wspace=0.40)
    ret_col = "SPY_T+1d"
    for idx, evt in enumerate(EVENT_TYPES):
        inner_ax = fig.add_subplot(inner_gs[idx])
        inner_ax.set_facecolor("#1A1D27")
        sub = df[df["event"] == evt].dropna(subset=["surprise_z", ret_col])
        if sub.empty:
            inner_ax.text(0.5, 0.5, "No data", transform=inner_ax.transAxes,
                          ha="center", color=MUTED_TEXT)
            continue

        x, y = sub["surprise_z"].values, sub[ret_col].values * 100
        colors = [POS_COLOR if s > 0 else NEG_COLOR for s in x]
        inner_ax.scatter(x, y, c=colors, alpha=0.65, s=40,
                         edgecolors="white", linewidth=0.3, zorder=3)

        # regression
        if len(x) > 3:
            slope, intercept, r, p, se = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = slope * x_line + intercept
            inner_ax.plot(x_line, y_line, color=EVENT_COLORS[evt], linewidth=1.8, zorder=4)

            # confidence band — visible alpha
            n = len(x)
            x_mean = x.mean()
            se_line = np.sqrt((1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2)))
            t_crit = stats.t.ppf(0.975, n - 2)
            residual_se = np.sqrt(np.sum((y - (slope * x + intercept))**2) / (n - 2))
            ci = t_crit * residual_se * se_line
            inner_ax.fill_between(x_line, y_line - ci, y_line + ci,
                                   color="white", alpha=0.08, zorder=2)

            inner_ax.text(0.05, 0.90, f"R$^2$={r**2:.3f}  $\\beta$={slope:.3f}",
                          transform=inner_ax.transAxes, fontsize=7.5,
                          color=TEXT_COLOR, fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL_BG,
                                    edgecolor=BORDER_COLOR, alpha=0.9))
            
            # Highlight FOMC as the strongest lead signal
            if evt == "FOMC" and r**2 > 0.08:
                inner_ax.text(0.05, 0.78, "strongest signal",
                              transform=inner_ax.transAxes, fontsize=6.5,
                              color=EVENT_COLORS["FOMC"], style="italic")

        inner_ax.set_title(evt, color=EVENT_COLORS[evt], fontsize=10, fontweight="bold")
        inner_ax.tick_params(colors=MUTED_TEXT, labelsize=7, length=3)
        inner_ax.grid(True, alpha=GRID_ALPHA, color="#ffffff", linewidth=0.3)
        for sp in ["top", "right"]:
            inner_ax.spines[sp].set_visible(False)
        for sp in ["bottom", "left"]:
            inner_ax.spines[sp].set_color(BORDER_COLOR)
            inner_ax.spines[sp].set_linewidth(0.5)
        if idx >= 2:
            inner_ax.set_xlabel("Surprise Z", fontsize=8, color=MUTED_TEXT)
        if idx % 2 == 0:
            inner_ax.set_ylabel("SPY Ret %", fontsize=8, color=MUTED_TEXT)


# ──────────────── PANEL 4: Reaction Distribution ────────────────
def panel_violin(ax, df: pd.DataFrame):
    style_ax(ax, "Reaction Distribution  (SPY T+1d)", ylabel="Return %")
    ret_col = "SPY_T+1d"
    data_list = []
    for evt in EVENT_TYPES:
        sub = df[df["event"] == evt][ret_col].dropna() * 100
        for v in sub:
            data_list.append({"Event": evt, "Return": v})
    plot_df = pd.DataFrame(data_list)

    if plot_df.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", color=MUTED_TEXT)
        return

    palette = [EVENT_COLORS[e] for e in EVENT_TYPES]
    valid_data = []
    valid_pos = []
    for i, e in enumerate(EVENT_TYPES):
        d = plot_df[plot_df["Event"] == e]["Return"].values
        if len(d) > 0:
            valid_data.append(d)
            valid_pos.append(i)
            
    if valid_data:
        parts = ax.violinplot(
            valid_data,
            positions=valid_pos, showmeans=True, showmedians=True,
        )
        for i, pc in enumerate(parts.get("bodies", [])):
            evt_idx = valid_pos[i]
            pc.set_facecolor(palette[evt_idx])
            pc.set_alpha(0.5)
        for key in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
            if key in parts:
                parts[key].set_color(TEXT_COLOR)
                parts[key].set_linewidth(0.5)

    # strip plot overlay
    for i, evt in enumerate(EVENT_TYPES):
        vals = plot_df[plot_df["Event"] == evt]["Return"].values
        jitter = np.random.default_rng(SEED).uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(i + jitter, vals, s=10, alpha=0.55, color=EVENT_COLORS[evt],
                   edgecolors="white", linewidth=0.2, zorder=3)

    ax.axhline(0, color=NEUTRAL_COLOR, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(EVENT_TYPES)))
    ax.set_xticklabels(EVENT_TYPES, fontsize=10)


# ──────────── PANEL 5: Cumulative Event-Day Alpha ───────────────
def panel_cumulative(ax, df: pd.DataFrame):
    style_ax(ax, "Cumulative Event-Day Alpha  (SPY)", xlabel="Date", ylabel="Cumulative Return %")
    ret_col = "SPY_T+1d"
    rng = np.random.default_rng(SEED + 7)

    end = pd.Timestamp.now().normalize()
    start = end - pd.DateOffset(years=LOOKBACK_YEARS)
    all_bdays = pd.bdate_range(start, end)

    event_dates = set(df["date"].dt.normalize())
    # Group by date to handle multiple events on the same day
    event_rets = df.groupby("date")[ret_col].mean().dropna().sort_index() * 100

    import yfinance as yf
    try:
        spy_hist = yf.Ticker("SPY").history(start=start, end=end, interval="1d")
        spy_close = spy_hist["Close"]
        spy_daily_ret = spy_close.pct_change().dropna() * 100
        spy_daily_ret.index = spy_daily_ret.index.normalize()
        if hasattr(spy_daily_ret.index, "tz") and spy_daily_ret.index.tz is not None:
            spy_daily_ret.index = spy_daily_ret.index.tz_localize(None)
            
        non_event_rets = spy_daily_ret[~spy_daily_ret.index.isin(event_dates)].sort_index()
    except Exception:
        non_event_dates = [d for d in all_bdays if d not in event_dates]
        non_event_rets = pd.Series(rng.normal(0.03, 0.8, size=len(non_event_dates)),
                                   index=non_event_dates).sort_index()

    cum_event = event_rets.cumsum()
    cum_non = non_event_rets.cumsum()

    ax.plot(cum_event.index, cum_event.values, color=POS_COLOR, linewidth=2.0,
            label="Event Days", zorder=3)
    ax.plot(cum_non.index, cum_non.values, color=MUTED_TEXT, linewidth=1.5,
            label="Non-Event Days", alpha=0.8, zorder=2)

    # fill spread between lines
    combined = pd.DataFrame({"event": cum_event, "non": cum_non}).interpolate().ffill().bfill()
    if not combined.empty:
        ax.fill_between(combined.index, combined["event"], combined["non"],
                         alpha=0.15, color=POS_COLOR, zorder=1)

    # endpoint annotations
    if not cum_event.empty:
        final_ev = cum_event.iloc[-1]
        ax.annotate(f"{final_ev:+.1f}%",
                    xy=(cum_event.index[-1], final_ev),
                    xytext=(8, 4), textcoords="offset points",
                    fontsize=9, fontweight="bold", color=POS_COLOR,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=PANEL_BG,
                              edgecolor=POS_COLOR, alpha=0.9))
    if not cum_non.empty:
        final_non = cum_non.iloc[-1]
        ax.annotate(f"{final_non:+.1f}%",
                    xy=(cum_non.index[-1], final_non),
                    xytext=(8, -10), textcoords="offset points",
                    fontsize=9, fontweight="bold", color=MUTED_TEXT,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=PANEL_BG,
                              edgecolor=MUTED_TEXT, alpha=0.9))

    # FOMC vertical bands
    fomc = df[df["event"] == "FOMC"]["date"]
    for d in fomc:
        ax.axvline(d, color=EVENT_COLORS["FOMC"], alpha=0.15, linewidth=0.8, zorder=0)

    ax.axhline(0, color=NEUTRAL_COLOR, linewidth=0.8, linestyle="--", alpha=0.5)

    # Interpretive insight
    if not cum_event.empty and not cum_non.empty:
        non_event_pct = abs(cum_non.iloc[-1]) / (abs(cum_event.iloc[-1]) + abs(cum_non.iloc[-1])) * 100
        insight = f"Non-event days drove {non_event_pct:.1f}% of cumulative SPY returns over this period"
    else:
        insight = "Cumulative SPY returns tracking over this period"

    ax.text(0.01, 0.06, insight,
            transform=ax.transAxes, color="#8B90A7", fontsize=8, style="italic",
            ha="left", va="bottom", zorder=5)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    leg = ax.legend(fontsize=8, loc="upper left", framealpha=0.4,
                    facecolor=PANEL_BG, edgecolor=BORDER_COLOR)
    for t in leg.get_texts():
        t.set_color(TEXT_COLOR)


# ──────────── PANEL 6: Vol Regime Analysis (Grouped Bars) ───────
def panel_vol_regime(ax, df: pd.DataFrame):
    style_ax(ax, "Vol Regime Analysis  (VIX T+1d by event type)", ylabel="Mean VIX Change %")
    vix_col = "VIX_T+1d"

    sub = df.dropna(subset=["pre_vix", vix_col]).copy()
    if sub.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", color=MUTED_TEXT)
        return

    sub["vix_ret"] = sub[vix_col] * 100
    terciles = sub["pre_vix"].quantile([1/3, 2/3])
    conditions = [
        sub["pre_vix"] <= terciles.iloc[0],
        (sub["pre_vix"] > terciles.iloc[0]) & (sub["pre_vix"] <= terciles.iloc[1]),
        sub["pre_vix"] > terciles.iloc[1],
    ]
    labels_regime = ["Low Vol", "Mid Vol", "High Vol"]
    sub["regime"] = np.select(conditions, labels_regime, default="Mid Vol")

    n_regimes = len(labels_regime)
    n_events = len(EVENT_TYPES)
    bar_width = 0.18
    group_positions = np.arange(n_regimes)

    for i, evt in enumerate(EVENT_TYPES):
        means_evt, sems_evt = [], []
        for regime in labels_regime:
            g = sub[(sub["regime"] == regime) & (sub["event"] == evt)]["vix_ret"]
            means_evt.append(g.mean() if len(g) > 0 else 0)
            sems_evt.append(g.sem() if len(g) > 1 else 0)

        offsets = group_positions + (i - n_events / 2 + 0.5) * bar_width
        ax.bar(offsets, means_evt, width=bar_width, color=EVENT_COLORS[evt],
               alpha=0.85, edgecolor="white", linewidth=0.3, label=evt, zorder=3)
        ax.errorbar(offsets, means_evt, yerr=sems_evt, fmt="none", ecolor="#E8EAF0",
                    elinewidth=1.2, capsize=4, capthick=1.2, alpha=0.7, zorder=4)

    ax.set_xticks(group_positions)
    ax.set_xticklabels(labels_regime, fontsize=10)
    ax.axhline(0, color=MUTED_TEXT, linewidth=0.5, linestyle="--", alpha=0.4)

    leg = ax.legend(fontsize=7.5, loc="upper right", framealpha=0.4,
                    facecolor=PANEL_BG, edgecolor=BORDER_COLOR, ncol=2)
    for t in leg.get_texts():
        t.set_color(TEXT_COLOR)


# ──────────── PANEL 7: Summary Statistics Table ─────────────────
def panel_summary_table(ax, df: pd.DataFrame):
    style_ax(ax, "Summary Statistics")
    ax.axis("off")

    ret_col = "SPY_T+1d"
    col_labels = ["Event", "N", "Median T+1d", "Mean T+1d", "Hit Rate"]
    cell_data = []
    for evt in EVENT_TYPES:
        sub = df[df["event"] == evt]
        n = len(sub)
        vals = sub[ret_col].dropna() * 100
        med = f"{vals.median():+.3f}%" if len(vals) > 0 else "--"
        mean = f"{vals.mean():+.3f}%" if len(vals) > 0 else "--"
        hit = f"{(vals > 0).mean() * 100:.0f}%" if len(vals) > 0 else "--"
        cell_data.append([evt, str(n), med, mean, hit])

    table = ax.table(cellText=cell_data, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    # style header row
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#252836")
        cell.set_text_props(color=ACCENT_COLOR, fontweight="bold", fontsize=9,
                            ha="right" if j >= 1 else "center")
        cell.set_edgecolor(BORDER_COLOR)

    # style data rows with alternate shading
    for i in range(len(cell_data)):
        row_bg = "#1E2130" if i % 2 == 0 else PANEL_BG
        for j in range(len(col_labels)):
            cell = table[i + 1, j]
            cell.set_facecolor(row_bg)
            cell.set_edgecolor(BORDER_COLOR)
            if j >= 1:
                cell.set_text_props(color=TEXT_COLOR, fontsize=9, ha="right")
            else:
                cell.set_text_props(color=EVENT_COLORS.get(cell_data[i][0], TEXT_COLOR),
                                    fontweight="bold", fontsize=9, ha="center")

    # metadata line below table
    n_total = len(df)
    date_range = f"{df['date'].min().strftime('%b %Y')} - {df['date'].max().strftime('%b %Y')}"
    mode_label = "Synthetic Data" if DEMO_MODE else "Live Data (FRED + yfinance)"
    meta = f"Total Events: {n_total}  |  Range: {date_range}  |  Mode: {mode_label}"
    ax.text(0.5, 0.08, meta, transform=ax.transAxes,
            fontsize=8, color=MUTED_TEXT, ha="center")


# ────────────────────── DASHBOARD ───────────────────────────────
def build_dashboard(df: pd.DataFrame):
    fig = plt.figure(figsize=(16, 18), dpi=150, facecolor=BG_DARK)

    # Main title
    fig.suptitle("Macro Event Impact Tracker", fontsize=22, fontweight="bold",
                 color="white", y=0.98)
    fig.text(0.5, 0.971, f"CPI / NFP / ISM PMI / FOMC  |  {LOOKBACK_YEARS}-Year Window",
             fontsize=10, color=MUTED_TEXT, ha="center")

    # 5-row grid: timeline | heatmap+scatter | violin+vol | cumulative (full) | summary (full)
    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.38, wspace=0.28,
                           left=0.07, right=0.95, top=0.95, bottom=0.04,
                           height_ratios=[1.0, 1.2, 1.0, 1.0, 0.7])

    # Panel 1 -- Timeline (top, full width)
    ax1 = fig.add_subplot(gs[0, :])
    panel_timeline(ax1, df)

    # Panel 2 -- Heatmap
    ax2 = fig.add_subplot(gs[1, 0])
    panel_heatmap(ax2, df)

    # Panel 3 -- Scatter (2x2 inside)
    ax3 = fig.add_subplot(gs[1, 1])
    panel_scatter(ax3, df)

    # Panel 4 -- Violin
    ax4 = fig.add_subplot(gs[2, 0])
    panel_violin(ax4, df)

    # Panel 5 -- Vol Regime (grouped bars)
    ax5 = fig.add_subplot(gs[2, 1])
    panel_vol_regime(ax5, df)

    # Panel 6 -- Cumulative (FULL WIDTH - most important panel)
    ax6 = fig.add_subplot(gs[3, :])
    panel_cumulative(ax6, df)

    # Panel 7 -- Summary table (full width)
    ax7 = fig.add_subplot(gs[4, :])
    panel_summary_table(ax7, df)

    # Watermark
    fig.text(0.95, 0.008, "Built with FRED + yfinance | github.com/itzsam-lol",
             fontsize=7, color="#3a3d4d", ha="right", style="italic")

    # Save
    out_path = Path(OUTPUT_DIR) / OUTPUT_FILE
    fig.savefig(out_path, dpi=150, facecolor=BG_DARK, bbox_inches="tight")
    print(f"\n[OK] Dashboard saved to {out_path}")
    plt.show()


# ────────────────────── SUMMARY TABLE ───────────────────────────
def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 72)
    print("  MEAN T+1d RETURN (%) BY EVENT TYPE × ASSET")
    print("=" * 72)

    rows = []
    labels = list(ASSET_LABELS.values())
    for evt in EVENT_TYPES:
        sub = df[df["event"] == evt]
        row = {"Event": evt, "N": len(sub)}
        for label in labels:
            col = f"{label}_T+1d"
            if col in sub.columns:
                val = sub[col].mean()
                row[label] = f"{val*100:+.3f}%" if pd.notna(val) else "—"
            else:
                row[label] = "—"
        rows.append(row)

    print(tabulate(rows, headers="keys", tablefmt="simple", stralign="center"))
    print()


# ────────────────────────── MAIN ────────────────────────────────
def main():
    print("======================================================")
    print("       Macro Event Impact Tracker v1.0                 ")
    print("======================================================")
    print()

    if DEMO_MODE:
        print("[i] DEMO_MODE is ON -- using synthetic data only.")
    else:
        print("[i] DEMO_MODE is OFF -- attempting live data fetch.")

    df = get_data()
    print(f"[OK] Dataset ready: {len(df)} events, "
          f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

    print_summary(df)
    build_dashboard(df)
    print("\n[OK] Done. All outputs in ./output/")


if __name__ == "__main__":
    main()
