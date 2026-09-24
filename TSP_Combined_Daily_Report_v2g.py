"""
TSP Combined Daily Report (Optimized)
=====================================
Merges Daily Signal, Pillars 3 & 4, and Triple-MACD into a single email report.
Optimized for faster Yahoo Finance fetches and clean in-memory rendering.
"""

import os
import smtplib
import ssl
import urllib.request
import csv
import io
import warnings
from datetime import datetime, timedelta
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

warnings.filterwarnings("ignore")

import pandas as pd
import requests
import urllib3
import yfinance as yf

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# MACD charting check
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import mplfinance as mpf
    MACD_AVAILABLE = True
except ImportError as _macd_import_err:
    MACD_AVAILABLE = False
    _MACD_IMPORT_ERROR_MSG = str(_macd_import_err)

# =====================================================================
# CONFIGURATION
# =====================================================================
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "your_email@gmail.com")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD", "your_app_password_here")
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL", "your_email@gmail.com")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

PROXIES = {
    "C Fund": "IVV",
    "S Fund": "VXF",
    "I Fund": "ACWX",
    "G Fund": "BIL",
}

MA_WINDOWS = [5, 20, 60, 100, 200]
COMPOSITE_WEIGHTS = {5: 0.15, 20: 0.40, 60: 0.45}
COMPOSITE_MA_WINDOWS = list(COMPOSITE_WEIGHTS.keys())
MIN_MAS_PASSED_TO_QUALIFY = 1
CORRELATION_THRESHOLD = 0.70

TSP_FUNDS = ["G Fund", "C Fund", "S Fund", "I Fund"]

INTERNALS_TICKERS = {
    "VIX": "^VIX",
    "10-Year Treasury Yield": "^TNX",
    "WTI Crude Oil": "CL=F",
    "Brent Crude Oil": "BZ=F",
}

MACD_TICKERS = {
    "^GSPC": "C Fund (S&P 500)",
    "^DWCPF": "S Fund (Completion Index)",
    "ACWX": "I Fund (International Index)",
}

class UnverifiedSignatureSession(requests.Session):
    def __init__(self):
        super().__init__()
        self.verify = False
        self.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        })

custom_session = UnverifiedSignatureSession()

# =====================================================================
# SECTION 1: DAILY SIGNAL
# =====================================================================

def download_historical_data(years=2, ma_warmup_days=260, include_vix=False):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(years * 365 + ma_warmup_days * 1.5))
    
    tickers_to_fetch = list(PROXIES.values())
    if include_vix:
        tickers_to_fetch.append("^VIX")
        
    try:
        data = yf.download(tickers_to_fetch, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            close_prices = data["Close"]
        else:
            close_prices = data[["Close"]]
    except Exception as e:
        print(f"[-] Batch download failed, fallback to ticker loop: {e}")
        close_prices = pd.DataFrame()
        for t_sym in tickers_to_fetch:
            try:
                t = yf.Ticker(t_sym, session=custom_session)
                hist = t.history(start=start_date, end=end_date, auto_adjust=True)
                if not hist.empty:
                    close_prices[t_sym] = hist["Close"]
            except Exception as ex:
                print(f"[-] Failed to fetch {t_sym}: {ex}")

    prices = pd.DataFrame()
    for fund_name, ticker in PROXIES.items():
        if ticker in close_prices.columns:
            prices[fund_name] = close_prices[ticker]

    if include_vix and "^VIX" in close_prices.columns:
        prices["VIX"] = close_prices["^VIX"]

    idx = prices.index
    if getattr(idx, "tz", None) is not None:
        prices.index = idx.tz_localize(None).normalize()
    else:
        prices.index = idx.normalize()

    prices = prices.ffill().dropna()
    if prices.empty:
        raise ValueError("Could not retrieve price data for Section 1 (Daily Signal).")
    return prices

def compute_moving_averages(prices):
    df = prices.copy()
    for fund in PROXIES.keys():
        for ma in MA_WINDOWS:
            df[f"{fund}_{ma}MA"] = df[fund].rolling(window=ma).mean()
    return df.dropna()

def compute_regime_correlation(df, window=20):
    returns = df[["C Fund", "S Fund", "I Fund"]].pct_change()
    pairs = [("C Fund", "S Fund"), ("C Fund", "I Fund"), ("S Fund", "I Fund")]
    pair_corrs = pd.DataFrame(index=df.index)
    for a, b in pairs:
        pair_corrs[f"{a}_{b}"] = returns[a].rolling(window).corr(returns[b])
    return pair_corrs.mean(axis=1)

def classify_regime(avg_corr_value):
    if pd.isna(avg_corr_value):
        return "UNKNOWN", "#6c757d"
    if avg_corr_value >= CORRELATION_THRESHOLD:
        return "CORRELATED", "#0d6efd"
    return "DIVERGENT", "#fd7e14"

TRIPOD_TIER_DESCRIPTIONS = {
    "BULL_CALM": ("Bullish & Calm", "#1e7e34", "Price is comfortably above its 250-day trend line and volatility is low."),
    "BULL_SHAKY": ("Bullish but Shaky", "#b8860b", "Price is above trend line, but volatility or drawdowns have increased."),
    "BEAR_CALM": ("Bearish but Calm", "#b8860b", "Price is below 250-day trend line, but volatility remains contained."),
    "BEAR_SCARY": ("Bearish & Volatile", "#c0392b", "Price is below trend line AND volatility is elevated."),
    "BUFFER": ("Neutral / Transition Zone", "#6c757d", "Price is close to its 250-day trend line."),
}

def compute_market_regime_tier(df, fund="C Fund", trend_ma_window=250):
    if len(df) < trend_ma_window + 10 or "VIX" not in df.columns:
        return None

    trend_ma = df[fund].rolling(window=trend_ma_window).mean()
    rolling_high = df[fund].rolling(window=252, min_periods=1).max()
    drawdown_pct = (df[fund] - rolling_high) / rolling_high * 100
    vix_10ma = df["VIX"].rolling(window=10).mean()

    latest_date = df.index[-1]
    price = df.loc[latest_date, fund]
    ma_val = trend_ma.loc[latest_date]
    dd_val = drawdown_pct.loc[latest_date]
    vix_val = vix_10ma.loc[latest_date]

    if pd.isna(ma_val) or pd.isna(dd_val) or pd.isna(vix_val):
        return None

    pct_vs_ma = (price - ma_val) / ma_val * 100

    if pct_vs_ma >= 1.0:
        tier = "BULL_CALM" if (vix_val < 22 and dd_val > -5) else "BULL_SHAKY"
    elif pct_vs_ma <= -5.0:
        tier = "BEAR_CALM" if vix_val < 15 else "BEAR_SCARY"
    else:
        tier = "BUFFER"

    return {"tier": tier, "pct_vs_ma": pct_vs_ma, "vix_10ma": vix_val, "drawdown_pct": dd_val}

def generate_market_regime_box_html(tier_info):
    if tier_info is None:
        return '<div style="background-color: #f8f9fa; padding: 12px; font-size: 13px; color: #999;">Market regime tier unavailable today.</div>'
    
    label, color, description = TRIPOD_TIER_DESCRIPTIONS[tier_info["tier"]]
    return f"""
    <div style="background-color: #ffffff; border-left: 5px solid {color}; border: 1px solid #dee2e6; border-radius: 4px; padding: 16px; margin-bottom: 20px;">
        <h2 style="margin-top: 0; font-size: 16px; color: #333333;">Market Regime Tier (C Fund)</h2>
        <p style="font-size: 20px; font-weight: bold; color: {color}; margin: 4px 0;">{label}</p>
        <p style="font-size: 13px; color: #555555; margin: 6px 0 12px 0;">{description}</p>
        <table style="font-size: 12.5px; color: #666666; border-collapse: collapse;">
            <tr><td style="padding: 2px 10px 2px 0;">Price vs. 250-day trend:</td><td style="font-weight: bold; color: {'#1e7e34' if tier_info['pct_vs_ma'] >= 0 else '#c0392b'};">{tier_info['pct_vs_ma']:+.1f}%</td></tr>
            <tr><td style="padding: 2px 10px 2px 0;">VIX (10-day average):</td><td style="font-weight: bold;">{tier_info['vix_10ma']:.1f}</td></tr>
            <tr><td style="padding: 2px 10px 2px 0;">Drawdown from 52-week high:</td><td style="font-weight: bold; color: {'#1e7e34' if tier_info['drawdown_pct'] >= -3 else '#c0392b'};">{tier_info['drawdown_pct']:.1f}%</td></tr>
        </table>
    </div>
    """

def build_diagnostics_table(df):
    latest = df.iloc[-1]
    latest_date = df.index[-1].strftime("%Y-%m-%d")
    fund_scores = {}
    html_rows = ""
    
    for fund in ["C Fund", "S Fund", "I Fund", "G Fund"]:
        price = latest[fund]
        pct_diffs = {}
        passed_count = 0
        composite_score = 0.0
        
        for ma in MA_WINDOWS:
            ma_val = latest[f"{fund}_{ma}MA"]
            diff_pct = (price - ma_val) / ma_val * 100
            pct_diffs[ma] = diff_pct
            if price > ma_val:
                passed_count += 1
                
        for ma, weight in COMPOSITE_WEIGHTS.items():
            composite_score += weight * pct_diffs[ma]
            
        fund_scores[fund] = {
            "price": price, "pct_diffs": pct_diffs,
            "passed_count": passed_count, "composite_score": composite_score,
        }

        def _c(v): return "#008000" if v >= 0 else "#CC0000"

        ma_cells = "".join(f'<td style="padding: 8px; color: {_c(pct_diffs[ma])};">{pct_diffs[ma]:+.2f}%</td>' for ma in MA_WINDOWS)
        html_rows += f"""
        <tr style="text-align: center; border-bottom: 1px solid #dddddd;">
            <td style="padding: 8px; font-weight: bold; text-align: left;">{fund}</td>
            <td style="padding: 8px;">${price:,.2f}</td>
            {ma_cells}
            <td style="padding: 8px; font-weight: bold;">{passed_count} / {len(MA_WINDOWS)}</td>
            <td style="padding: 8px; font-weight: bold; color: {_c(composite_score)};">{composite_score:+.2f}%</td>
        </tr>
        """
    return latest_date, html_rows, fund_scores

def determine_allocation(fund_scores):
    stock_funds = ["C Fund", "S Fund", "I Fund"]
    
    def _mas_passed_short(fund):
        diffs = fund_scores[fund]["pct_diffs"]
        return sum(1 for ma in COMPOSITE_MA_WINDOWS if diffs[ma] >= 0)

    eligible = {f: fund_scores[f] for f in stock_funds if _mas_passed_short(f) >= MIN_MAS_PASSED_TO_QUALIFY}

    if not eligible:
        return "G Fund", f"No stock fund passed at least {MIN_MAS_PASSED_TO_QUALIFY} short-term moving average filters."

    target = max(eligible, key=lambda f: eligible[f]["composite_score"])
    top_score = eligible[target]["composite_score"]
    passed = _mas_passed_short(target)
    reason = f"{target} holds top composite momentum score ({top_score:+.2f}%) among funds passing trend filters ({passed}/3 passed)."
    return target, reason

def generate_allocation_box_html(target, reason, fund_scores):
    def _color(v): return "#008000" if v >= 0 else "#CC0000"

    rows = "".join(
        f"""<tr>
            <td style="padding:6px 10px;border:1px solid #dee2e6;font-weight:bold;">{f}</td>
            <td style="padding:6px 10px;border:1px solid #dee2e6;text-align:right;">${fund_scores[f]['price']:,.2f}</td>
            <td style="padding:6px 10px;border:1px solid #dee2e6;text-align:right;color:{_color(fund_scores[f]['pct_diffs'][5])};">{fund_scores[f]['pct_diffs'][5]:+.2f}%</td>
            <td style="padding:6px 10px;border:1px solid #dee2e6;text-align:right;color:{_color(fund_scores[f]['pct_diffs'][20])};">{fund_scores[f]['pct_diffs'][20]:+.2f}%</td>
            <td style="padding:6px 10px;border:1px solid #dee2e6;text-align:right;color:{_color(fund_scores[f]['pct_diffs'][60])};">{fund_scores[f]['pct_diffs'][60]:+.2f}%</td>
            <td style="padding:6px 10px;border:1px solid #dee2e6;text-align:right;font-weight:bold;color:{_color(fund_scores[f]['composite_score'])};">{fund_scores[f]['composite_score']:+.2f}%</td>
        </tr>"""
        for f in ["C Fund", "S Fund", "I Fund"]
    )
    return f"""
    <div style="background-color: #fff8f0; border-left: 5px solid #d93025; border: 1px solid #dee2e6; border-radius: 4px; padding: 16px; margin-bottom: 20px;">
        <h2 style="margin-top: 0; color: #d93025; font-size: 16px;">TSP Allocation Suggestion</h2>
        <p style="font-size: 15px; margin-bottom: 4px;"><b>Suggested Allocation:</b>
            <span style="font-size: 20px; color: #1a73e8; font-weight: bold;">{target}</span></p>
        <p style="font-size: 13px; color: #333333; margin-top: 4px;"><b>Rationale:</b> {reason}</p>
        <table style="width: 100%; border-collapse: collapse; font-size: 12.5px; margin-top: 12px; background-color: #ffffff;">
            <thead><tr style="background-color: #f1f3f4; text-align: right;">
                <th style="padding: 6px 10px; text-align: left;">Fund</th><th style="padding: 6px 10px;">Price</th>
                <th style="padding: 6px 10px;">5MA (%)</th><th style="padding: 6px 10px;">20MA (%)</th>
                <th style="padding: 6px 10px;">60MA (%)</th><th style="padding: 6px 10px;">Composite</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    """

def generate_signal_chart(df, lookback_days=252):
    chart_data = df.iloc[-lookback_days:].copy()
    funds = ["C Fund", "S Fund", "I Fund"]
    colors = {"5": "#4CAF50", "20": "#FF9800", "60": "#9C27B0", "100": "#2196F3", "200": "#E91E63"}
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    
    for idx, fund in enumerate(funds):
        ax = axes[idx]
        ax.plot(chart_data.index, chart_data[fund], label=f"{fund} Price", color="#111111", linewidth=2.0)
        for ma in MA_WINDOWS:
            ax.plot(chart_data.index, chart_data[f"{fund}_{ma}MA"], label=f"{ma}-Day MA",
                    color=colors[str(ma)], linewidth=1.1, linestyle="--" if ma != 200 else "-")
        ax.set_title(f"{fund}: 1-Year Price & Moving Averages", fontsize=11, fontweight="bold")
        ax.set_ylabel("Price ($)", fontsize=9)
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.2f}"))
        ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", ncols=6, fontsize=7.5)
        
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()

def run_section1_daily_signal(prices):
    df = compute_moving_averages(prices)
    latest_date, diagnostics_html, fund_scores = build_diagnostics_table(df)
    target, reason = determine_allocation(fund_scores)
    allocation_html = generate_allocation_box_html(target, reason, fund_scores)

    regime_series = compute_regime_correlation(df)
    latest_corr = regime_series.iloc[-1]
    regime_label, regime_color = classify_regime(latest_corr)
    corr_display = "N/A" if pd.isna(latest_corr) else f"{latest_corr:+.2f}"

    regime_html = f"""
    <div style="background-color: #ffffff; border-left: 4px solid {regime_color}; border: 1px solid #dee2e6; border-radius: 4px; padding: 12px; font-size: 13px; margin-bottom: 20px;">
        <b>Market Regime (20-Day Rolling Avg Correlation):</b>
        <span style="color: {regime_color}; font-weight: bold;"> {regime_label} ({corr_display})</span>
    </div>
    """

    regime_info = compute_market_regime_tier(df, fund="C Fund")
    market_tier_html = generate_market_regime_box_html(regime_info)
    chart_bytes = generate_signal_chart(df)

    ma_headers = "".join(f'<th style="padding: 8px;">{ma} MA (%)</th>' for ma in MA_WINDOWS)
    section_html = f"""
    <h2>Pillar 5: Daily Signal</h2>
    {allocation_html}
    {regime_html}
    {market_tier_html}
    <h3>Moving Average Diagnostics &amp; Composite Score</h3>
    <table style="border-collapse: collapse; width: 100%; max-width: 900px; font-size: 13px; border: 1px solid #dddddd;">
        <thead><tr style="background-color: #2c3e50; color: #ffffff; text-align: center;">
            <th style="padding: 8px; text-align: left;">Fund</th><th style="padding: 8px;">Price</th>
            {ma_headers}<th style="padding: 8px;">MAs Passed</th><th style="padding: 8px;">Composite Score</th>
        </tr></thead>
        <tbody>{diagnostics_html}</tbody>
    </table>
    <br>
    <h3>1-Year Price Trajectories (C / S / I)</h3>
    <img src="cid:signal_chart" style="max-width: 100%; height: auto; border: 1px solid #ccc;">
    """
    return section_html, chart_bytes

# =====================================================================
# SECTION 2: PILLARS 3 & 4
# =====================================================================

def fetch_official_tsp_prices(lookback_days=10):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    urls_to_try = [
        "https://www.tsp.gov/data/fund-price-history.csv",
        f"https://www.tsp.gov/data/getSharePrices_startdate_{start_date.strftime('%Y%m%d')}_enddate_{end_date.strftime('%Y%m%d')}_Lfunds_1_InvFunds_1_download_1.csv",
    ]
    browser_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.tsp.gov/share-price-history/",
    }
    for url in urls_to_try:
        try:
            req = urllib.request.Request(url, headers=browser_headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                raw_csv = response.read().decode("utf-8")
            reader = csv.DictReader(io.StringIO(raw_csv))
            rows = list(reader)
            if not rows: continue
            df = pd.DataFrame(rows)
            date_col = next((c for c in df.columns if c.strip().lower() == "date"), None)
            if not date_col: continue
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col).sort_index()
            available_funds = [f for f in TSP_FUNDS if f in df.columns]
            for f in available_funds:
                df[f] = pd.to_numeric(df[f], errors="coerce")
            return df[available_funds].dropna(how="all")
        except Exception:
            continue
    return pd.DataFrame()

def fetch_market_internals():
    results = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10)
    try:
        data = yf.download(list(INTERNALS_TICKERS.values()), start=start_date, end=end_date, auto_adjust=True, progress=False)
        closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
        for label, ticker in INTERNALS_TICKERS.items():
            if ticker in closes.columns:
                series = closes[ticker].dropna()
                if len(series) >= 2:
                    latest_price = series.iloc[-1]
                    prev_price = series.iloc[-2]
                    change_pct = (latest_price - prev_price) / prev_price * 100
                    results[label] = {"price": latest_price, "change_pct": change_pct}
    except Exception as e:
        print(f"[-] Market internals fetch failed: {e}")
    return results

def build_pillar3_html(official_df, cached_prices):
    if official_df.empty:
        return '<div style="background-color: #fff3cd; padding: 12px; font-size: 13px;">Could not retrieve official TSP prices today.</div>'
    
    latest_date = official_df.index[-1]
    rows_html = ""
    for fund in TSP_FUNDS:
        if fund not in official_df.columns: continue
        series = official_df[fund].dropna()
        if series.empty: continue
        latest_price = series.iloc[-1]
        official_change_pct = (latest_price - series.iloc[-2]) / series.iloc[-2] * 100 if len(series) >= 2 else None
        
        proxy_change_pct = None
        if fund in cached_prices and len(cached_prices[fund]) >= 2:
            p_series = cached_prices[fund].dropna()
            proxy_change_pct = (p_series.iloc[-1] - p_series.iloc[-2]) / p_series.iloc[-2] * 100

        def _color(v): return "#999999" if v is None else ("#1e7e34" if v >= 0 else "#c0392b")

        official_str = f"{official_change_pct:+.2f}%" if official_change_pct is not None else "N/A"
        proxy_str = f"{proxy_change_pct:+.2f}%" if proxy_change_pct is not None else "N/A"
        
        rows_html += f"""
        <tr style="border-bottom: 1px solid #dee2e6;">
            <td style="padding: 8px; font-weight: bold; text-align: left;">{fund}</td>
            <td style="padding: 8px; text-align: right;">${latest_price:.2f}</td>
            <td style="padding: 8px; text-align: right; color: {_color(official_change_pct)}; font-weight: bold;">{official_str}</td>
            <td style="padding: 8px; text-align: right; color: {_color(proxy_change_pct)};">{proxy_str}</td>
        </tr>
        """
    return f"""
    <div style="background-color: #ffffff; border-left: 5px solid #185fa5; border: 1px solid #dee2e6; border-radius: 4px; padding: 16px; margin-bottom: 20px;">
        <h2 style="margin-top: 0; font-size: 16px; color: #333333;">Pillar 3: Official TSP Fund Prices</h2>
        <p style="font-size: 12px; color: #666666; margin-bottom: 12px;">Source: tsp.gov. As of {latest_date.strftime('%Y-%m-%d')}.</p>
        <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
            <thead><tr style="background-color: #f1f3f4; text-align: right;">
                <th style="padding: 6px 10px; text-align: left;">Fund</th><th style="padding: 6px 10px;">Official Price</th>
                <th style="padding: 6px 10px;">Official Change</th><th style="padding: 6px 10px;">Proxy ETF Change</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """

def build_pillar4_html(internals):
    if not internals:
        return '<div style="background-color: #fff3cd; padding: 12px; font-size: 13px;">Market internals unavailable.</div>'
    rows_html = ""
    for label, data in internals.items():
        change_pct = data["change_pct"]
        color = "#1e7e34" if change_pct >= 0 else "#c0392b"
        unit = "%" if label == "10-Year Treasury Yield" else ""
        rows_html += f"""
        <tr style="border-bottom: 1px solid #dee2e6;">
            <td style="padding: 8px; font-weight: bold; text-align: left;">{label}</td>
            <td style="padding: 8px; text-align: right;">{data['price']:.2f}{unit}</td>
            <td style="padding: 8px; text-align: right; color: {color}; font-weight: bold;">{change_pct:+.2f}%</td>
        </tr>
        """
    return f"""
    <div style="background-color: #ffffff; border-left: 5px solid #d93025; border: 1px solid #dee2e6; border-radius: 4px; padding: 16px; margin-bottom: 20px;">
        <h2 style="margin-top: 0; font-size: 16px; color: #333333;">Pillar 4: Market Internals</h2>
        <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
            <thead><tr style="background-color: #f1f3f4; text-align: right;">
                <th style="padding: 6px 10px; text-align: left;">Indicator</th><th style="padding: 6px 10px;">Level</th><th style="padding: 6px 10px;">Day Change</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """

def run_section2_pillars_3_4(cached_prices):
    official_df = fetch_official_tsp_prices()
    internals = fetch_market_internals()
    p3_html = build_pillar3_html(official_df, cached_prices)
    p4_html = build_pillar4_html(internals)
    return f"<h2>Pillars 3 &amp; 4: Prices &amp; Internals</h2>{p3_html}{p4_html}"

# =====================================================================
# SECTION 3: TRIPLE-MACD
# =====================================================================

def calculate_macd(df, fast=12, slow=26, signal=9):
    exp1 = df["Close"].ewm(span=fast, adjust=False).mean()
    exp2 = df["Close"].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - macd_signal
    return macd, macd_signal, hist

def fetch_and_compute_macd(ticker_symbol, fund_label):
    t = yf.Ticker(ticker_symbol, session=custom_session)
    df = t.history(period="6mo")
    if df.empty: return None, None
    
    latest_date_str = df.index[-1].strftime("%Y-%m-%d")
    df["MACD_S"], df["Sig_S"], df["Hist_S"] = calculate_macd(df, 6, 13, 5)
    df["MACD_M"], df["Sig_M"], df["Hist_M"] = calculate_macd(df, 12, 26, 9)
    df["MACD_L"], df["Sig_L"], df["Hist_L"] = calculate_macd(df, 24, 52, 18)
    
    df["Signal_Short"] = (df["MACD_S"] > df["Sig_S"]).astype(int)
    df["Signal_Medium"] = (df["MACD_M"] > df["Sig_M"]).astype(int)
    df["Signal_Long"] = (df["MACD_L"] > df["Sig_L"]).astype(int)
    df["Total_Score"] = df["Signal_Short"] + df["Signal_Medium"] + df["Signal_Long"]
    df["Buy_Signal"] = (df["Total_Score"] >= 2).astype(int)
    
    latest = df.iloc[-1]
    summary = {
        "ticker": ticker_symbol, "fund_label": fund_label, "date": latest_date_str,
        "short_bullish": bool(latest["Signal_Short"]), "medium_bullish": bool(latest["Signal_Medium"]),
        "long_bullish": bool(latest["Signal_Long"]), "score": int(latest["Total_Score"]),
        "verdict_bullish": bool(latest["Buy_Signal"]),
    }
    return df, summary

def build_combined_macd_chart(fund_data):
    _MC = mpf.make_marketcolors(
        up="#00B050", down="#FF0000", edge="inherit", wick="inherit",
        volume={"up": "#00B050", "down": "#FF0000"},
    )
    _STYLE = mpf.make_mpf_style(
        base_mpf_style="charles", marketcolors=_MC, gridcolor="#666666",
        facecolor="#111111", rc={"font.size": 8, "grid.linewidth": 1.2, "grid.linestyle": "--"},
    )
    n_funds = len(fund_data)
    fig = plt.figure(figsize=(12, 7.5 * n_funds), facecolor="#111111")
    gs = fig.add_gridspec(4 * n_funds, 1, height_ratios=[4, 1.3, 1.3, 1.3] * n_funds, hspace=0.45, top=0.97, bottom=0.03)

    for i, (ticker_symbol, fund_label, df, summary) in enumerate(fund_data):
        base_row = i * 4
        ax_price = fig.add_subplot(gs[base_row])
        ax_short = fig.add_subplot(gs[base_row + 1], sharex=ax_price)
        ax_medium = fig.add_subplot(gs[base_row + 2], sharex=ax_price)
        ax_long = fig.add_subplot(gs[base_row + 3], sharex=ax_price)

        def _hist_colors(s): return ["#00B050" if v >= 0 else "#FF0000" for v in s]

        add_plots = [
            mpf.make_addplot(df["Hist_S"], ax=ax_short, type="bar", color=_hist_colors(df["Hist_S"])),
            mpf.make_addplot(df["MACD_S"], ax=ax_short, color="#00FFFF", width=1.0),
            mpf.make_addplot(df["Sig_S"], ax=ax_short, color="#FF9900", width=1.0),
            mpf.make_addplot(df["Hist_M"], ax=ax_medium, type="bar", color=_hist_colors(df["Hist_M"])),
            mpf.make_addplot(df["MACD_M"], ax=ax_medium, color="#00FFFF", width=1.0),
            mpf.make_addplot(df["Sig_M"], ax=ax_medium, color="#FF9900", width=1.0),
            mpf.make_addplot(df["Hist_L"], ax=ax_long, type="bar", color=_hist_colors(df["Hist_L"])),
            mpf.make_addplot(df["MACD_L"], ax=ax_long, color="#00FFFF", width=1.0),
            mpf.make_addplot(df["Sig_L"], ax=ax_long, color="#FF9900", width=1.0),
        ]
        
        buy_signals = df["Close"] * (df["Buy_Signal"] == 1)
        buy_signals[buy_signals == 0] = None
        add_plots.append(mpf.make_addplot(buy_signals, ax=ax_price, type="scatter", markersize=35, marker="^", color="#00FF00"))

        mpf.plot(df, type="candle", volume=False, addplot=add_plots, style=_STYLE, ax=ax_price, axtitle=f"{fund_label} ({ticker_symbol}) -- {summary['date']}")

        for ax in (ax_price, ax_short, ax_medium, ax_long):
            ax.set_facecolor("#111111")
            ax.tick_params(axis="both", colors="white", labelsize=7)
            for spine in ax.spines.values(): spine.set_color("#666666")
            ax.grid(True, color="#444444", linestyle="--", linewidth=0.6, alpha=0.6)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), dpi=110)
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()

def run_section3_macd():
    fund_data, summaries = [], []
    for symbol, label in MACD_TICKERS.items():
        df, summary = fetch_and_compute_macd(symbol, label)
        if df is not None:
            fund_data.append((symbol, label, df, summary))
            summaries.append(summary)

    if not fund_data:
        raise ValueError("No MACD data fetched.")

    chart_bytes = build_combined_macd_chart(fund_data)
    
    rows_html = ""
    for s in summaries:
        verdict_color = "#1e7e34" if s["verdict_bullish"] else "#c0392b"
        verdict_bg = "#e8f5e9" if s["verdict_bullish"] else "#fdecea"
        verdict_label = "&#128293; Bullish Momentum" if s["verdict_bullish"] else "&#9888; Bearish / Caution"
        rows_html += f"""
        <tr style="background-color: {verdict_bg}; border-bottom: 1px solid #dee2e6;">
            <td style="padding: 8px; font-weight: bold; text-align: left;">{s['fund_label']}</td>
            <td style="padding: 8px; text-align: center;">{"Bullish" if s['short_bullish'] else "Bearish"}</td>
            <td style="padding: 8px; text-align: center;">{"Bullish" if s['medium_bullish'] else "Bearish"}</td>
            <td style="padding: 8px; text-align: center;">{"Bullish" if s['long_bullish'] else "Bearish"}</td>
            <td style="padding: 8px; text-align: center; font-weight: bold;">{s['score']} / 3</td>
            <td style="padding: 8px; text-align: center; color: {verdict_color}; font-weight: bold;">{verdict_label}</td>
        </tr>
        """
        
    html = f"""
    <h2>Triple-MACD Momentum Confirmation</h2>
    <table style="width: 100%; max-width: 700px; border-collapse: collapse; font-size: 13px; border: 1px solid #dddddd;">
        <thead><tr style="background-color: #2c3e50; color: #ffffff; text-align: center;">
            <th style="padding: 10px; text-align: left;">Fund</th><th>Short</th><th>Medium</th><th>Long</th><th>Score</th><th>Verdict</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    <img src="cid:macd_chart" style="max-width: 100%; height: auto; margin-top: 16px;">
    """
    return html, chart_bytes

# =====================================================================
# MAIN EXECUTION & EMAIL
# =====================================================================

def send_combined_email(report_date, section_htmls, images):
    msg = MIMEMultipart("related")
    msg["Subject"] = f"TSP Combined Daily Report ({report_date})"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL

    full_html = f"""
    <html>
    <body style="font-family: Arial, sans-serif; color: #333333; line-height: 1.5;">
        <h1>TSP Combined Daily Report ({report_date})</h1>
        {''.join(section_htmls)}
    </body>
    </html>
    """
    msg.attach(MIMEText(full_html, "html"))

    for cid, img_bytes, filename in images:
        if img_bytes:
            img = MIMEImage(img_bytes)
            img.add_header("Content-ID", f"<{cid}>")
            img.add_header("Content-Disposition", "inline", filename=filename)
            msg.attach(img)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        print(f"[+] Combined report sent successfully to {RECIPIENT_EMAIL}!")
    except Exception as e:
        print(f"[-] Failed to send email: {e}")

def main():
    print("[*] Running TSP Combined Daily Report...")
    section_htmls, images = [], []
    report_date = datetime.now().strftime("%Y-%m-%d")

    # Fetch global proxy prices once for both Section 1 and Section 2
    cached_prices = download_historical_data(years=2, ma_warmup_days=260, include_vix=True)

    # Section 1
    try:
        html1, chart1_bytes = run_section1_daily_signal(cached_prices)
        section_htmls.append(html1)
        images.append(("signal_chart", chart1_bytes, "tsp_signal_chart.png"))
    except Exception as e:
        section_htmls.append(f"<div><h2>Section 1 Failed</h2><p>{e}</p></div>")

    # Section 2
    try:
        html2 = run_section2_pillars_3_4(cached_prices)
        section_htmls.append(html2)
    except Exception as e:
        section_htmls.append(f"<div><h2>Section 2 Failed</h2><p>{e}</p></div>")

    # Section 3
    if MACD_AVAILABLE:
        try:
            html3, chart3_bytes = run_section3_macd()
            section_htmls.append(html3)
            images.append(("macd_chart", chart3_bytes, "tsp_triple_macd_combined.png"))
        except Exception as e:
            section_htmls.append(f"<div><h2>Section 3 Failed</h2><p>{e}</p></div>")

    send_combined_email(report_date, section_htmls, images)

if __name__ == "__main__":
    main()