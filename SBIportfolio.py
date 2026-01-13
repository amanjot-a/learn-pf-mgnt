import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

# =====================================================
# 1. USER CONFIGURATION
# =====================================================

BASE_DIR = r"D:\e\VS\Portfolio Management"

ASSET_FILE = os.path.join(BASE_DIR, "SBI Historical Data.xlsx")
MARKET_FILE = os.path.join(BASE_DIR, "Market.xlsx")
OUTPUT_FILE = os.path.join(BASE_DIR, "SBI_Risk_Report.xlsx")

PRICE_COL = "Adj Close"          # auto-fallback if missing
ANNUAL_RF = 0.06                 # 6% annual risk-free rate
PERIODS_PER_YEAR = 252           # daily data
TARGET_VOL = 0.12                # 12% annual target volatility
ROLLING_WINDOW = 60


# =====================================================
# 2. LOAD PRICES (ROBUST)
# =====================================================

def load_prices_excel(path):
    df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)
    return df


# =====================================================
# 3. LOAD ASSET DATA
# =====================================================

asset_df = load_prices_excel(ASSET_FILE)

price_col = PRICE_COL if PRICE_COL in asset_df.columns else asset_df.columns[-1]
asset_prices = asset_df[price_col].astype(float)

asset_returns = asset_prices.pct_change()


# =====================================================
# 4. CREATE MARKET FILE IF NOT EXISTS (PRICES, NOT RETURNS)
# =====================================================

if not os.path.exists(MARKET_FILE):
    print("⚠️ Market file not found. Creating synthetic market prices...")

    market_prices = (1 + asset_returns.fillna(0)).cumprod()

    market_df = pd.DataFrame({
        "Date": market_prices.index,
        "Market": market_prices.values
    })

    market_df.to_excel(MARKET_FILE, index=False)
    print(f"✅ Market file created: {MARKET_FILE}")

market_df = load_prices_excel(MARKET_FILE)
market_prices = market_df.iloc[:, 0].astype(float)
market_returns = market_prices.pct_change()


# =====================================================
# 5. CLEAN & ALIGN DATA (CRITICAL)
# =====================================================

data = pd.concat([asset_returns, market_returns], axis=1)
data.columns = ["Asset", "Market"]

data = data.replace([np.inf, -np.inf], np.nan).dropna()

asset_returns = data["Asset"]
market_returns = data["Market"]


# =====================================================
# 6. RISK-FREE RATE (ANNUAL → DAILY)
# =====================================================

rf = (1 + ANNUAL_RF) ** (1 / PERIODS_PER_YEAR) - 1


# =====================================================
# 7. CORE PERFORMANCE METRICS
# =====================================================

def sharpe_ratio(r):
    return (r.mean() - rf) / r.std(ddof=1)

def sortino_ratio(r):
    downside = r[r < rf]
    return (r.mean() - rf) / downside.std(ddof=1)

def max_drawdown(r):
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    return ((cum - peak) / peak).min()

summary_df = pd.DataFrame({
    "Annual Return": [(1 + asset_returns.mean()) ** PERIODS_PER_YEAR - 1],
    "Annual Volatility": [asset_returns.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)],
    "Sharpe Ratio": [sharpe_ratio(asset_returns)],
    "Sortino Ratio": [sortino_ratio(asset_returns)],
    "Max Drawdown": [max_drawdown(asset_returns)]
}, index=["SBI"])


# =====================================================
# 8. CAPM REGRESSION (SAFE & CLEAN)
# =====================================================

excess_asset = asset_returns - rf
excess_market = market_returns - rf

X = sm.add_constant(excess_market)

capm_model = sm.OLS(excess_asset, X, missing="drop").fit()

capm_df = pd.DataFrame({
    "Coefficient": capm_model.params,
    "t-Stat": capm_model.tvalues,
    "p-Value": capm_model.pvalues
})

capm_df.loc["R-Squared", "Coefficient"] = capm_model.rsquared


# =====================================================
# 9. RISK DECOMPOSITION
# =====================================================

beta = capm_model.params[1]
market_var = market_returns.var(ddof=1)
systematic_var = (beta ** 2) * market_var
idiosyncratic_var = capm_model.resid.var(ddof=1)

risk_decomp_df = pd.DataFrame({
    "Variance": [systematic_var, idiosyncratic_var],
    "Contribution %": [
        systematic_var / (systematic_var + idiosyncratic_var),
        idiosyncratic_var / (systematic_var + idiosyncratic_var)
    ]
}, index=["Systematic Risk", "Idiosyncratic Risk"])


# =====================================================
# 10. PORTFOLIO CONSTRUCTION (CAL / TARGET VOL)
# =====================================================

asset_ann_vol = asset_returns.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)

weight_risky = TARGET_VOL / asset_ann_vol
weight_rf = 1 - weight_risky

portfolio_df = pd.DataFrame({
    "Weight": [weight_risky, weight_rf],
    "Explanation": [
        "Scaled risky asset allocation",
        "Risk-free allocation"
    ]
}, index=["SBI (Risky)", "Risk-Free"])


# =====================================================
# 11. ROLLING METRICS
# =====================================================

rolling_df = pd.DataFrame({
    "Rolling Sharpe": asset_returns.rolling(ROLLING_WINDOW).apply(
        lambda x: (x.mean() - rf) / x.std(ddof=1)
    ),
    "Rolling Beta": (
        asset_returns.rolling(ROLLING_WINDOW).cov(market_returns)
        / market_returns.rolling(ROLLING_WINDOW).var()
    )
}).dropna()


# =====================================================
# 12. WRITE PROFESSIONAL EXCEL REPORT
# =====================================================

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="Summary")
    capm_df.to_excel(writer, sheet_name="CAPM_Regression")
    risk_decomp_df.to_excel(writer, sheet_name="Risk_Decomposition")
    portfolio_df.to_excel(writer, sheet_name="Portfolio_Construction")
    rolling_df.to_excel(writer, sheet_name="Rolling_Metrics")

print("\n✅ SUCCESS: Professional risk report generated")
print(f"📊 Output file: {OUTPUT_FILE}")
