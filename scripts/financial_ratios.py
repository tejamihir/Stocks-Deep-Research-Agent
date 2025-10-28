
import yfinance as yf
import pandas as pd
from pathlib import Path

# Save under repo_root/data/yfinance_outputs (folder must already exist)
repo_root = Path(__file__).resolve().parents[1]
out_dir = repo_root / 'data' / 'yfinance_outputs'

# Load S&P 500 tickers
sp500_path = repo_root / 'sp500_tickers.csv'
if sp500_path.exists():
    sp500_df = pd.read_csv(sp500_path)
else:
    # fallback to online if local file missing
    sp500_df = pd.read_csv(
        'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'
    )
sp500_symbols = sp500_df['Symbol'].dropna().unique().tolist()

def build_one_row(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame([{'ticker': ticker}])
    df = df.reset_index().rename(columns={'index': 'metrics'})
    value_col = next((c for c in df.columns if c != 'metrics'), None)
    if value_col is None:
        return pd.DataFrame([{'ticker': ticker}])
    series = df.set_index('metrics')[value_col]
    row = {'ticker': ticker}
    row.update(series.to_dict())
    return pd.DataFrame([row])

financials_rows: list[pd.DataFrame] = []
balance_rows: list[pd.DataFrame] = []
cashflow_rows: list[pd.DataFrame] = []

for symbol in sp500_symbols:
    try:
        stk = yf.Ticker(symbol)
        fin = build_one_row(stk.financials, symbol)
        bs = build_one_row(stk.balance_sheet, symbol)
        cf = build_one_row(stk.cashflow, symbol)
    except Exception:
        # ensure at least ticker present on failures
        fin = pd.DataFrame([{'ticker': symbol}])
        bs = pd.DataFrame([{'ticker': symbol}])
        cf = pd.DataFrame([{'ticker': symbol}])
    financials_rows.append(fin)
    balance_rows.append(bs)
    cashflow_rows.append(cf)

# Concatenate into final outputs
financials_all = pd.concat(financials_rows, ignore_index=True).drop_duplicates(subset=['ticker'])
balance_sheet_all = pd.concat(balance_rows, ignore_index=True).drop_duplicates(subset=['ticker'])
cashflow_all = pd.concat(cashflow_rows, ignore_index=True).drop_duplicates(subset=['ticker'])

# Save
financials_all.to_csv(out_dir / "financials.csv", index=False)
balance_sheet_all.to_csv(out_dir / "balance_sheet.csv", index=False)
cashflow_all.to_csv(out_dir / "cashflow.csv", index=False)
# Example: Compute some common financial ratios
# 1. Current Ratio = Current Assets / Current Liabilities
# current_assets = balance_sheet.loc['Total Current Assets'][0]
# current_liabilities = balance_sheet.loc['Total Current Liabilities'][0]
# current_ratio = current_assets / current_liabilities

# # 2. Debt-to-Equity Ratio = Total Liabilities / Total Shareholders' Equity
# total_liabilities = balance_sheet.loc['Total Liab'][0]
# total_equity = balance_sheet.loc["Total Stockholder Equity"][0]
# debt_to_equity = total_liabilities / total_equity

# # 3. Return on Equity (ROE) = Net Income / Total Equity
# net_income = financials.loc['Net Income'][0]
# roe = net_income / total_equity

# # 4. Profit Margin = Net Income / Total Revenue
# revenue = financials.loc['Total Revenue'][0]
# profit_margin = net_income / revenue

# # Print results
# print(f"Financial Ratios for {ticker}:")
# print(f"Current Ratio: {current_ratio:.2f}")
# print(f"Debt-to-Equity Ratio: {debt_to_equity:.2f}")
# print(f"Return on Equity (ROE): {roe:.2%}")
# print(f"Profit Margin: {profit_margin:.2%}")


