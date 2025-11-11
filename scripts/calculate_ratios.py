import pandas as pd
import numpy as np

from pathlib import Path


BASE_PATH = Path("/Users/mihirrao/mihir/Stocks-Deep-Research-Agent")
BALANCE_SHEET_CSV = BASE_PATH / "data" / "yfinance_outputs" / "balance_sheet.csv"


def load_balance_sheet(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError("Expected a 'ticker' column in the balance sheet export.")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df


def get_series(df: pd.DataFrame, column_candidates: list[str]) -> pd.Series:
    for column in column_candidates:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.div(denominator.replace({0: np.nan}))
    result = result.replace({np.inf: np.nan, -np.inf: np.nan})
    return result


def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    current_assets = get_series(df, ["Current Assets"])
    current_liabilities = get_series(df, ["Current Liabilities"])
    inventory = get_series(df, ["Inventory", "Inventories"])
    prepaid_assets = get_series(df, ["Prepaid Assets"])
    total_assets = get_series(df, ["Total Assets"])
    total_debt = get_series(df, ["Total Debt"])
    net_debt = get_series(df, ["Net Debt"])
    working_capital_reported = get_series(df, ["Working Capital"])
    tangible_book_value = get_series(df, ["Tangible Book Value"])

    book_value = get_series(
        df,
        [
            "Common Stock Equity",
            "Stockholders Equity",
            "Total Equity Gross Minority Interest",
        ],
    )

    shares_outstanding = get_series(
        df,
        [
            "Ordinary Shares Number",
            "Share Issued",
            "Common Stock Shares Outstanding",
        ],
    )

    quick_assets = current_assets - inventory.fillna(0) - prepaid_assets.fillna(0)
    quick_assets = quick_assets.clip(lower=0)

    enriched["current_ratio"] = safe_div(current_assets, current_liabilities)
    enriched["quick_ratio"] = safe_div(quick_assets, current_liabilities)
    enriched["working_capital_computed"] = current_assets - current_liabilities
    enriched["working_capital_reported"] = working_capital_reported
    enriched["book_value"] = book_value
    enriched["book_value_per_share"] = safe_div(book_value, shares_outstanding)
    enriched["tangible_book_value"] = tangible_book_value
    enriched["tangible_book_value_per_share"] = safe_div(tangible_book_value, shares_outstanding)
    enriched["debt_to_equity_ratio"] = safe_div(total_debt, book_value)
    enriched["net_debt_to_equity_ratio"] = safe_div(net_debt, book_value)
    enriched["equity_ratio"] = safe_div(book_value, total_assets)

    return enriched


def save_balance_sheet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    balance_sheet_df = load_balance_sheet(BALANCE_SHEET_CSV)
    updated_df = compute_ratios(balance_sheet_df)
    save_balance_sheet(updated_df, BALANCE_SHEET_CSV)
    print(f"Updated balance sheet with calculated ratios at {BALANCE_SHEET_CSV}")
    import pandas as pd



if __name__ == "__main__":
    main()

