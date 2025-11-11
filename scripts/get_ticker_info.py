from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

def main() -> None:
    """
    Fetch the S&P 500 constituents table from Wikipedia and save it to CSV/Excel files.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    df = fetch_sp500_table()
    if df is None:
        return

    output_dir = Path(__file__).resolve().parents[1] / "yfinance_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sp500_tickers_sectors.csv"

    df.to_csv(csv_path, index=False)

    print(f"Saved {csv_path} under {output_dir}")


@lru_cache(maxsize=1)
def fetch_sp500_table() -> Optional[pd.DataFrame]:
    html = fetch_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    if html is None:
        return None

    tables = pd.read_html(html, attrs={"id": "constituents"})
    df = tables[0][["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]]
    df.columns = ["Ticker", "Company", "Sector", "Industry"]
    return df


def fetch_html(url: str, headers: Optional[dict] = None) -> Optional[str]:
    try:
        response = requests.get(url, headers=headers or DEFAULT_HEADERS, timeout=15)
    except requests.RequestException as exc:
        print(f"Failed to fetch page: {exc}")
        return None

    if response.status_code == 403:
        print(
            "Wikipedia returned HTTP 403 (Forbidden). Try rerunning later, using a VPN, "
            "or updating the request headers to include a valid User-Agent."
        )
        return None

    if response.status_code != 200:
        print(f"Failed to fetch page: HTTP {response.status_code}")
        return None

    return response.text


def get_sp500_ticker_details(ticker: str) -> Optional[dict]:
    """
    Return metadata for a specific ticker from the S&P 500 table, if available.
    """
    df = fetch_sp500_table()
    if df is None:
        return None

    matches = df[df["Ticker"].str.upper() == ticker.upper()]
    if matches.empty:
        return None

    return matches.iloc[0].to_dict()


if __name__ == "__main__":
    main()

