import argparse
from pprint import pprint

import yfinance as yf


def inspect_ticker(ticker: str) -> None:
    ticker = ticker.strip().upper()
    stock = yf.Ticker(ticker)

    # Attempt to pull the key data sources used in get_stock_price
    info = stock.info or {}
    fast_info = getattr(stock, "fast_info", {}) or {}

    print(f"\n=== Debugging {ticker} ===")
    print("Current price candidates:")
    print(f"  info['currentPrice']: {info.get('currentPrice')}")
    print(f"  info['regularMarketPrice']: {info.get('regularMarketPrice')}")
    print(f"  fast_info['lastPrice']: {fast_info.get('lastPrice')}")
    print(f"  fast_info['previousClose']: {fast_info.get('previousClose')}")

    # Show a few additional metrics that get_stock_price expects
    for key in [
        "longName",
        "shortName",
        "currency",
        "previousClose",
        "marketCap",
        "trailingEps",
        "pegRatio",
        "trailingPE",
        "forwardPE",
    ]:
        print(f"  info['{key}']: {info.get(key)}")

    # Grab a small sample of price history as a fallback reference
    try:
        history = stock.history(period="5d")["Close"]
        print("\nRecent closing prices (last 5d):")
        print(history.tail())
    except Exception as exc:
        print(f"Failed to retrieve history for {ticker}: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug Yahoo Finance ticker data")
    parser.add_argument("ticker", help="Ticker symbol to inspect (e.g., GDDY)")
    args = parser.parse_args()

    inspect_ticker(args.ticker)
