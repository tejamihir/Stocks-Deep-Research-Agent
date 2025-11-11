import io
import os

import warnings

import json
from typing import Optional

# Suppress warnings before importing libraries that may trigger them

warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL 1.1.1+.*")

warnings.filterwarnings("ignore", message=".*tokenizers.*")

# Set tokenizers parallelism environment variable to suppress warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import urllib3

from urllib.error import HTTPError

import pandas as pd
from pathlib import Path

from sentence_transformers import SentenceTransformer

from contextlib import redirect_stdout

from chromadb import PersistentClient

import openai

import yfinance as yf

from get_ANALYT_ESTIMATES import (
    fetch_news_newsapi,
    get_analyst_estimates_and_price_target,
    get_top_news_yahoo,
)

# Suppress urllib3/OpenSSL warnings

urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)



# Initialize OpenAI API key from environment or set it here

# Option 1: Set via environment variable (recommended)
# export OPENAI_API_KEY="your_api_key_here"

# Option 2: Set directly in code (not recommended for production, but useful for testing)
# Uncomment the line below and set your API key:
DIRECT_API_KEY = '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'  # Change to "your_api_key_here" if not using environment variable

# Initialize OpenAI client (modern API v1.0+)

api_key = os.getenv("OPENAI_API_KEY") or DIRECT_API_KEY

openai_client = openai.OpenAI(api_key=api_key)



# PART 1: Load ChromaDB and Embedding Model

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
BALANCE_SHEET_CSV = DATA_DIR / "yfinance_outputs" / "balance_sheet.csv"

client = PersistentClient(path=str(DATA_DIR / "chroma_db"))

collections = {

    "balance_sheet": client.get_collection(name="balance_sheet"),

    "financial_ratios": client.get_collection(name="financial_ratios"),

    "cashflow_statement": client.get_collection(name="cashflow_statement"),

}

try:
    ticker_info_collection = client.get_collection(name="ticker_info")
except Exception:
    ticker_info_collection = None

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

_balance_sheet_df: Optional[pd.DataFrame] = None


def load_balance_sheet_dataframe() -> pd.DataFrame:
    global _balance_sheet_df
    if _balance_sheet_df is None:
        if BALANCE_SHEET_CSV.exists():
            _balance_sheet_df = pd.read_csv(BALANCE_SHEET_CSV)
        else:
            _balance_sheet_df = pd.DataFrame()
    return _balance_sheet_df


def get_balance_sheet_metrics(ticker: str, metrics: set[str]) -> dict[str, Optional[str]]:
    collection = collections.get("balance_sheet")
    default_response = {metric: None for metric in metrics}
    if collection is None:
        return default_response

    normalized_targets = {
        metric.replace(" ", "_").strip().lower(): metric for metric in metrics
    }
    values = {metric: None for metric in metrics}

    try:
        result = collection.get(where={"ticker": ticker.upper()}, include=["metadatas"])
    except Exception:
        result = {"metadatas": []}

    metadatas = result.get("metadatas") or []
    rows = []
    for entry in metadatas:
        if isinstance(entry, list):
            rows.extend(entry)
        elif isinstance(entry, dict):
            rows.append(entry)

    debug_columns = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        column = row.get("column")
        if isinstance(column, str):
            debug_columns.append(column)
            normalized_column = column.replace(" ", "_").strip().lower()
            original_metric = normalized_targets.get(normalized_column)
            if original_metric and row.get("value") is not None:
                values[original_metric] = str(row["value"])

    print(f"DEBUG balance_sheet columns for {ticker}: {debug_columns}")

    # Compute any missing metrics directly from the raw CSV
    missing_metrics = {metric for metric, value in values.items() if value is None}
    if missing_metrics:
        df = load_balance_sheet_dataframe()
        if not df.empty:
            ticker_rows = df[df["ticker"].astype(str).str.upper() == ticker.upper()]
            if not ticker_rows.empty:
                row = ticker_rows.iloc[0]

                def _get_numeric(column_name: str) -> Optional[float]:
                    if column_name in row and pd.notna(row[column_name]):
                        try:
                            return float(row[column_name])
                        except (TypeError, ValueError):
                            return None
                    return None

                current_assets = _get_numeric("Current Assets")
                current_liabilities = _get_numeric("Current Liabilities")
                stockholders_equity = _get_numeric("Stockholders Equity")
                total_liabilities = _get_numeric("Total Liabilities Net Minority Interest")

                if (
                    "current_ratio" in missing_metrics
                    and current_assets is not None
                    and current_liabilities not in (None, 0)
                ):
                    try:
                        values["current_ratio"] = f"{current_assets / current_liabilities:.2f}"
                    except ZeroDivisionError:
                        values["current_ratio"] = None

                if "book_value" in missing_metrics and stockholders_equity is not None:
                    values["book_value"] = f"{stockholders_equity:.2f}"

                if (
                    "debt_to_equity_ratio" in missing_metrics
                    and total_liabilities is not None
                    and stockholders_equity not in (None, 0)
                ):
                    try:
                        values["debt_to_equity_ratio"] = f"{total_liabilities / stockholders_equity:.2f}"
                    except ZeroDivisionError:
                        values["debt_to_equity_ratio"] = None

    return values


def _extract_tickers_from_where(where_clause) -> set[str]:
    tickers: set[str] = set()
    if isinstance(where_clause, dict):
        if "ticker" in where_clause:
            ticker_value = where_clause["ticker"]
            if isinstance(ticker_value, str):
                tickers.add(ticker_value.upper())
            elif isinstance(ticker_value, dict):
                if "$in" in ticker_value and isinstance(ticker_value["$in"], list):
                    tickers.update(str(val).upper() for val in ticker_value["$in"])
        for value in where_clause.values():
            if isinstance(value, list):
                for item in value:
                    tickers.update(_extract_tickers_from_where(item))
            elif isinstance(value, dict):
                tickers.update(_extract_tickers_from_where(value))
    elif isinstance(where_clause, list):
        for item in where_clause:
            tickers.update(_extract_tickers_from_where(item))
    return tickers


def retrieve_across_collections(query, top_k=3, where={}):

    query_emb = embed_model.encode([query]).tolist()

    all_contexts = []
    print(where)

    balance_metrics = {"current_ratio", "book_value", "debt_to_equity_ratio"}
    encountered_tickers: set[str] = set()
    added_metric_entries: set[tuple[str, str]] = set()

    encountered_tickers.update(_extract_tickers_from_where(where))

    for name, col in collections.items():

        # Build query parameters
        query_params = {
            "query_embeddings": query_emb,
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        
        # Explicitly add where clause when provided
        if where:
            query_params["where"] = where
            
        # Execute query with explicit where clause
        results = col.query(**query_params)
        
        # Iterate through documents and metadatas together
        documents = results["documents"][0] if results["documents"] else []
        metadatas = (
            results["metadatas"][0]
            if results.get("metadatas") and results["metadatas"]
            else []
        )

        for i, doc in enumerate(documents):
            ticker = ""
            column = ""
            value = ""
            if i < len(metadatas) and metadatas[i]:
                metadata = metadatas[i]
                ticker = metadata.get("ticker", "")
                column = metadata.get("column", "")
                value = metadata.get("value", "")

            column_key = column.replace(" ", "_").strip().lower() if column else ""

            if name == "balance_sheet" and ticker:
                encountered_tickers.add(ticker)
                if column_key in balance_metrics:
                    all_contexts.append(
                        f"[{name}] Ticker: {ticker} | {column_key}: {value}"
                    )
                    continue

            if ticker:
                all_contexts.append(f"[{name}] Ticker: {ticker} | {doc}")
            else:
                all_contexts.append(f"[{name}] {doc}")

    # Ensure key balance-sheet metrics are present even if not retrieved
    metrics_summary: list[str] = []
    for ticker in encountered_tickers:
        metric_values = get_balance_sheet_metrics(ticker, balance_metrics)
        for metric, metric_value in metric_values.items():
            identifier = (ticker, metric)
            if metric_value is not None and identifier not in added_metric_entries:
                all_contexts.append(
                    f"[balance_sheet] Ticker: {ticker} | {metric}: {metric_value}"
                )
                added_metric_entries.add(identifier)
                metrics_summary.append(
                    f"{ticker} {metric.replace('_', ' ').title()}: {metric_value}"
                )

    print("All Contexts:", all_contexts, "\n")
    return all_contexts, metrics_summary



def generate_with_openai(prompt):
    

    response = openai_client.chat.completions.create(
        model="gpt-4o",

        messages=[{"role": "user", "content": prompt}],

        max_tokens=300,

        temperature=0.0,

    )

    return response.choices[0].message.content


def get_stock_price(ticker):
    """
    Fetch current stock price and basic info using Yahoo Finance API.
    Returns a formatted string with price information.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        #print("DEBUG info keys:", list(info.keys())[:10])
        # Get current price
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        
        # Get additional useful info
        company_name = info.get('longName') or info.get('shortName', ticker)
        currency = info.get('currency', 'USD')
        previous_close = info.get('previousClose')
        market_cap = info.get('marketCap')
        
        price_info = f"Current Price: {current_price} {currency}"
        
        if previous_close:
            change = current_price - previous_close if current_price and previous_close else None
            change_pct = ((change / previous_close) * 100) if change and previous_close else None
            if change_pct:
                price_info += f"\nPrevious Close: {previous_close} {currency}"
                price_info += f"\nChange: {change:+.2f} {currency} ({change_pct:+.2f}%)"
        
        if market_cap:
            market_cap_billions = market_cap / 1e9
            price_info += f"\nMarket Cap: ${market_cap_billions:.2f}B"

        trailing_eps = info.get("trailingEps")
        if trailing_eps is not None:
            price_info += f"\nTrailing EPS: {trailing_eps}"

        peg_ratio = info.get("pegRatio")
        if peg_ratio is not None:
            price_info += f"\nPEG Ratio: {peg_ratio}"

        pe_ratio = info.get("trailingPE") or info.get("forwardPE")
        if pe_ratio is not None:
            label = "Trailing P/E" if info.get("trailingPE") else "Forward P/E"
            price_info += f"\n{label}: {pe_ratio}"
       # print("DEBUG price_info:", price_info)
        return f"{company_name} ({ticker}):\n{price_info}"
        
    except HTTPError:
        return f"No price data available for {ticker} (symbol not found or unavailable)."
    except Exception as e:
        return f"Error fetching price for {ticker}: {str(e)}"


def get_yahoo_news_section(ticker: str) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        get_top_news_yahoo(ticker, summarize=True)
    output = buffer.getvalue().strip()
    buffer.close()
    return output


def get_analyst_estimates_section(ticker: str) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        get_analyst_estimates_and_price_target(ticker)
    output = buffer.getvalue().strip()
    buffer.close()
    return output


def get_newsapi_section(
    query: str,
    summarize: bool = True,
    summarize_outlook: bool = False,
    page_size: int = 5,
) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        fetch_news_newsapi(
            query=query,
            summarize=summarize,
            summarize_outlook=summarize_outlook,
            page_size=page_size,
        )
    output = buffer.getvalue().strip()
    buffer.close()
    return output


def get_ticker_classification(ticker: str) -> Optional[dict]:
    if ticker_info_collection is None:
        return None

    try:
        result = ticker_info_collection.get(
            where={"ticker": ticker.upper()}, include=["metadatas"]
        )
    except Exception:
        return None

    metadatas = result.get("metadatas")
    if not metadatas:
        return None

    if isinstance(metadatas, dict):
        return metadatas  # unexpected but handle

    if isinstance(metadatas, list):
        for entry in metadatas:
            if isinstance(entry, dict):
                return entry
            if isinstance(entry, list):
                for inner in entry:
                    if isinstance(inner, dict):
                        return inner

    return None

    return None



def rag_answer(query):
    parsed = extract_metadata_with_llm(query)

    where = build_metadata_filter(parsed)

    # Check if user is asking for stock price
    price_context = ""
    tickers = parsed.get("tickers", [])
    if tickers:
        price_info_list = []
        for ticker in tickers:
            price_info = get_stock_price(ticker)
            price_info_list.append(price_info)
        price_context = "\n\nReal-time Stock Prices:\n" + "\n\n".join(price_info_list) + "\n"
    #print("Price Conext:",price_context,"\n")
    news_sections = []
    analyst_sections = []
    newsapi_sections = []
    classification_lines = []
    if tickers:
        for ticker in tickers:
            news_output = get_yahoo_news_section(ticker)
            if news_output:
                news_sections.append(news_output)
            analyst_output = get_analyst_estimates_section(ticker)
            if analyst_output:
                analyst_sections.append(f"Analyst Estimates for {ticker}:\n{analyst_output}")
            classification = get_ticker_classification(ticker)
            if classification:
                sector = classification.get("sector")
                industry = classification.get("industry")
                # Sector summary intentionally disabled; keeping industry coverage only.
                if industry:
                    industry_news = get_newsapi_section(
                        query=industry+" in the next 10 years", summarize=False, summarize_outlook=True
                    )
                    if industry_news:
                        newsapi_sections.append(
                            f"NewsAPI Industry Coverage for {ticker} ({industry}):\n{industry_news}"
                        )
                classification_lines.append(
                    f"{classification.get('ticker', ticker)}: "
                    f"Sector - {sector or 'Unknown'}; "
                    f"Industry - {industry or 'Unknown'} "
                    f"| Company - {classification.get('company', 'Unknown')}"
                )

    contexts, metrics_summary = retrieve_across_collections(query, top_k=3, where=where)
    combined_context = "\n".join(contexts)
    print("Combined Context:", combined_context, "\n")

    metrics_block = ""
    if metrics_summary:
        metrics_block = "Balance Sheet Metrics:\n" + "\n".join(metrics_summary)

    context_parts = []
    if price_context.strip():
        context_parts.append(price_context.strip())
    if combined_context.strip():
        context_parts.append(combined_context.strip())
    if metrics_block:
        context_parts.append(metrics_block)

    full_context = "\n\n".join(context_parts)
    print("Full Context:", full_context, "\n")
    prompt = f"Context:\n{full_context}\n\nQuestion: {query}\n\nAnswer:"
    generated_text = generate_with_openai(prompt)

    if news_sections:
        generated_text += "\n\nYahoo Finance News:\n" + "\n\n".join(news_sections)
    if analyst_sections:
        generated_text += "\n\nAnalyst Targets & Recommendations:\n" + "\n\n".join(analyst_sections)
    if newsapi_sections:
        generated_text += "\n\nNewsAPI Headlines:\n" + "\n\n".join(newsapi_sections)
    if classification_lines:
        generated_text += "\n\nTicker Classification:\n" + "\n".join(classification_lines)

    return generated_text


def extract_metadata_with_llm(user_query: str):
    """
    Use a lightweight LLM to extract structured metadata (tickers, metrics, years)
    from a natural-language financial query.
    """

    system_prompt = """
    You are a financial text parser that extracts structured fields from user queries.
    Always return valid JSON in this format:

    {
      "tickers": ["GOOG", "AAPL"],
      "metrics": ["Net Income", "Total Assets"],
      "years": [2023]
    }

    Rules:
    - Use uppercase for tickers.
    - If no year mentioned, return [].
    - If no metrics explicitly mentioned, return [].
    - Do not explain or add text outside JSON.
    """

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",  # ✅ lightweight and cheap
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        max_tokens=100
    )

    # Extract text
    raw_output = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        print("⚠️ LLM returned invalid JSON, defaulting to empty fields.")
        parsed = {"tickers": [], "metrics": [], "years": []}

    return parsed


def build_metadata_filter(parsed):
    """
    Build a ChromaDB 'where' filter based on extracted fields.
    Supports multiple tickers and metrics.
    ChromaDB requires $and operator when combining multiple conditions.
    """

    conditions = []
    tickers = parsed.get("tickers", [])
    metrics = parsed.get("metrics", [])
    years = parsed.get("years", [])

    if tickers:
        if len(tickers) == 1:
            conditions.append({"ticker": tickers[0]})
        else:
            conditions.append({"ticker": {"$in": tickers}})

    if metrics:
        if len(metrics) == 1:
            conditions.append({"column": metrics[0]})
        else:
            conditions.append({"column": {"$in": metrics}})

    if years:
        if len(years) == 1:
            conditions.append({"fiscal_year": years[0]})
        else:
            conditions.append({"fiscal_year": {"$in": years}})

    # If only one condition, return it directly
    if len(conditions) == 1:
        return conditions[0]
    # If multiple conditions, wrap in $and
    elif len(conditions) > 1:
        return {"$and": conditions}
    # If no conditions, return empty dict
    else:
        return {}

if __name__ == "__main__":

    user_query = input("Enter your query: ")

    print(f"\nUser Query: {user_query}")

    rag_response = rag_answer(user_query)

    print(rag_response)

