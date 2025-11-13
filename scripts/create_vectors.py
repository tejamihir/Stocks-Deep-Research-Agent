# NOTE: run these helpers to rebuild the Chromadb collections from the CSV snapshots.
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from chromadb import Client, PersistentClient
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
BALANCE_SHEET_CSV = DATA_DIR / "yfinance_outputs" / "balance_sheet.csv"
INCOME_STATEMENT_CSV = DATA_DIR / "yfinance_outputs" / "financials.csv"
CASHFLOW_STATEMENT_CSV = DATA_DIR / "yfinance_outputs" / "cashflow.csv"
TICKER_INFO_CSV = DATA_DIR / "yfinance_outputs" / "sp500_tickers_sectors.csv"
CHROMA_PATH = DATA_DIR / "chroma_db"
BATCH_SIZE = 5000
MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Low-level ingestion utilities
# ---------------------------------------------------------------------------
def _normalize_column(col: str) -> str:
    return (
        str(col)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("\n", "_")
    )


def _ingest_numeric_table(
    df: pd.DataFrame,
    collection,
    collection_name: str,
    ticker_column: str,
    model: SentenceTransformer,
    batch_size: int,
) -> int:
    documents: List[str] = []
    metadatas: List[dict] = []
    ids: List[str] = []

    for row_idx, row in df.iterrows():
        ticker = str(row[ticker_column]).strip()
        for col in df.columns:
            if col == ticker_column:
                continue
            value = row[col]
            if pd.isna(value):
                continue

            documents.append(f"{col}: {value}")
            metadatas.append(
                {
                    "ticker": ticker,
                    "column": col,
                    "value": str(value),
                    "row_index": int(row_idx),
                }
            )
            ids.append(f"{ticker}_{_normalize_column(col)}_{row_idx}")

    print(f"Total {collection_name} documents to process: {len(documents)}")
    try:
        collection.delete(where={"*": "*"})
        print(f"Cleared existing records from {collection_name} collection.")
    except Exception as exc:
        print(f"Warning: failed to clear {collection_name} collection: {exc}")

    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        batch_documents = documents[start:end]
        batch_metadatas = metadatas[start:end]
        batch_ids = ids[start:end]

        print(
            f"Encoding {collection_name} batch {start // batch_size + 1} "
            f"with {len(batch_documents)} documents..."
        )
        embeddings = model.encode(batch_documents).tolist()
        collection.add(
            ids=batch_ids,
            documents=batch_documents,
            embeddings=embeddings,
            metadatas=batch_metadatas,
        )

    return len(documents)


def ingest_balance_sheet_documents(
    collections: Dict[str, object],
    model: SentenceTransformer,
    csv_path: Path = BALANCE_SHEET_CSV,
    batch_size: int = BATCH_SIZE,
) -> int:
    if not csv_path.exists():
        raise FileNotFoundError(f"Balance sheet CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    ticker_col = next((c for c in df.columns if c.lower() == "ticker"), None)
    if ticker_col is None:
        raise ValueError(
            "No ticker column found. Ensure the balance sheet CSV has a 'ticker' column."
        )

    return _ingest_numeric_table(
        df,
        collection=collections["balance_sheet"],
        collection_name="balance_sheet",
        ticker_column=ticker_col,
        model=model,
        batch_size=batch_size,
    )


def ingest_income_statement_documents(
    collections: Dict[str, object],
    model: SentenceTransformer,
    csv_path: Path = INCOME_STATEMENT_CSV,
    batch_size: int = BATCH_SIZE,
) -> int:
    if not csv_path.exists():
        raise FileNotFoundError(f"Income statement CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    ticker_col = next((c for c in df.columns if c.lower() == "ticker"), None)
    if ticker_col is None:
        raise ValueError(
            "No ticker column found. Ensure the income statement CSV has a 'ticker' column."
        )

    return _ingest_numeric_table(
        df,
        collection=collections["financial_ratios"],
        collection_name="financial_ratios",
        ticker_column=ticker_col,
        model=model,
        batch_size=batch_size,
    )


def ingest_cashflow_documents(
    collections: Dict[str, object],
    model: SentenceTransformer,
    csv_path: Path = CASHFLOW_STATEMENT_CSV,
    batch_size: int = BATCH_SIZE,
) -> int:
    if not csv_path.exists():
        raise FileNotFoundError(f"Cashflow statement CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    ticker_col = next((c for c in df.columns if c.lower() == "ticker"), None)
    if ticker_col is None:
        raise ValueError(
            "No ticker column found. Ensure the cashflow statement CSV has a 'ticker' column."
        )

    return _ingest_numeric_table(
        df,
        collection=collections["cashflow_statement"],
        collection_name="cashflow_statement",
        ticker_column=ticker_col,
        model=model,
        batch_size=batch_size,
    )


def ingest_ticker_info_documents(
    collections: Dict[str, object],
    model: SentenceTransformer,
    csv_path: Path = TICKER_INFO_CSV,
) -> int:
    if not csv_path.exists():
        print(f"Ticker info CSV not found at {csv_path}. Skipping ticker_info ingestion.")
        return 0

    df = pd.read_csv(csv_path)
    required = ["Ticker", "Company", "Sector", "Industry"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Ticker info CSV missing required columns: {missing}")

    documents: List[str] = []
    metadatas: List[dict] = []
    ids: List[str] = []

    for _, row in df.iterrows():
        ticker = str(row["Ticker"]).strip()
        company = str(row["Company"]).strip()
        sector = str(row["Sector"]).strip()
        industry = str(row["Industry"]).strip()

        documents.append(
            f"{ticker} | Company: {company} | Sector: {sector} | Industry: {industry}"
        )
        metadatas.append(
            {
                "ticker": ticker,
                "company": company,
                "sector": sector,
                "industry": industry,
            }
        )
        ids.append(f"{ticker}_info")

    print(f"Encoding {len(documents)} ticker info records...")
    embeddings = model.encode(documents).tolist()
    collections["ticker_info"].add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(documents)


# ---------------------------------------------------------------------------
# Orchestrator helpers
# ---------------------------------------------------------------------------
def build_chroma_collections(
    client: Client,
    model: SentenceTransformer,
    batch_size: int = BATCH_SIZE,
    balance_sheet_csv: Path = BALANCE_SHEET_CSV,
    income_statement_csv: Path = INCOME_STATEMENT_CSV,
    cashflow_csv: Path = CASHFLOW_STATEMENT_CSV,
    ticker_info_csv: Path = TICKER_INFO_CSV,
) -> Dict[str, object]:
    collections = {
        "balance_sheet": client.get_or_create_collection("balance_sheet"),
        "financial_ratios": client.get_or_create_collection("financial_ratios"),
        "cashflow_statement": client.get_or_create_collection("cashflow_statement"),
        "ticker_info": client.get_or_create_collection("ticker_info"),
    }

    ingest_balance_sheet_documents(collections, model, balance_sheet_csv, batch_size)
    ingest_income_statement_documents(collections, model, income_statement_csv, batch_size)
    ingest_cashflow_documents(collections, model, cashflow_csv, batch_size)
    ingest_ticker_info_documents(collections, model, ticker_info_csv)

    return collections


def build_persistent_chroma(
    batch_size: int = BATCH_SIZE,
) -> Tuple[Client, Dict[str, object], SentenceTransformer]:
    client = PersistentClient(path=str(CHROMA_PATH))
    model = SentenceTransformer(MODEL_NAME)
    collections = build_chroma_collections(client, model, batch_size=batch_size)
    return client, collections, model
