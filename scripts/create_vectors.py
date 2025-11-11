from pathlib import Path

from typing import List

import pandas as pd

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
BALANCE_SHEET_CSV = DATA_DIR / "yfinance_outputs" / "balance_sheet.csv"
INCOME_STATEMENT_CSV = DATA_DIR / "yfinance_outputs" / "financials.csv"
TICKER_INFO_CSV = DATA_DIR / "yfinance_outputs" / "sp500_tickers_sectors.csv"
CHROMA_PATH = DATA_DIR / "chroma_db"
BATCH_SIZE = 5000

model = SentenceTransformer("all-MiniLM-L6-v2")
client = PersistentClient(path=str(CHROMA_PATH))

balance_sheet_collection = client.get_or_create_collection(name="balance_sheet")
income_statement_collection = client.get_or_create_collection(name="financial_ratios")
ticker_info_collection = client.get_or_create_collection(name="ticker_info")


def ingest_balance_sheet_documents(
    csv_path: Path = BALANCE_SHEET_CSV, batch_size: int = BATCH_SIZE
) -> int:
    if not csv_path.exists():
        raise FileNotFoundError(f"Balance sheet CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    ticker_col = next((c for c in df.columns if c.lower() == "ticker"), None)
    if ticker_col is None:
        raise ValueError(
            "No ticker column found. Ensure the balance sheet CSV has a 'ticker' column."
        )

    documents: List[str] = []
    metadatas: List[dict] = []
    ids: List[str] = []

    for row_idx, row in df.iterrows():
        ticker = str(row[ticker_col]).strip()
        for col in df.columns:
            if col == ticker_col:
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
            clean_col = str(col).replace(" ", "_").replace("/", "_").replace("-", "_")
            ids.append(f"{ticker}_{clean_col}_{row_idx}")

    print(f"Total balance sheet documents to process: {len(documents)}")
    try:
        balance_sheet_collection.delete(where={"*": "*"})
        print("Cleared existing records from balance_sheet collection.")
    except Exception as exc:
        print(f"Warning: failed to clear balance_sheet collection: {exc}")

    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        batch_documents = documents[start:end]
        batch_metadatas = metadatas[start:end]
        batch_ids = ids[start:end]

        print(
            f"Encoding balance sheet batch {start // batch_size + 1} with {len(batch_documents)} documents..."
        )
        embeddings = model.encode(batch_documents).tolist()
        balance_sheet_collection.add(
            ids=batch_ids,
            documents=batch_documents,
            embeddings=embeddings,
            metadatas=batch_metadatas,
        )

    return len(documents)


def ingest_income_statement_documents(
    csv_path: Path = INCOME_STATEMENT_CSV, batch_size: int = BATCH_SIZE
) -> int:
    if not csv_path.exists():
        raise FileNotFoundError(f"Income statement CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    ticker_col = next((c for c in df.columns if c.lower() == "ticker"), None)
    if ticker_col is None:
        raise ValueError(
            "No ticker column found. Ensure the income statement CSV has a 'ticker' column."
        )

    documents: List[str] = []
    metadatas: List[dict] = []
    ids: List[str] = []

    for row_idx, row in df.iterrows():
        ticker = str(row[ticker_col]).strip()
        for col in df.columns:
            if col == ticker_col:
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
            clean_col = str(col).replace(" ", "_").replace("/", "_").replace("-", "_")
            ids.append(f"{ticker}_{clean_col}_{row_idx}")

    print(f"Total income statement documents to process: {len(documents)}")
    try:
        income_statement_collection.delete(where={"*": "*"})
        print("Cleared existing records from financial_ratios collection.")
    except Exception as exc:
        print(f"Warning: failed to clear financial_ratios collection: {exc}")

    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        batch_documents = documents[start:end]
        batch_metadatas = metadatas[start:end]
        batch_ids = ids[start:end]

        print(
            f"Encoding income statement batch {start // batch_size + 1} "
            f"with {len(batch_documents)} documents..."
        )
        embeddings = model.encode(batch_documents).tolist()
        income_statement_collection.add(
            ids=batch_ids,
            documents=batch_documents,
            embeddings=embeddings,
            metadatas=batch_metadatas,
        )

    return len(documents)


def ingest_ticker_info_documents(csv_path: Path = TICKER_INFO_CSV) -> int:
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
    ticker_info_collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(documents)


def query_balance_sheet_vectors(query_text: str, top_k: int = 5):
    query_embedding = model.encode([query_text]).tolist()
    return balance_sheet_collection.query(query_embeddings=query_embedding, n_results=top_k)


def main() -> None:
    balance_count = ingest_balance_sheet_documents()
    print(f"Ingested {balance_count} documents into balance_sheet collection.")

    income_statement_count = ingest_income_statement_documents()
    print(
        f"Ingested {income_statement_count} documents into financial_ratios collection."
    )

    ticker_count = ingest_ticker_info_documents()
    if ticker_count:
        print(f"Ingested {ticker_count} records into ticker_info.")

    # Example exploration helper (disabled by default)
    # query = "Total Assets"
    # results = query_balance_sheet_vectors(query, top_k=5)
    # print("Top results for query:", query)
    # print("-" * 60)
    # documents = results.get("documents", [[]])
    # if not documents or not documents[0]:
    #     print("No similar documents found.")
    #     return
    # metadatas = results.get("metadatas", [[]])
    # ids = results.get("ids", [[]])
    # distances = results.get("distances", [[]])
    # for index, doc in enumerate(documents[0], start=1):
    #     metadata = metadatas[0][index - 1] if metadatas and metadatas[0] else {}
    #     doc_id = ids[0][index - 1] if ids and ids[0] else "N/A"
    #     distance = distances[0][index - 1] if distances and distances[0] else None
    #     print(f"\n{index}. ID: {doc_id}")
    #     print(
    #         f"   Hierarchy: {metadata.get('ticker', 'N/A')} >> "
    #         f"{metadata.get('column', 'N/A')} >> {metadata.get('value', 'N/A')}"
    #     )
    #     print(f"   Document: {doc}")
    #     if distance is not None:
    #         print(f"   Similarity Distance: {distance:.4f}")
    #     print("-" * 60)


if __name__ == "__main__":
    main()
