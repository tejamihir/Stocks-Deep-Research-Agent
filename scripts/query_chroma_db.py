import argparse
from pathlib import Path

from chromadb import PersistentClient

REPO_ROOT = Path(__file__).resolve().parents[1]
CHROMA_PATH = REPO_ROOT / "data" / "chroma_db"


def list_columns(ticker: str, collection_name: str = "balance_sheet") -> None:
    client = PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(name=collection_name)

    query = collection.get(where={"ticker": ticker.upper()}, include=["metadatas"])
    metadatas = query.get("metadatas", [])

    if not metadatas:
        print(f"No records found for {ticker} in collection '{collection_name}'.")
        return

    # metadatas is typically a list containing a list of dicts
    records = []
    for entry in metadatas:
        if isinstance(entry, list):
            records.extend(entry)
        elif isinstance(entry, dict):
            records.append(entry)

    if not records:
        print(f"No metadata records found for {ticker}.")
        return

    unique_columns = {}
    for record in records:
        column = str(record.get("column", "unknown")).strip()
        value = record.get("value")
        unique_columns[column] = value

    print(f"\nColumns for {ticker} ({collection_name}):")
    for column, value in sorted(unique_columns.items()):
        print(f"  {column}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect columns available for a ticker in a ChromaDB collection."
    )
    parser.add_argument("ticker", help="Ticker symbol to inspect")
    parser.add_argument(
        "--collection",
        default="balance_sheet",
        help="ChromaDB collection name (default: balance_sheet)",
    )
    args = parser.parse_args()

    list_columns(args.ticker, args.collection)
