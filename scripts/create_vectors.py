import pandas as pd

from sentence_transformers import SentenceTransformer

from chromadb import PersistentClient



# Initialize Chroma client (using local persistence)

client = PersistentClient(path="/Users/mihirrao/mihir/Stocks-Deep-Research-Agent/data/chroma_db")



# Create (or get) a collection named 'financial_ratios'

collection = client.get_or_create_collection(name="cashflow_statement")



# Load CSV containing financial ratios (adjust file path)

csv_file = "/Users/mihirrao/mihir/Stocks-Deep-Research-Agent/data/yfinance_outputs/cashflow.csv"

df = pd.read_csv(csv_file)



# Initialize sentence transformer model for embeddings

model = SentenceTransformer('all-MiniLM-L6-v2')



# Find ticker column (case-insensitive)

ticker_col = None

for col in df.columns:

    if col.lower() in ['ticker']:

        ticker_col = col

        break

if ticker_col is None:

    raise ValueError("No ticker column found. Please ensure CSV has a 'ticker', 'symbol', 'stock', or 'ticker_symbol' column.")



# Structure: ticker >> column >> column value

# Prepare lists for documents, metadatas, and IDs

texts = []

metadatas = []

ids = []



# Iterate through each row (ticker)

for row_idx, row in df.iterrows():

    ticker = str(row[ticker_col]).strip()

    

    # For each column (excluding the ticker column), create a separate document

    for col in df.columns:

        if col == ticker_col:

            continue  # Skip ticker column itself

        

        column_value = row[col]

        

        # Skip NaN/None values

        if pd.isna(column_value):

            continue

        

        # Create document text: "Column Name: Column Value"

        doc_text = f"{col}: {column_value}"

        texts.append(doc_text)

        

        # Create metadata with hierarchy: ticker > column > value

        metadata = {

            "ticker": ticker,

            "column": col,

            "value": str(column_value),

            "row_index": int(row_idx)

        }

        metadatas.append(metadata)

        

        # Create ID: ticker_column_index

        # Format: {ticker}_{column_name}_{row_index}

        # Replace spaces/special chars in column name for cleaner IDs

        clean_col = col.replace(" ", "_").replace("/", "_").replace("-", "_")

        vector_id = f"{ticker}_{clean_col}_{row_idx}"

        ids.append(vector_id)



# Batch size for ChromaDB (to avoid max batch size error)
# ChromaDB has a max batch size limit (~5461), so we'll use a smaller safe batch size
BATCH_SIZE = 5000

print(f"Total documents to process: {len(texts)}")
print(f"Processing in batches of {BATCH_SIZE}...")

# Generate embeddings and add to collection in batches
for i in range(0, len(texts), BATCH_SIZE):
    batch_texts = texts[i:i+BATCH_SIZE]
    batch_metadatas = metadatas[i:i+BATCH_SIZE]
    batch_ids = ids[i:i+BATCH_SIZE]
    
    # Generate embeddings for this batch
    print(f"Generating embeddings for batch {i//BATCH_SIZE + 1} ({len(batch_texts)} documents)...")
    batch_embeddings = model.encode(batch_texts).tolist()
    
    # Add batch to collection
    print(f"Adding batch {i//BATCH_SIZE + 1} to collection...")
    collection.add(
        ids=batch_ids,
        documents=batch_texts,
        embeddings=batch_embeddings,
        metadatas=batch_metadatas
    )
    print(f"Batch {i//BATCH_SIZE + 1} completed.\n")

print(f"Successfully added {len(texts)} documents to the collection!")

# Note: PersistentClient automatically persists data, no need for client.persist()



def query_vectors(query_text, top_k=5):

    query_embedding = model.encode([query_text]).tolist()

    results = collection.query(

        query_embeddings=query_embedding,

        n_results=top_k

    )

    return results



if __name__ == "__main__":

    # Example usage: query similar financial entries

    query = "Total Assets"

    results = query_vectors(query, top_k=5)

    print("Top results for query:", query)

    print("-" * 60)

    

    # Results include 'ids', 'documents', 'metadatas', and 'distances'

    # Each result follows the hierarchy: ticker >> column >> value

    for i in range(len(results['documents'][0])):

        doc_id = results['ids'][0][i]

        doc = results['documents'][0][i]

        metadata = results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'][0] else {}

        distance = results['distances'][0][i] if results.get('distances') and results['distances'][0] else None

        

        print(f"\n{i+1}. ID: {doc_id}")

        print(f"   Hierarchy: {metadata.get('ticker', 'N/A')} >> {metadata.get('column', 'N/A')} >> {metadata.get('value', 'N/A')}")

        print(f"   Document: {doc}")

        if distance is not None:

            print(f"   Similarity Distance: {distance:.4f}")

        print("-" * 60)

