import os

import warnings

# Suppress warnings before importing libraries that may trigger them

warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL 1.1.1+.*")

warnings.filterwarnings("ignore", message=".*tokenizers.*")

# Set tokenizers parallelism environment variable to suppress warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import urllib3

import pandas as pd

from sentence_transformers import SentenceTransformer

from chromadb import PersistentClient

import openai

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

client = PersistentClient(path="/Users/mihirrao/mihir/Stocks-Deep-Research-Agent/data/chroma_db")

collections = {

    "balance_sheet": client.get_collection(name="balance_sheet"),

    "financial_ratios": client.get_collection(name="financial_ratios"),

    "cashflow_statement": client.get_collection(name="cashflow_statement"),

}

embed_model = SentenceTransformer("all-MiniLM-L6-v2")



def retrieve_across_collections(query, top_k=3):

    query_emb = embed_model.encode([query]).tolist()

    all_contexts = []

    for name, col in collections.items():

        results = col.query(query_embeddings=query_emb, n_results=top_k)

        for doc in results["documents"][0]:

            all_contexts.append(f"[{name}] {doc}")

    return all_contexts



def generate_with_openai(prompt):

    response = openai_client.chat.completions.create(

        model="gpt-3.5-turbo",

        messages=[{"role": "user", "content": prompt}],

        max_tokens=300,

        temperature=0.0,

    )

    return response.choices[0].message.content



def rag_answer(query):

    contexts = retrieve_across_collections(query, top_k=6)

    combined_context = "\n".join(contexts)

    prompt = f"Context:\n{combined_context}\n\nQuestion: {query}\n\nAnswer:"
    print(prompt)
    generated_text = generate_with_openai(prompt)

    return generated_text



if __name__ == "__main__":

    user_query = input("Enter your query: ")

    print(f"\nUser Query: {user_query}")

    rag_response = rag_answer(user_query)

    print("\nGenerated RAG Answer:\n", rag_response)

