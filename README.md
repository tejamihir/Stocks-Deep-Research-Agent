# Finnova AI – RAG‑powered Equity Research Assistant
Finnova AI is an intelligent financial research application that provides actionable, ticker‑level insights on U.S. equities, currently scoped to the S&P 500 universe. It uses a Retrieval‑Augmented Generation (RAG) pipeline to ground large language model responses in up‑to‑date market and fundamentals data, reducing hallucinations and enabling explainable outputs. 
Under the hood, the app orchestrates:
	•	Data ingestion & retrieval: Fetches real‑time and historical market data, fundamentals, and news via external APIs, then normalizes and indexes this data into a vector store for semantic retrieval over company‑specific context (e.g., price history, key ratios, business descriptions). 
	•	RAG workflow: For each user query, a retriever component selects the most relevant financial documents/snippets, which are then injected into the LLM prompt so that generated answers are explicitly grounded in retrieved evidence. 
	•	Analytics & aggregation layer: Implements Python-based analytics for return statistics, simple factor-style metrics, and comparative summaries across tickers (within the S&P 500 universe), which are surfaced as part of the model’s response. 
	•	Application layer: A Streamlit front end (or similar web UI) handles interactive querying, session state, and display of both the natural‑language answer and the underlying retrieved context (e.g., tables, charts, and source snippets). 
The codebase encapsulates the full stack of this workflow: API client modules for data acquisition, retrieval and vector indexing logic, RAG orchestration, and an interactive query interface for end users.
Link to Streamlit application-
