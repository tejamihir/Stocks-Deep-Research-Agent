# Finnova AI – RAG-powered Equity Research Assistant

Finnova AI is an intelligent financial research application that provides **actionable, ticker-level insights** on U.S. equities (currently scoped to S&P 500). It uses a **Retrieval-Augmented Generation (RAG)** pipeline to ground LLM responses in real-time market and fundamentals data, reducing hallucinations while enabling explainable outputs.

## Core Components

- **Data Ingestion & Retrieval**  
  Fetches real-time/historical market data, fundamentals, and news via external APIs. Normalizes and indexes into a **vector store** for semantic retrieval of company-specific context (price history, key ratios, business descriptions).

- **RAG Workflow**  
  Query → **Retriever** selects top-k relevant financial documents → Injects into LLM prompt → Generates grounded, evidence-based answers.

- **Analytics Layer**  
  Python-based computations for return statistics, factor-style metrics, and cross-ticker comparisons (S&P 500 scope).

- **Application Layer**  
  **Streamlit** frontend with interactive querying, session state management, and dual-display of NL answers + retrieved context (tables, charts, sources).

- **Technical Architecture**

  flowchart TD
    A[Analyze Apple stock]

    %% Layer 1: Dynamic Data Fetch
    A --> B[Dynamic Data Fetch]

    B --> B1[Yahoo Finance\nPrice / OHLCV]
    B --> B2[News API\nHeadlines]
    B --> B3[Vector DB\nFin. Statements\n(Balance Sheet, Income, Cash)]
    B --> B4[Analyst Ratings]
    B --> B5[Industry Outlook]

    %% Layer 2: Semantic Retrieval
    B --> C[Semantic Retrieval\nEmbed Query → Similarity Search → Top‑K Context]

    %% Layer 3: LLM Generation
    C --> D[LLM Generation\nQuery + Multi‑Source Context → Grounded Answer]



## Tech Stack
Frontend: Streamlit 
Backend: Python
Data: Vector DB (Chroma), 
APIs- Yahoo Finance, NewsAPI
Infra: Streamlit Cloud deployable and docker ready

## Live Demo
## 🔮 Future Roadmap
- Expand to full US market + international equities
- Complex user queries support- 'Is NVIDIA overvalued?', 'Top 5 stocks in IT Industry'
- Multi-modal RAG (SEC filings, earnings transcripts)

