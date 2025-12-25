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
  
Analyze Apple stock │ ▼ ┌─────────────────────────────────────────────────────┐ │              Dynamic Data Fetch                     │ │  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │  │ Yahoo Finance   │  │    News API      │  │   Vector DB     │ │  │ -  Price/OHLCV   │  │ -  Headlines      │  │ -  Fin Statements│ │  │ -  Analyst Ratings│ │ -  Industry Outlook││ (Balance Sheet, │ │  └─────────────────┘  └──────────────────┘  │  Income, Cash)  │ │                                             └─────────────────┘ └─────────────────────────────────────────────────────┘ │ ▼ ┌─────────────────────────────────────────────────────┐ │                Semantic Retrieval                   │ │  Embed Query → Similarity Search → Top-K Context    │ └─────────────────────────────────────────────────────┘ │ ▼ ┌─────────────────────────────────────────────────────┐ │             LLM Generation                          │ │  Query + Multi-Source Context → Grounded Answer     │ └─────────────────────────────────────────────────────┘ │


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

