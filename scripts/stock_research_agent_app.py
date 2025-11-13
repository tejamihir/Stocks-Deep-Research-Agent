"""
Streamlit front-end for the Stock Research Agent.

Run locally with:
    streamlit run scripts/stock_research_agent_app.py
"""

from pathlib import Path

import streamlit as st

# Ensure project modules are importable when the app is run via `streamlit run`
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_answers import rag_answer  # noqa: E402


def main() -> None:
    st.set_page_config(page_title="Stock Research Agent", layout="wide")
    st.title("📊 StockSage AI")

    st.markdown(
        "Enter a financial research question below. The application will gather relevant financial data and insights, then generate a comprehensive, informed answer."
    )

    with st.form(key="query_form", clear_on_submit=False):
        user_query = st.text_area(
            "Your question",
            height=150,
            placeholder="e.g., Provide an investment outlook for META for the next year",
        )
        submitted = st.form_submit_button("Analyze")

    if submitted:
        if not user_query.strip():
            st.warning("Please enter a question before submitting.")
            return

        with st.spinner("Gathering data and generating analysis..."):
            try:
                answer = rag_answer(user_query.strip())
            except Exception as exc:
                st.error(f"An error occurred while generating the answer: {exc}")
                return

        st.markdown("### 🧠 Application Response")
        st.markdown(answer)


if __name__ == "__main__":
    main()

