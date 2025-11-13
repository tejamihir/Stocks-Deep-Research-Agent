"""
Streamlit Cloud entrypoint.

Uses the deployment-ready rag_answer (in-memory Chroma rebuild) and
exposes simple form input UI. Run locally with:

    streamlit run scripts/streamlit_cloud.py
"""

from pathlib import Path
import sys

import streamlit as st

# Ensure project modules are importable when run via `streamlit run`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_answers_deployment import rag_answer


@st.cache_resource
def init_pipeline() -> None:
    """
    Trigger initialization of the in-memory Chroma pipeline once.

    The deployment module reconstructs the database lazily the first time
    rag_answer is invoked, but caching ensures Streamlit Cloud doesn’t repeat
    the expensive rebuild on every rerun.
    """
    # Simply call rag_answer with a harmless empty string to force initialization.
    # The function short-circuits when no valid tickers are detected.
    rag_answer("")


def main() -> None:
    st.set_page_config(page_title="StockSage AI", layout="wide")
    st.title("📊 StockSage AI")

    st.markdown(
        "Enter a financial research question below. The application will gather "
        "relevant financial data and insights, then generate a comprehensive, "
        "informed answer."
    )

    # Ensure pipeline is ready (first run only).
    init_pipeline()

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

