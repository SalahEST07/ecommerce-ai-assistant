from pathlib import Path
import sys
import asyncio
import traceback
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.backend import load_assistant

st.set_page_config(
    page_title="E-commerce AI Assistant",
    page_icon="🛍️",
    layout="wide"
)

@st.cache_resource
def get_assistant():
    return load_assistant()

st.title("🛍️ E-commerce AI Assistant")

st.markdown("""
Ask questions about shipping, returns, payments,
customer support, and e-commerce statistics.
""")

st.divider()

question = st.text_area(
    "Enter your question:",
    placeholder="Example: How long does standard delivery take?",
    height=100
)

if st.button("Submit question", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Processing your question..."):
                assistant = get_assistant()
                result = asyncio.run(
                    assistant.ainvoke({
                        "question": question.strip()
                    })
                )

            if result.get("error"):
                st.error(result["error"])
            else:
                st.subheader("Assistant response")
                st.markdown(result.get("answer", "No answer generated."))

                st.divider()

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Detected route**")
                    st.info(result.get("route", "UNKNOWN"))

                with col2:
                    st.markdown("**Sources**")
                    for source in result.get("sources") or []:
                        st.write(f"- {source}")

                generated_sql = result.get("generated_sql")
                if generated_sql:
                    st.subheader("Generated SQL")
                    st.code(generated_sql, language="sql")

        except Exception as e:
            st.error("An error occurred while processing your question.")

            error_details = traceback.format_exc()
            st.code(error_details, language="text")

            if isinstance(e, BaseExceptionGroup):
                for index, sub_exception in enumerate(e.exceptions, 1):
                    st.error(f"Nested error {index}: {type(sub_exception).__name__}")
                    st.code(
                        "".join(traceback.format_exception(sub_exception)),
                        language="text"
                    )
