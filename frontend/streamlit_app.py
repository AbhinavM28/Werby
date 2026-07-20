"""Werby Streamlit frontend.

Deliberately a *pure API client*: it imports nothing from ``app/``. It talks
to the backend over HTTP exactly like any other consumer would. This proves
the API contract works and means the frontend can be swapped for React later
with zero backend changes.

Run:  streamlit run frontend/streamlit_app.py
"""

import os

import httpx
import streamlit as st

API_BASE = os.getenv("WERBY_API_URL", "http://localhost:8000/api/v1")
TIMEOUT = httpx.Timeout(120.0)  # LLM calls can be slow; don't time out early

st.set_page_config(page_title="Werby — AI Engineering Copilot", page_icon="🏗️")


def api_get(path: str) -> dict:
    response = httpx.get(f"{API_BASE}{path}", timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def api_post(path: str, **kwargs) -> dict:
    response = httpx.post(f"{API_BASE}{path}", timeout=TIMEOUT, **kwargs)
    if response.status_code >= 400:
        detail = response.json().get("detail", response.text)
        raise RuntimeError(detail)
    return response.json()


# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.title("🏗️ Werby")
    st.caption("AI Engineering Copilot")

    try:
        health = api_get("/health")
        st.success(f"API online · v{health['version']} · {health['environment']}")
    except Exception:
        st.error("API offline — start the backend first:\n`make run`")
        st.stop()

    st.divider()
    st.subheader("📄 Ingest documentation")
    uploaded = st.file_uploader(
        "Upload a manual, SOP, or spec sheet",
        type=["pdf", "txt", "md"],
    )
    if uploaded and st.button("Ingest", use_container_width=True):
        with st.spinner(f"Ingesting {uploaded.name}..."):
            try:
                result = api_post(
                    "/documents",
                    files={"file": (uploaded.name, uploaded.getvalue())},
                )
                st.success(
                    f"Ingested **{result['document']}** — "
                    f"{result['chunks_created']} chunks"
                )
            except RuntimeError as err:
                st.error(str(err))

    st.divider()
    st.subheader("📚 Corpus")
    try:
        stats = api_get("/documents")
        st.metric("Total chunks", stats["total_chunks"])
        for doc in stats["documents"]:
            st.text(f"• {doc}")
    except Exception:
        st.caption("Could not load corpus stats.")

# ------------------------------------------------------------------- chat --
st.header("Ask your documentation")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for source in message.get("sources", []):
            with st.expander(
                f"📎 {source['document']} · chunk {source['chunk_index']} "
                f"· score {source['score']:.2f}"
            ):
                st.text(source["excerpt"])

if question := st.chat_input("e.g. What is the rated load of the AS/RS crane?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and reasoning..."):
            try:
                result = api_post("/query", json={"question": question})
                st.markdown(result["answer"])
                st.caption(
                    f"{result['model']} · {result['retrieved_chunks']} chunks "
                    f"· {result['latency_ms']} ms"
                )
                for source in result["sources"]:
                    with st.expander(
                        f"📎 {source['document']} · chunk {source['chunk_index']} "
                        f"· score {source['score']:.2f}"
                    ):
                        st.text(source["excerpt"])
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    }
                )
            except RuntimeError as err:
                st.error(str(err))
