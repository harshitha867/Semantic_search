import streamlit as st
import fitz
import time

from extract import extract_text_from_pdf
from chunking import chunk_text
from embeddings import generate_embeddings
from faiss_search import create_faiss_index
from faiss_search import search_faiss


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Semantic PDF Search Engine",
    page_icon="📚",
    layout="wide"
)


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

with st.sidebar:

    st.header("📖 Project Information")

    st.write("""
    Semantic PDF Search Engine

    Technologies Used:
    - PyMuPDF
    - Sentence Transformers
    - FAISS
    - Streamlit

    Features:
    - PDF Upload
    - Semantic Search
    - Vector Retrieval
    - Analytics Dashboard
    """)


# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title("📚 Semantic PDF Search Engine")

st.markdown(
    "Upload a PDF and perform semantic search using vector embeddings."
)


# -------------------------------------------------
# PDF UPLOAD
# -------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

query = st.text_input(
    "Enter your search query"
)

search_button = st.button("🔍 Search")


# -------------------------------------------------
# PROCESS PDF
# -------------------------------------------------

if uploaded_file:

    pdf_path = f"uploads/{uploaded_file.name}"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ PDF Uploaded Successfully")

    # PDF Statistics
    doc = fitz.open(pdf_path)
    page_count = len(doc)
    doc.close()

    # Extract Text
    text = extract_text_from_pdf(pdf_path)

    # Chunking
    chunks = chunk_text(text)

    # Embeddings
    embeddings = generate_embeddings(chunks)

    # FAISS Index
    faiss_index = create_faiss_index(embeddings)

    # -------------------------------------------------
    # ANALYTICS DASHBOARD
    # -------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Pages",
            page_count
        )

    with col2:
        st.metric(
            "Chunks",
            len(chunks)
        )

    with col3:
        st.metric(
            "Embedding Dimension",
            embeddings.shape[1]
        )

    st.success(
        f"PDF processed successfully. {len(chunks)} chunks generated."
    )

    # -------------------------------------------------
    # SEARCH
    # -------------------------------------------------

    if search_button and query:

        start_time = time.time()

        results = search_faiss(
            query,
            chunks,
            faiss_index,
            top_k=3
        )

        end_time = time.time()

        retrieval_time = end_time - start_time

        st.metric(
            "Search Time",
            f"{retrieval_time:.4f} sec"
        )

        st.subheader("🔍 Search Results")

        for i, (result, distance) in enumerate(results, start=1):

            confidence = (1 / (1 + distance)) * 100

            st.markdown(f"## Result {i}")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Distance",
                    f"{distance:.4f}"
                )

            with col2:
                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            preview = result[:300] + "..."

            st.write(preview)

            with st.expander("View Full Result"):
                st.write(result)

            st.divider()