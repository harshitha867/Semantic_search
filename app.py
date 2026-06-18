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
# CACHE HEAVY OPERATIONS
# -------------------------------------------------

@st.cache_resource
def process_pdf(pdf_path):

    text = extract_text_from_pdf(pdf_path)

    chunks = chunk_text(text)

    embeddings = generate_embeddings(chunks)

    faiss_index = create_faiss_index(embeddings)

    return chunks, embeddings, faiss_index


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

with st.sidebar:

    st.header("📖 Project Information")

    st.markdown("""
### Technologies

- PyMuPDF
- Sentence Transformers
- FAISS
- Streamlit

### Features

- PDF Upload
- Semantic Search
- Vector Retrieval
- Analytics Dashboard
""")

    st.markdown("---")

    st.caption(
        "MSc Mini Project\n\nSemantic Document Retrieval System"
    )


# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.markdown("""
# 📚 Semantic PDF Search Engine

### Search documents using AI-powered semantic retrieval
""")


# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

query = st.text_input(
    "Ask anything from your document",
    placeholder="Example: What is normal distribution?"
)

search_button = st.button(
    "🔍 Search",
    use_container_width=True
)


# -------------------------------------------------
# PROCESS PDF
# -------------------------------------------------

if uploaded_file:

    pdf_path = f"uploads/{uploaded_file.name}"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ PDF Uploaded Successfully")

    # Page Count
    doc = fitz.open(pdf_path)
    page_count = len(doc)
    doc.close()

    # Cached Processing
    chunks, embeddings, faiss_index = process_pdf(pdf_path)

    # -------------------------------------------------
    # DASHBOARD
    # -------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "📄 Pages",
            page_count
        )

    with col2:
        st.metric(
            "🧩 Chunks",
            len(chunks)
        )

    with col3:
        st.metric(
            "🧠 Embedding Dimension",
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
            "⚡ Search Time",
            f"{retrieval_time:.4f} sec"
        )

        st.markdown("---")

        st.subheader("🔍 Search Results")

        for i, (result, distance) in enumerate(results, start=1):

            confidence = (1 / (1 + distance)) * 100

            st.markdown(f"### 📄 Result {i}")

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

            st.progress(
                float(min(confidence / 100, 1.0))
            )

            preview = result[:300] + "..."

            st.info(preview)

            with st.expander("📖 View Full Result"):
                st.write(result)

            st.divider()


# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown("---")

st.caption(
    "Built using Sentence Transformers, FAISS and Streamlit"
)