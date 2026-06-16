import fitz
from chunking import chunk_text
from embeddings import generate_embeddings
from search import semantic_search


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    """

    text = ""

    try:
        doc = fitz.open(pdf_path)

        for page in doc:
            text += page.get_text()

        doc.close()

        return text

    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return None


if __name__ == "__main__":

    pdf_path = "uploads/sample.pdf"

    extracted_text = extract_text_from_pdf(pdf_path)

    if extracted_text:

        # Create chunks
        chunks = chunk_text(extracted_text)

        # Generate embeddings
        embeddings = generate_embeddings(chunks)

        print(f"\nTotal Chunks: {len(chunks)}")

        print("\nEmbedding Shape:")
        print(embeddings.shape)

        print("\nFirst Chunk:\n")
        print(chunks[0])

        print("\nWords in First Chunk:")
        print(len(chunks[0].split()))

        print("\nWords in Last Chunk:")
        print(len(chunks[-1].split()))

                # Semantic Search Test
        query = "What is normal distribution?"

        results = semantic_search(
            query,
            chunks,
            embeddings,
            top_k=3
        )

        print("\nQuery:")
        print(query)

        print("\nTop 3 Results:\n")

        for i, (result, score) in enumerate(results, start=1):

            print(f"\nResult {i}")
            print(f"Similarity Score: {score:.4f}")

            print("\nRetrieved Chunk:\n")
            print(result[:500])

            print("\n" + "-" * 80)

    else:
        print("Failed to extract text from PDF.")