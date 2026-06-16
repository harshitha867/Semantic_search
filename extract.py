import fitz
from chunking import chunk_text
from embeddings import generate_embeddings


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

    else:
        print("Failed to extract text from PDF.")