import fitz
from chunking import chunk_text

def extract_text_from_pdf(pdf_path):
    text = ""

    doc = fitz.open(pdf_path)

    for page in doc:
        text += page.get_text()

    return text


if __name__ == "__main__":

    pdf_path = "uploads/sample.pdf"

    extracted_text = extract_text_from_pdf(pdf_path)

    chunks = chunk_text(extracted_text)

    print(f"\nTotal Chunks: {len(chunks)}")

    print("\nFirst Chunk:\n")
    print(chunks[0])

    print("\nWords in First Chunk:")
    print(len(chunks[0].split()))

    print("\nWords in Last Chunk:")
    print(len(chunks[-1].split()))