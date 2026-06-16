import fitz

def extract_text_from_pdf(pdf_path):
    text = ""

    doc = fitz.open(pdf_path)

    for page in doc:
        text += page.get_text()

    return text


if __name__ == "__main__":
    pdf_path = "uploads/sample.pdf"

    extracted_text = extract_text_from_pdf(pdf_path)

    print(extracted_text[:1000])