import pdfplumber
import PyPDF2
import io

def extract_text_from_pdf(uploaded_file):
    """
    Try pdfplumber first (more accurate),
    fall back to PyPDF2 if it fails.
    """
    text = ""

    # Method 1: pdfplumber (preferred)
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            return text.strip()
    except Exception as e:
        print(f"pdfplumber failed: {e}")

    # Method 2: PyPDF2 fallback
    try:
        uploaded_file.seek(0)  # reset file pointer
        reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"PyPDF2 also failed: {e}")
        return ""