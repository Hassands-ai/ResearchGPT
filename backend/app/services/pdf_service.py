from pypdf import PdfReader
from io import BytesIO
import re


def extract_text_from_pdf(file_data: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(file_data))
        parts = []

        for page in reader.pages:
            text = page.extract_text() or ""
            parts.append(text)

        full = "\n\n".join(parts)

        # Remove null bytes
        full = full.replace("\x00", "")

        # Fix missing spaces between words (common PDF issue)
        # e.g. "Animimportantoutcome" -> try to restore readability
        full = re.sub(r"([a-z])([A-Z])", r"\1 \2", full)
        full = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", full)
        full = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", full)
        full = re.sub(r"([.,;:!?])([A-Za-z])", r"\1 \2", full)
        full = re.sub(r"[ \t]+", " ", full)
        full = re.sub(r"\n{3,}", "\n\n", full)

        return full.strip()
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return ""