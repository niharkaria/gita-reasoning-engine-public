"""Render every page of the source PDF and OCR it with Tesseract (Gujarati).

Why this file exists:
    Phase 4 ingestion needs raw text before it can be parsed into verses/
    translations/commentaries. This script does exactly one job — turn the
    PDF into a single OCR'd text file with clear page-boundary markers —
    so the parser (parse_gita_text.py) can be developed and tested against
    a stable text file instead of re-running slow OCR on every iteration.

Usage:
    python ocr_pdf.py path/to/book.pdf output.txt [start_page] [end_page]

Notes:
    - dpi=300 chosen based on the sample test (page 159) that produced
      clean, readable output at that resolution. Lower DPI is faster but
      risks accuracy loss; not worth it for a one-time ingestion run.
    - Page markers use a form ("===PAGE 42===") deliberately unlikely to
      appear in real OCR'd text, so the parser can split on it reliably.
"""

import sys

import pytesseract
from pdf2image import convert_from_path

LANG = "guj"
DPI = 300


def ocr_pdf(pdf_path: str, output_path: str, start_page: int = 1, end_page: int | None = None) -> None:
    print(f"Rendering pages {start_page}-{end_page or 'end'} of {pdf_path}...")
    pages = convert_from_path(pdf_path, first_page=start_page, last_page=end_page, dpi=DPI)

    with open(output_path, "w", encoding="utf-8") as out:
        for i, page_image in enumerate(pages):
            page_number = start_page + i
            print(f"  OCR page {page_number}...")
            text = pytesseract.image_to_string(page_image, lang=LANG)
            out.write(f"===PAGE {page_number}===\n")
            out.write(text)
            out.write("\n")

    print(f"Done. Wrote OCR text for {len(pages)} pages to {output_path}")


if __name__ == "__main__":
    pdf_path = sys.argv[1]
    output_path = sys.argv[2]
    start_page = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    end_page = int(sys.argv[4]) if len(sys.argv) > 4 else None

    ocr_pdf(pdf_path, output_path, start_page, end_page)
