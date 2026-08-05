"""Quick test: OCR a single page of the Gujarati PDF with Tesseract.

Usage:
    python test_ocr.py path/to/your.pdf 5
    (5 = page number to test, 1-indexed; pick a page with real body text,
    not a cover/blank page)
"""

import sys

import pytesseract
from pdf2image import convert_from_path

pdf_path = sys.argv[1]
page_number = int(sys.argv[2]) if len(sys.argv) > 2 else 1

print(f"Rendering page {page_number} of {pdf_path} to an image...")
pages = convert_from_path(pdf_path, first_page=page_number, last_page=page_number, dpi=300)
page_image = pages[0]

page_image.save("test_page.png")
print("Saved rendered page as test_page.png — open it to see what Tesseract is reading.")

print("\nRunning Tesseract OCR (Gujarati)...\n")
text = pytesseract.image_to_string(page_image, lang="guj")

print("--- OCR OUTPUT ---")
print(text)
print("--- END ---")
