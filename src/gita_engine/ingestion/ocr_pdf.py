"""Render every page of the source PDF and OCR it with Tesseract (Gujarati).

Why this file exists:
    Phase 4 ingestion needs raw text before it can be parsed into verses/
    translations/commentaries. This script does exactly one job — turn the
    PDF into a single OCR'd text file with clear page-boundary markers —
    so the parser (parse_gita_text.py) can be developed and tested against
    a stable text file instead of re-running slow OCR on every iteration.

Usage:
    python ocr_pdf.py path/to/book.pdf output.txt [start_page] [end_page]

    Safe to re-run after a crash/interrupt: it automatically resumes from
    the last successfully written page in output.txt rather than starting
    over, as long as you pass the same output.txt path.

Notes:
    - dpi=300 chosen based on the sample test (page 159) that produced
      clean, readable output at that resolution. Lower DPI is faster but
      risks accuracy loss; not worth it for a one-time ingestion run.
    - Page markers use a form ("===PAGE 42===") deliberately unlikely to
      appear in real OCR'd text, so the parser can split on it reliably.
    - Pages are rendered and OCR'd ONE AT A TIME, not all loaded into
      memory at once. Rendering 320 pages simultaneously at 300 DPI can
      use several GB of RAM, which can crash/freeze lower-memory machines
      (e.g. 16GB with other applications open). Processing one page at a
      time keeps peak memory usage to roughly one page's worth, regardless
      of how many total pages are being OCR'd.
    - Output is flushed to disk after every single page, so a crash loses
      at most the page in progress, not the whole run.
"""

import sys
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path

LANG = "guj"
DPI = 300


def _last_completed_page(output_path: str) -> int | None:
    """Scan an existing output file for the highest page number already
    written, so a re-run can resume from there instead of starting over."""
    path = Path(output_path)
    if not path.exists():
        return None

    last_page = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("===PAGE ") and line.rstrip().endswith("==="):
                try:
                    last_page = int(line.strip().removeprefix("===PAGE ").removesuffix("==="))
                except ValueError:
                    continue
    return last_page


def ocr_pdf(pdf_path: str, output_path: str, start_page: int = 1, end_page: int | None = None) -> None:
    resume_from = _last_completed_page(output_path)
    if resume_from is not None and resume_from >= start_page:
        print(f"Found existing progress in {output_path} up to page {resume_from}.")
        print(f"Resuming from page {resume_from + 1} instead of {start_page}.")
        start_page = resume_from + 1
        file_mode = "a"
    else:
        file_mode = "w"

    if end_page is not None and start_page > end_page:
        print(f"Nothing to do — already OCR'd through page {resume_from}.")
        return

    print(f"OCR'ing pages {start_page}-{end_page or 'end'} of {pdf_path}, one page at a time...")

    with open(output_path, file_mode, encoding="utf-8") as out:
        page_number = start_page
        while end_page is None or page_number <= end_page:
            # Render exactly ONE page per iteration — keeps memory usage
            # roughly constant regardless of total page count.
            pages = convert_from_path(
                pdf_path, first_page=page_number, last_page=page_number, dpi=DPI
            )
            if not pages:
                break  # ran past the end of the document

            print(f"  OCR page {page_number}...")
            text = pytesseract.image_to_string(pages[0], lang=LANG)
            out.write(f"===PAGE {page_number}===\n")
            out.write(text)
            out.write("\n")
            out.flush()  # persist immediately so a crash doesn't lose progress

            page_number += 1

    print(f"Done. OCR text written through page {page_number - 1} to {output_path}")


if __name__ == "__main__":
    pdf_path = sys.argv[1]
    output_path = sys.argv[2]
    start_page = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    end_page = int(sys.argv[4]) if len(sys.argv) > 4 else None

    ocr_pdf(pdf_path, output_path, start_page, end_page)
