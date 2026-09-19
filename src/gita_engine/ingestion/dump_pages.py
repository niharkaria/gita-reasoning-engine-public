"""Print raw OCR text for a specific list of pages, so we can inspect all
remaining problem spots in one pass instead of many separate greps."""

import re

TARGET_PAGES = {117, 133, 176, 179, 209, 210, 251, 68, 77, 90, 91, 105, 106, 114, 120, 126, 127, 241, 262}

with open("data/raw/full_book.txt", encoding="utf-8") as f:
    text = f.read()
pages = re.split(r"(===PAGE \d+===)", text)

current_page = None
for chunk in pages:
    m = re.match(r"===PAGE (\d+)===", chunk)
    if m:
        current_page = int(m.group(1))
        if current_page in TARGET_PAGES:
            print(f"\n{'=' * 20} PAGE {current_page} {'=' * 20}")
    elif current_page in TARGET_PAGES:
        print(chunk)
