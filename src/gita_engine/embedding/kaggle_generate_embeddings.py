"""Generate BGE-M3 embeddings for exported verse text. RUN THIS ON KAGGLE,
not locally — BGE-M3 (560M params) needs a GPU to encode ~1200 texts in a
reasonable time; your laptop has none.

Why this file exists:
    Bridges the local export (export_for_embedding.py) to the local import
    (import_embeddings.py). This is the one piece of the pipeline that
    genuinely needs Kaggle's free GPU, per the Phase 0 architecture
    decision (see docs/architecture/).

How to use on Kaggle:
    1. Create a new Kaggle Notebook.
    2. Settings -> Accelerator -> GPU (T4 x2 or P100 — either is fine).
    3. Upload embedding_export.jsonl (produced locally by
       export_for_embedding.py) as a Kaggle Dataset, attach it to the
       notebook (it will appear under /kaggle/input/<dataset-name>/).
    4. Paste this script into a notebook cell (or upload as a .py and
       %run it), update INPUT_PATH below to match your dataset's actual
       path, and run all cells.
    5. Download the output file (embedding_output.jsonl) from
       /kaggle/working/ once it finishes.
    6. Bring it back to your laptop and run:
           python import_embeddings.py embedding_output.jsonl

Output format (one JSON object per line, mirrors the input plus embedding):
    {"table": "translation", "id": 123, "embedding": [0.0123, -0.0456, ...]}
"""

import json

# Adjust this to match wherever Kaggle mounts your uploaded dataset —
# check the "Input" panel in the Kaggle notebook UI for the exact path.
INPUT_PATH = "/kaggle/input/gita-embedding-export/embedding_export.jsonl"
OUTPUT_PATH = "/kaggle/working/embedding_output.jsonl"
BATCH_SIZE = 32

# BGE-M3: install if not already present in the Kaggle environment.
# !pip install -q sentence-transformers

from sentence_transformers import SentenceTransformer  # noqa: E402

print("Loading BAAI/bge-m3 (this downloads ~2.2GB the first time)...")
model = SentenceTransformer("BAAI/bge-m3", device="cuda")

records = []
with open(INPUT_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"Loaded {len(records)} texts to embed.")

texts = [r["text"] for r in records]
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    normalize_embeddings=True,  # required for cosine similarity to behave correctly
    show_progress_bar=True,
)

with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
    for record, embedding in zip(records, embeddings, strict=True):
        out.write(
            json.dumps(
                {
                    "table": record["table"],
                    "id": record["id"],
                    "embedding": embedding.tolist(),
                }
            )
            + "\n"
        )

print(f"Done. Wrote {len(records)} embeddings to {OUTPUT_PATH}")
print("Download this file from the Kaggle 'Output' panel, then run")
print("import_embeddings.py with it locally.")
