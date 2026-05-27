"""
Download multilingual models file-by-file to avoid HF rate-limit on bulk metadata fetch.
Resumes automatically — already-downloaded files are skipped.
"""
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from huggingface_hub import hf_hub_download, snapshot_download

SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "multilingual-e5-small"))
os.makedirs(SAVE_DIR, exist_ok=True)

REPO = "intfloat/multilingual-e5-small"

print(f"=== Downloading {REPO} ===")
print(f"Saving to: {SAVE_DIR}")
print("This model is small (~450MB) and very fast.")
print()

try:
    snapshot_download(
        repo_id=REPO,
        local_dir=SAVE_DIR,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "onnx/*"], # download only necessary formats
    )
    print(" === Embedding model downloaded successfully ===")
except Exception as e:
    print(f" FAIL: {e}")

print()
print("=== All downloads complete ===")
