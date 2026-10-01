"""Download an embedding model into ``sig/models/`` at a pinned revision (docs/04_representation.md §4.4).

    python scripts/download_model.py                       # Qwen/Qwen3-Embedding-0.6B (default)
    python scripts/download_model.py --model <repo> --revision <sha>

The folder name is the last path segment of the repo id (``models/Qwen3-Embedding-0.6B``),
which is where ``ltir.encoder.model_folder`` loads it from, offline, afterwards.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PINNED = {"Qwen/Qwen3-Embedding-0.6B": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"}
IGNORE = ["*.onnx", "onnx/*", "openvino/*", "*.h5", "*.msgpack", "*.gguf"]
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--revision", default=None, help="commit sha (default: the pinned one)")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    revision = args.revision or PINNED.get(args.model)
    if not revision:
        print(f"no pinned revision for {args.model}; pass --revision <sha>", file=sys.stderr)
        return 2
    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import snapshot_download

    target = args.model_dir / args.model.split("/")[-1]
    snapshot_download(repo_id=args.model, revision=revision, local_dir=str(target), ignore_patterns=IGNORE)
    if not (target / "modules.json").is_file():
        print(f"{target} is not a sentence-transformers folder (no modules.json)", file=sys.stderr)
        return 1
    (target / "REVISION").write_text(f"{args.model}@{revision}\n", encoding="utf-8")
    size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
    print(f"{args.model}@{revision[:12]} -> {target} ({size / 2**30:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
