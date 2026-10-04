"""The embedding model on disk: the pinned Qwen3 checkpoint, verified file by file (docs/09_operations.md).

    python -m insight_graph_service.core.model_store [MODEL_DIR] [--seed DIR]

Run once before the service starts (the compose ``model-init`` container; on a host, once after cloning). A copy
that verifies is kept; otherwise a seed that verifies (the host's ``models/``) is copied; otherwise the pinned
revision is downloaded. It exits non-zero unless the result verifies, so a corrupt or partial copy never reaches the
service, which checks the same manifest (presence and sizes) before it loads the model.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

from attractor_topology.encoder import QWEN_MODEL, QWEN_REVISION, model_folder

# file -> (size, sha256) at QWEN_REVISION; sizes and the LFS hashes (model.safetensors, tokenizer.json) checked against the Hub
MANIFEST: dict[str, tuple[int, str]] = {
    "1_Pooling/config.json": (313, "37bf193fa101f19101bfad9c31d3eb0f786e247b7b1e5cb7f007d730eed1ddbd"),
    "config.json": (727, "b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9"),
    "config_sentence_transformers.json": (215, "10667c72ddb772627bf1780cb7f86af8e2ae0032b8c243c731172064105c6961"),
    "generation_config.json": (117, "28396d421a2108acce96383f6a7de78008f7f1b17f807958f3c14c51dbfb65fb"),
    "merges.txt": (1671853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
    "model.safetensors": (1191586416, "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"),
    "modules.json": (349, "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf"),
    "tokenizer.json": (11423705, "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a"),
    "tokenizer_config.json": (9706, "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0"),
    "vocab.json": (2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
}


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def problems(folder: Path, *, hashes: bool = True) -> list[str]:
    """What is wrong with a model folder ([] = complete and intact); ``hashes=False`` checks presence and size only."""
    found = []
    for name, (size, sha) in MANIFEST.items():
        path = folder / name
        if not path.is_file():
            found.append(f"{name}: missing")
        elif path.stat().st_size != size:
            found.append(f"{name}: {path.stat().st_size} bytes, expected {size}")
        elif hashes and _sha256(path) != sha:
            found.append(f"{name}: sha256 mismatch")
    return found


def provision(model_dir: Path, seed_dir: Path | None = None) -> str:
    """Make ``model_dir`` hold the verified checkpoint; returns what was done."""
    target = model_folder(model_dir)
    if not problems(target):
        return "present"
    seed = model_folder(seed_dir) if seed_dir else None
    if seed is not None and seed.resolve() != target.resolve() and not problems(seed):
        for name in MANIFEST:
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(seed / name, target / name)
        done = f"copied from {seed}"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=QWEN_MODEL, revision=QWEN_REVISION, local_dir=str(target), allow_patterns=list(MANIFEST))
        done = f"downloaded {QWEN_MODEL}@{QWEN_REVISION[:12]}"
    bad = problems(target)
    if bad:
        raise RuntimeError(f"{target} does not verify after provisioning: {bad}")
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", nargs="?", type=Path, help="default: the service's MODEL_DIR (environment, then .env)")
    parser.add_argument("--seed", type=Path, default=None, help="a folder holding a copy to reuse when it verifies")
    args = parser.parse_args(argv)
    if args.model_dir is None:  # the folder the service will read
        from insight_graph_service.core.settings import load_env_file, load_settings

        load_env_file()
        args.model_dir = load_settings().topology.model_dir
    try:
        print(f"{model_folder(args.model_dir)}: {provision(args.model_dir, args.seed)}")
    except Exception as exc:  # report and fail: the service must not start on a bad copy
        print(f"model provisioning failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
