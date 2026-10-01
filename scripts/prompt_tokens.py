"""Token cost of the LLM-facing text (docs/07_question_answering.md §7.7): evidence prompt and canonical documents.

    python scripts/prompt_tokens.py --label baseline

Tokenizers (offline): the bundled MiniLM tokenizer (XLM-R SentencePiece, 250k vocabulary —
a proxy for Gemma's 256k SentencePiece; the Gemma tokenizer is gated) and the Qwen3
byte-level BPE tokenizer when ``models/Qwen3-Embedding-0.6B`` is present.
"""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

from scratch import DEMO_QUESTION, demo_engine, save_result

from ltir.config import load_config

TOKENIZER_FOLDERS = {"xlmr_sentencepiece": "paraphrase-multilingual-MiniLM-L12-v2", "qwen3_bpe": "Qwen3-Embedding-0.6B"}


def load_tokenizers() -> dict:
    """Available offline tokenizers by name."""
    from transformers import AutoTokenizer

    model_dir = Path(load_config().model_dir)
    found = {}
    for name, folder in TOKENIZER_FOLDERS.items():
        if (model_dir / folder / "tokenizer_config.json").is_file():
            with contextlib.redirect_stderr(io.StringIO()):
                found[name] = AutoTokenizer.from_pretrained(str(model_dir / folder))
    return found


def measure(text: str, tokenizers: dict) -> dict:
    """Characters, non-ASCII characters and token counts of one text."""
    row = {"chars": len(text), "non_ascii": sum(1 for ch in text if ord(ch) > 127)}
    for name, tok in tokenizers.items():
        row[name] = len(tok(text, add_special_tokens=False, verbose=False)["input_ids"])
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="current", help="name of this measurement (e.g. baseline, canon3)")
    parser.add_argument(
        "--backend",
        default="hashing",
        help="embedding backend of the scratch run: the documents do not depend on it, the evidence prompt does (retrieval)",
    )
    args = parser.parse_args()
    engine = demo_engine("prompt_tokens", embedding_backend=args.backend)
    graph = engine.graph()
    qa = engine.ask(DEMO_QUESTION, use_llm=False)
    tokenizers = load_tokenizers()
    docs = [graph.canonical_document(n["id"]) for n in graph.of_kind("Pattern")]
    surfaces = {
        "evidence_prompt": qa.evidence["prompt"],
        "evidence_only_answer": qa.answer,
        "canonical_documents_total": "\n".join(docs),
    }
    result = {name: measure(text, tokenizers) for name, text in surfaces.items()}
    result["documents"] = len(docs)
    result["evidence_items"] = len(qa.evidence["items"])
    cols = ["chars", "non_ascii", *tokenizers]
    print(f"{'surface':<28}" + "".join(f"{c:>20}" for c in cols))
    for name in surfaces:
        print(f"{name:<28}" + "".join(f"{result[name][c]:>20}" for c in cols))
    print(f"({result['evidence_items']} evidence items, {result['documents']} documents)  saved: {save_result('prompt_tokens', args.label, result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
