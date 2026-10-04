"""Append-only journal on disk: record log, activation log and a float32 vector matrix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from insight_graph_service.core.fileio import replace_file, retry_sharing


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """All records of a jsonl log ([] when absent)."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _file_size(path: Path) -> int:
    """Byte size of a file (0 when absent)."""
    return path.stat().st_size if path.exists() else 0


class ChunkJournal:
    def __init__(self, cache_dir: Path, *, records_name: str = "chunks.jsonl") -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_path = self.cache_dir / records_name
        self.activations_path = self.cache_dir / "activations.jsonl"
        self.embeddings_path = self.cache_dir / "embeddings.mmap"
        self.meta_path = self.cache_dir / "embeddings_meta.json"

    def append_batch(
        self,
        chunks: list[dict[str, Any]],
        vectors: np.ndarray,
        activations: list[dict[str, Any]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors row count mismatch")

        with self.chunks_path.open("a", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        if len(vectors):
            existing = self.load_embeddings()
            new = np.asarray(vectors, dtype=np.float32)
            self._write_embeddings(np.vstack([existing, new]) if existing.size else new)

        with self.activations_path.open("a", encoding="utf-8") as f:
            for edge in activations:
                f.write(json.dumps(edge) + "\n")

    def rewrite(self, chunks: list[dict[str, Any]], vectors: np.ndarray, activations: list[dict[str, Any]]) -> None:
        """Replace the whole journal (SIG dataset deletion); each file is written aside and swapped."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors row count mismatch")
        for path, rows in ((self.chunks_path, chunks), (self.activations_path, activations)):
            tmp = path.with_suffix(".jsonl.tmp")
            tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            replace_file(tmp, path)
        self._write_embeddings(np.asarray(vectors, dtype=np.float32))

    def _write_embeddings(self, matrix: np.ndarray) -> None:
        """Replace the vector matrix; written aside and swapped so readers never see a partial file."""
        if len(matrix) == 0:
            self.embeddings_path.unlink(missing_ok=True)
            self.meta_path.unlink(missing_ok=True)
            return
        tmp_path = self.embeddings_path.with_suffix(".mmap.tmp")
        mmap = np.memmap(tmp_path, dtype=np.float32, mode="w+", shape=matrix.shape)
        mmap[:] = matrix
        mmap.flush()
        del mmap
        meta_tmp = self.meta_path.with_suffix(".json.tmp")
        meta_tmp.write_text(json.dumps({"rows": int(matrix.shape[0]), "dim": int(matrix.shape[1])}), encoding="utf-8")
        replace_file(tmp_path, self.embeddings_path)  # waits out a reader in another process (Windows)
        replace_file(meta_tmp, self.meta_path)

    def load_chunks(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.chunks_path)

    def load_activations(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.activations_path)

    def load_embeddings(self) -> np.ndarray:
        if not (self.embeddings_path.exists() and self.meta_path.exists()):
            return np.empty((0, 0), dtype=np.float32)
        meta = json.loads(retry_sharing(lambda: self.meta_path.read_text(encoding="utf-8")))
        shape = (int(meta["rows"]), int(meta["dim"]))
        return retry_sharing(lambda: np.array(np.memmap(self.embeddings_path, dtype=np.float32, mode="r", shape=shape)))

    def row_count(self) -> int:
        if not self.chunks_path.exists():
            return 0
        with self.chunks_path.open(encoding="utf-8") as f:
            return sum(1 for _ in f)

    def mark(self) -> dict[str, int]:
        """Current journal extent, restorable with ``truncate``."""
        return {
            "records_bytes": _file_size(self.chunks_path),
            "activations_bytes": _file_size(self.activations_path),
            "rows": self.row_count(),
        }

    def truncate(self, mark: dict[str, int]) -> None:
        """Cut the journal back to a previous ``mark`` (rollback of a failed batch)."""
        for path, size in ((self.chunks_path, mark["records_bytes"]), (self.activations_path, mark["activations_bytes"])):
            if path.exists():
                with path.open("r+b") as f:
                    f.truncate(size)
        vectors = self.load_embeddings()
        if len(vectors) > mark["rows"]:
            self._write_embeddings(vectors[: mark["rows"]])
