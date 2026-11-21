"""Lecture bornée de parquet audio, sans charger les corpus entiers.

POURQUOI : les dossiers wolof font plusieurs gigaoctets. Un essai CPU ou un
dépôt Drive/Kaggle ne doit lire que les extraits nécessaires. Les fichiers
``test-*.parquet`` ne sont jamais mélangés aux fichiers ``train-*.parquet``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from faadjou_ai.audio import split_parquet_paths
from faadjou_ai.config_utils import resolve_path

logger = logging.getLogger(__name__)

AUDIO_SCHEMA = pa.schema(
    [
        pa.field("audio", pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])),
        pa.field("transcription", pa.string()),
    ]
)


def _audio_bytes(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, dict):
        raw = value.get("bytes") or b""
        return bytes(raw) if raw else b""
    return b""


def sample_one_directory(
    directory: str | Path,
    split: str,
    text_column: str,
    limit: int,
    seed: int,
    language_column: str | None = None,
    language_keep: str | None = None,
    max_clip_bytes: int | None = None,
    byte_budget: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Prend au plus ``limit`` extraits dans les fichiers train OU test.

    On s'arrête dès que le quota ou le budget d'octets est atteint : les
    fichiers suivants ne sont pas ouverts.
    """
    if limit <= 0:
        return [], 0
    root = resolve_path(directory)
    train_files, test_files = split_parquet_paths(root)
    files = test_files if split == "test" else train_files
    if not files:
        logger.warning("Aucun parquet %s dans %s", split, root)
        return [], 0

    order = np.random.default_rng(seed).permutation(len(files))
    rows: list[dict[str, Any]] = []
    used = 0
    columns = ["audio", text_column]
    # La colonne langue n'existe pas dans tous les corpus.
    for path in (files[int(i)] for i in order):
        if len(rows) >= limit:
            break
        schema_names = pq.ParquetFile(path).schema_arrow.names
        batch_cols = [c for c in columns if c in schema_names]
        if language_column and language_column in schema_names:
            batch_cols.append(language_column)
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=32, columns=batch_cols):
            for record in batch.to_pylist():
                if language_column and language_keep is not None:
                    if language_column not in record:
                        continue
                    if record.get(language_column) != language_keep:
                        continue
                raw = _audio_bytes(record.get("audio"))
                if not raw:
                    continue
                if max_clip_bytes is not None and len(raw) > max_clip_bytes:
                    continue
                if byte_budget is not None and used + len(raw) > byte_budget:
                    continue
                text = record.get(text_column)
                if text is None or not str(text).strip():
                    continue
                path_hint = None
                if isinstance(record.get("audio"), dict):
                    path_hint = record["audio"].get("path")
                rows.append(
                    {
                        "audio": {"bytes": raw, "path": path_hint or "clip.mp3"},
                        "transcription": str(text).strip(),
                    }
                )
                used += len(raw)
                if len(rows) >= limit:
                    break
            if len(rows) >= limit:
                break
    return rows, used


def sample_sources(
    sources: list[dict[str, Any]],
    split: str,
    limit: int,
    seed: int,
    max_clip_bytes: int | None = None,
    byte_budget: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Répartit le quota entre les sources, puis complète si une source est courte."""
    if limit <= 0 or not sources:
        return [], 0
    rows: list[dict[str, Any]] = []
    used = 0
    remaining = limit
    for index, src in enumerate(sources):
        share = remaining if index == len(sources) - 1 else max(1, limit // len(sources))
        share = min(share, remaining)
        budget = None if byte_budget is None else byte_budget - used
        part, part_bytes = sample_one_directory(
            src["path"],
            split=split,
            text_column=src.get("text_column", "transcription"),
            limit=share,
            seed=seed + index,
            language_column=src.get("language_column"),
            language_keep=src.get("language_keep"),
            max_clip_bytes=max_clip_bytes,
            byte_budget=budget,
        )
        rows.extend(part)
        used += part_bytes
        remaining = limit - len(rows)
        if remaining <= 0:
            return rows[:limit], used
    if remaining > 0 and sources:
        budget = None if byte_budget is None else byte_budget - used
        part, part_bytes = sample_one_directory(
            sources[0]["path"],
            split=split,
            text_column=sources[0].get("text_column", "transcription"),
            limit=remaining,
            seed=seed + 17,
            language_column=sources[0].get("language_column"),
            language_keep=sources[0].get("language_keep"),
            max_clip_bytes=max_clip_bytes,
            byte_budget=budget,
        )
        rows.extend(part)
        used += part_bytes
    return rows[:limit], used


def write_audio_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=AUDIO_SCHEMA)
    pq.write_table(table, path)


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
