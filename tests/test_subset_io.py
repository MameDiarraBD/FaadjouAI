"""Le test officiel n'est pas puisé dans les fichiers train."""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faadjou_ai.subset_io import sample_one_directory, write_audio_parquet


def _write(path: Path, texts: list[str], split_name: str) -> None:
    rows = [
        {"audio": {"bytes": f"{split_name}-{i}".encode(), "path": f"{i}.mp3"}, "transcription": text}
        for i, text in enumerate(texts)
    ]
    # Nommé train- ou test- pour le classeur de fichiers.
    write_audio_parquet(rows, path)


def test_sample_keeps_test_separate(tmp_path: Path) -> None:
    _write(tmp_path / "train-00000.parquet", ["aa", "bb", "cc", "dd"], "train")
    _write(tmp_path / "test-00000.parquet", ["ee", "ff"], "test")
    train, train_bytes = sample_one_directory(tmp_path, "train", "transcription", limit=3, seed=1)
    test, _ = sample_one_directory(tmp_path, "test", "transcription", limit=10, seed=1)
    assert len(train) == 3
    assert train_bytes > 0
    assert {row["transcription"] for row in test} == {"ee", "ff"}
    assert "ee" not in {row["transcription"] for row in train}


def test_language_filter(tmp_path: Path) -> None:
    table = pa.table(
        {
            "audio": [{"bytes": b"a", "path": "a.mp3"}, {"bytes": b"b", "path": "b.mp3"}],
            "transcription": ["wolof", "francais"],
            "language": ["wo", "fr"],
        }
    )
    pq.write_table(table, tmp_path / "train-00000.parquet")
    rows, _ = sample_one_directory(
        tmp_path,
        "train",
        "transcription",
        limit=10,
        seed=0,
        language_column="language",
        language_keep="wo",
    )
    assert [row["transcription"] for row in rows] == ["wolof"]
