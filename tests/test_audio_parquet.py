"""Les fichiers test-*.parquet ne doivent pas être classés avec le train."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faadjou_ai.audio import split_parquet_paths


def test_split_parquet_paths_keeps_official_test(tmp_path: Path) -> None:
    (tmp_path / "train-00000-of-00002.parquet").write_bytes(b"x")
    (tmp_path / "train-00001-of-00002 (1).parquet").write_bytes(b"x")
    (tmp_path / "test-00000-of-00001.parquet").write_bytes(b"x")

    train, test = split_parquet_paths(tmp_path)

    assert [p.name for p in train] == ["train-00000-of-00002.parquet"]
    assert [p.name for p in test] == ["test-00000-of-00001.parquet"]
