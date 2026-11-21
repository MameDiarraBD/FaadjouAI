"""Tests — normalisation de texte et WER sur un cas connu."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faadjou_ai.metrics import compute_wer, normalize_text


def test_normalize_text_lowercase_and_punct() -> None:
    assert normalize_text("  Bonjour,  WOLOF!!! ") == "bonjour wolof"


def test_normalize_text_spaces() -> None:
    assert normalize_text("a\t\nb   c") == "a b c"


def test_wer_identical() -> None:
    assert compute_wer(["salut monde"], ["Salut, monde!"]) == 0.0


def test_wer_known_case() -> None:
    # 1 substitution sur 2 mots → WER = 0.5
    refs = ["bonjour monde"]
    hyps = ["bonjour terre"]
    assert abs(compute_wer(refs, hyps) - 0.5) < 1e-6
