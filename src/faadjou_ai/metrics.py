"""Métriques partagées — normalisation de texte, WER, CER.

POURQUOI une normalisation unique : sans elle, WER/CER varient selon la
casse et la ponctuation, ce qui fausse la comparaison entre modèles.
"""

from __future__ import annotations

import re
import string
import unicodedata

from jiwer import cer, wer


def normalize_text(text: str) -> str:
    """Normalisation unique partagée partout (ASR, triage, tests).

    Étapes :
    1. Unicode NFC
    2. Minuscules
    3. Suppression de la ponctuation
    4. Compression des espaces
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.lower()
    # Ponctuation ASCII + quelques signes fréquents
    punct = string.punctuation + "«»…–—‚„""''"
    table = str.maketrans({c: " " for c in punct})
    text = text.translate(table)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(references: list[str], hypotheses: list[str]) -> float:
    """WER (Word Error Rate) après normalisation commune."""
    refs = [normalize_text(r) for r in references]
    hyps = [normalize_text(h) for h in hypotheses]
    # jiwer échoue sur listes vides ou refs toutes vides
    if not refs or all(r == "" for r in refs):
        return 0.0 if all(h == "" for h in hyps) else 1.0
    return float(wer(refs, hyps))


def compute_cer(references: list[str], hypotheses: list[str]) -> float:
    """CER (Character Error Rate) après normalisation commune."""
    refs = [normalize_text(r) for r in references]
    hyps = [normalize_text(h) for h in hypotheses]
    if not refs or all(r == "" for r in refs):
        return 0.0 if all(h == "" for h in hyps) else 1.0
    return float(cer(refs, hyps))
