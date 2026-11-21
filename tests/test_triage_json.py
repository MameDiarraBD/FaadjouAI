"""Tests — validation JSON de triage."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from jsonschema import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faadjou_ai.rules import load_specialites
from faadjou_ai.triage import (
    build_triage_schema,
    default_triage,
    extract_json_object,
    validate_triage_json,
)


def test_extract_json_object() -> None:
    raw = 'Voici le résultat:\n```json\n{"specialite": "orl", "urgence": "faible", "signaux_alarme": [], "justification": "Orientation ORL"}\n```'
    data = extract_json_object(raw)
    assert data["specialite"] == "orl"


def test_validate_ok() -> None:
    root = Path(__file__).resolve().parents[1]
    specs = load_specialites(root / "rules" / "specialites.json")
    schema = build_triage_schema(specs)
    data = {
        "specialite": "medecin_generaliste",
        "urgence": "moyenne",
        "signaux_alarme": [],
        "justification": "Orientation prudente, pas un diagnostic.",
    }
    validate_triage_json(data, schema)


def test_validate_bad_urgence() -> None:
    root = Path(__file__).resolve().parents[1]
    specs = load_specialites(root / "rules" / "specialites.json")
    schema = build_triage_schema(specs)
    data = {
        "specialite": "medecin_generaliste",
        "urgence": "critique",  # invalide
        "signaux_alarme": [],
        "justification": "x",
    }
    with pytest.raises(ValidationError):
        validate_triage_json(data, schema)


def test_default_triage_prudent() -> None:
    config = {
        "triage": {
            "default_specialite": "medecin_generaliste",
            "default_urgence": "moyenne",
        }
    }
    out = default_triage(config, reason="test")
    assert out["specialite"] == "medecin_generaliste"
    assert out["urgence"] == "moyenne"
    assert out["fallback"] is True
