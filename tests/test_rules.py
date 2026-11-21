"""Tests — règles de signaux d'alarme (cas grave → vitale)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faadjou_ai.rules import apply_red_flag_rules, load_red_flags


def test_red_flag_forces_vitale() -> None:
    root = Path(__file__).resolve().parents[1]
    data = load_red_flags(root / "rules" / "red_flags.json")
    triage = {
        "specialite": "medecin_generaliste",
        "urgence": "faible",
        "signaux_alarme": [],
        "justification": "Orientation simple.",
    }
    out = apply_red_flag_rules(
        triage,
        french_text="Le patient a une douleur thoracique soudaine.",
        red_flags_data=data,
        emergency_phone="TEST",
    )
    assert out["urgence"] == "vitale"
    assert out["specialite"] == "urgences"
    assert out["regles_appliquees"] is True
    assert len(out["red_flags_detectes"]) >= 1


def test_no_red_flag_keeps_urgency() -> None:
    root = Path(__file__).resolve().parents[1]
    data = load_red_flags(root / "rules" / "red_flags.json")
    triage = {
        "specialite": "dermatologie",
        "urgence": "faible",
        "signaux_alarme": [],
        "justification": "Éruption cutanée légère.",
    }
    out = apply_red_flag_rules(
        triage,
        french_text="Petite rougeur sur le bras qui démange.",
        red_flags_data=data,
    )
    assert out["urgence"] == "faible"
    assert out["regles_appliquees"] is False
