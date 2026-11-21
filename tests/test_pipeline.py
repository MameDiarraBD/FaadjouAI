"""Chaîne Wax sur un cas minuscule : traduction fictive puis règle d'alarme."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faadjou_ai.config_utils import load_config
from faadjou_ai.pipeline import orient


class _Translator:
    def translate(self, text: str) -> str:
        return "douleur thoracique soudaine"


class _LLM:
    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        return (
            '{"specialite": "medecin_generaliste", "urgence": "faible", '
            '"signaux_alarme": [], "justification": "Orientation generale, pas un diagnostic."}'
        )


def test_wolof_text_then_red_flag_forces_vitale() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "config.yaml")
    out = orient(
        config,
        llm=_LLM(),
        translator=_Translator(),
        wolof_text="sama bopp dafa metti",
    )
    assert out["transcription"] == "sama bopp dafa metti"
    assert "thoracique" in out["traduction"]
    assert out["triage"]["urgence"] == "vitale"
    assert out["triage"]["specialite"] == "urgences"
    assert out["triage"]["regles_appliquees"] is True
