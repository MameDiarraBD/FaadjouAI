"""Contrat HTTP sans télécharger les modèles."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from faadjou_ai.api import create_app
from faadjou_ai.config_utils import load_config


class _Translator:
    def translate(self, text: str) -> str:
        return "mal à la gorge depuis trois jours"


class _LLM:
    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        return (
            '{"specialite": "orl", "urgence": "faible", '
            '"signaux_alarme": [], "justification": "Orientation ORL, pas un diagnostic."}'
        )


def test_health_and_triage_text() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "config.yaml")
    app = create_app(
        config,
        components={"llm": _LLM(), "translator": _Translator(), "sample_rate": 16000},
    )
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        response = client.post("/triage", data={"texte": "sama bakkan dafa metti"})
        assert response.status_code == 200
        body = response.json()
        assert body["transcription"] == "sama bakkan dafa metti"
        assert body["triage"]["specialite"] == "orl"
        assert "disclaimer" in body
