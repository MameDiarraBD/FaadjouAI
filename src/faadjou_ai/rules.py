"""Couche de règles explicites (signaux d'alarme) — APRÈS le modèle.

POURQUOI des règles après le LLM :
Le modèle ne décide jamais seul d'un cas grave. Si un motif de red_flags.json
apparaît dans le texte français OU dans la sortie du modèle, on force
urgence='vitale' et spécialité='urgences'. C'est une garde-fou défendable
devant un jury clinique.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from faadjou_ai.config_utils import resolve_path
from faadjou_ai.metrics import normalize_text

logger = logging.getLogger(__name__)


def load_specialites(path: str | Path) -> list[str]:
    p = resolve_path(path)
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    return list(data["specialites"])


def load_red_flags(path: str | Path) -> dict[str, Any]:
    p = resolve_path(path)
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def detect_red_flags(
    texts: list[str],
    red_flags_data: dict[str, Any],
) -> list[dict[str, str]]:
    """Retourne la liste des red flags dont un motif apparaît dans un des textes."""
    joined = " ".join(normalize_text(t) for t in texts if t)
    hits: list[dict[str, str]] = []
    for flag in red_flags_data.get("red_flags", []):
        for motif in flag.get("motifs", []):
            if normalize_text(motif) in joined:
                hits.append(
                    {
                        "id": flag.get("id", ""),
                        "motif": motif,
                        "raison": flag.get("raison", ""),
                    }
                )
                break
    return hits


def apply_red_flag_rules(
    triage_result: dict[str, Any],
    french_text: str,
    red_flags_data: dict[str, Any],
    emergency_phone: str = "À RENSEIGNER",
) -> dict[str, Any]:
    """Force urgence vitale + urgences si signal d'alarme détecté.

    Le message rappelle que le système ORIENTE et ne diagnostique pas,
    et pointe vers le numéro d'urgence de la config.
    """
    model_signals = triage_result.get("signaux_alarme") or []
    texts = [french_text, " ".join(str(s) for s in model_signals)]
    # Inclure aussi justification
    texts.append(str(triage_result.get("justification", "")))

    hits = detect_red_flags(texts, red_flags_data)
    result = dict(triage_result)
    result["regles_appliquees"] = False
    result["red_flags_detectes"] = []

    if not hits:
        return result

    result["regles_appliquees"] = True
    result["urgence"] = "vitale"
    result["specialite"] = "urgences"
    result["red_flags_detectes"] = hits
    # Enrichir signaux_alarme sans dupliquer
    existing = [normalize_text(s) for s in (result.get("signaux_alarme") or [])]
    for h in hits:
        if normalize_text(h["motif"]) not in existing:
            result.setdefault("signaux_alarme", []).append(h["motif"])
            existing.append(normalize_text(h["motif"]))

    result["justification"] = (
        f"[RÈGLE EXPLICITE] Signal(aux) d'alarme détecté(s). "
        f"Orientation vers les urgences (pas un diagnostic). "
        f"Composer le numéro d'urgence : {emergency_phone}. "
        f"Détail : {'; '.join(h['raison'] for h in hits)}. "
        f"Sortie modèle initiale conservée en contexte : "
        f"{triage_result.get('justification', '')}"
    )
    result["message_orientation"] = (
        "Ceci est une ORIENTATION automatique, pas un diagnostic médical. "
        f"En présence de signes d'alarme, contactez immédiatement les urgences "
        f"({emergency_phone}) ou rendez-vous au service d'urgence le plus proche."
    )
    logger.warning("Red flags déclenchés : %s", [h["id"] for h in hits])
    return result
