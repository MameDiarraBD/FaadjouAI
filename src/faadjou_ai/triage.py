"""Triage : prompt JSON, validation jsonschema, retry, défaut prudent, règles.

POURQUOI un JSON strict + schéma :
- Interopérable avec un backend Spring Boot.
- Validable (jury + tests).
- En cas d'échec : une seule nouvelle tentative, puis orientation prudente
  (médecin généraliste, urgence moyenne) — jamais un silence, jamais un
  diagnostic inventé.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from jsonschema import ValidationError, validate

from faadjou_ai.llm import LLMBackend
from faadjou_ai.rules import apply_red_flag_rules, load_red_flags, load_specialites

logger = logging.getLogger(__name__)

URGENCES = ("faible", "moyenne", "elevee", "vitale")


def build_triage_schema(specialites: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["specialite", "urgence", "signaux_alarme", "justification"],
        "additionalProperties": True,
        "properties": {
            "specialite": {"type": "string", "enum": specialites},
            "urgence": {"type": "string", "enum": list(URGENCES)},
            "signaux_alarme": {
                "type": "array",
                "items": {"type": "string"},
            },
            "justification": {"type": "string", "minLength": 1},
        },
    }


def build_triage_prompt(french_text: str, specialites: list[str]) -> str:
    specs = ", ".join(specialites)
    return f"""Tu ORIENTES un patient vers une spécialité. Tu ne poses AUCUN diagnostic.

Texte du patient (français) :
\"\"\"{french_text}\"\"\"

Réponds UNIQUEMENT avec un JSON valide de la forme exacte :
{{
  "specialite": "<une valeur parmi : {specs}>",
  "urgence": "<faible|moyenne|elevee|vitale>",
  "signaux_alarme": ["..."],
  "justification": "<phrase courte d'ORIENTATION, sans diagnostic>"
}}

Rappels :
- specialite doit être EXACTEMENT l'une des valeurs listées.
- justification doit préciser qu'il s'agit d'une orientation, pas d'un diagnostic.
- Si doute, choisis medecin_generaliste et urgence moyenne.
- Pas de markdown, pas de texte hors JSON.
"""


def extract_json_object(text: str) -> dict[str, Any]:
    """Extrait le premier objet JSON d'une réponse modèle (parfois entourée de markdown)."""
    text = text.strip()
    # Bloc ```json ... ```
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    # Premier {...}
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise json.JSONDecodeError("Aucun objet JSON trouvé", text, 0)


def default_triage(config: dict[str, Any], reason: str) -> dict[str, Any]:
    """Orientation prudente par défaut (jamais un diagnostic)."""
    triage_cfg = config["triage"]
    return {
        "specialite": triage_cfg.get("default_specialite", "medecin_generaliste"),
        "urgence": triage_cfg.get("default_urgence", "moyenne"),
        "signaux_alarme": [],
        "justification": (
            f"Orientation par défaut prudente ({reason}). "
            "Ceci n'est pas un diagnostic. Consultez un professionnel de santé."
        ),
        "json_valide": False,
        "fallback": True,
        "message_orientation": (
            "Le système n'a pas pu produire une réponse structurée valide. "
            "Orientation prudente vers un médecin généraliste. "
            "Ceci n'est PAS un diagnostic."
        ),
    }


def validate_triage_json(
    data: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, Any]:
    validate(instance=data, schema=schema)
    return data


def run_triage(
    french_text: str,
    llm: LLMBackend,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Pipeline complet : prompt → LLM → JSON → retry → défaut → règles red flags."""
    paths = config["paths"]
    specialites = load_specialites(paths["specialites_file"])
    red_flags = load_red_flags(paths["red_flags_file"])
    schema = build_triage_schema(specialites)
    prompt = build_triage_prompt(french_text, specialites)
    max_retries = int(config["triage"].get("max_retries", 1))
    emergency_phone = str(config["triage"].get("emergency_phone", "À RENSEIGNER"))

    last_error = "inconnu"
    raw_outputs: list[str] = []

    for attempt in range(max_retries + 1):
        try:
            raw = llm.generate(prompt)
            raw_outputs.append(raw)
            parsed = extract_json_object(raw)
            validate_triage_json(parsed, schema)
            parsed["json_valide"] = True
            parsed["fallback"] = False
            parsed["message_orientation"] = (
                "Ceci est une ORIENTATION automatique vers un professionnel de santé. "
                "Ce n'est PAS un diagnostic médical et cela ne remplace pas une consultation."
            )
            parsed = apply_red_flag_rules(
                parsed, french_text, red_flags, emergency_phone=emergency_phone
            )
            parsed["raw_llm"] = raw
            return parsed
        except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as exc:
            last_error = str(exc)
            logger.warning("Tentative triage %d échouée : %s", attempt + 1, exc)
            # Relance avec rappel
            prompt = (
                build_triage_prompt(french_text, specialites)
                + "\nIMPORTANT : ta réponse précédente n'était pas un JSON valide "
                "conforme au schéma. Réponds UNIQUEMENT avec le JSON demandé."
            )

    result = default_triage(config, reason=last_error)
    result["raw_llm"] = raw_outputs[-1] if raw_outputs else ""
    result = apply_red_flag_rules(
        result, french_text, red_flags, emergency_phone=emergency_phone
    )
    return result
