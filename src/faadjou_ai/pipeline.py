"""Chaîne Wax : audio ou texte → transcription → français → JSON → règles.

Le modèle propose une orientation. Les signaux d'alarme (rules.py) sont
appliqués APRÈS, dans run_triage. Le système n'émet pas de diagnostic.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from faadjou_ai.triage import run_triage


def orient(
    config: dict[str, Any],
    llm: Any,
    translator: Any | None = None,
    asr: tuple[Any, Any, dict[str, Any]] | None = None,
    french_text: str | None = None,
    wolof_text: str | None = None,
    audio: np.ndarray | None = None,
    sample_rate: int = 16000,
) -> dict[str, Any]:
    """Un seul point d'entrée pour l'API et l'évaluation.

    Priorité : audio wolof, sinon texte wolof à traduire, sinon texte français.
    """
    transcription = None
    translation = None

    if audio is not None:
        if asr is None:
            raise ValueError("Un audio est fourni mais aucun modèle ASR n'est chargé.")
        from faadjou_ai.asr import transcribe_array

        model, processor, gen_kwargs = asr
        transcription = transcribe_array(
            model,
            processor,
            audio,
            sample_rate=sample_rate,
            gen_kwargs=gen_kwargs,
        )
        if translator is None:
            raise ValueError("Un audio est fourni mais le traducteur n'est pas chargé.")
        translation = translator.translate(transcription)
        french = translation
    elif wolof_text and str(wolof_text).strip():
        transcription = str(wolof_text).strip()
        if translator is None:
            raise ValueError("Un texte wolof est fourni mais le traducteur n'est pas chargé.")
        translation = translator.translate(transcription)
        french = translation
    elif french_text and str(french_text).strip():
        french = str(french_text).strip()
        translation = french
    else:
        raise ValueError("Fournir un audio, un texte wolof ou un texte français.")

    triage = run_triage(french, llm, config)
    return {
        "transcription": transcription,
        "traduction": translation,
        "triage": triage,
    }
