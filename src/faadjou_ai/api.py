"""API FastAPI. Les modèles sont injectés ou chargés une seule fois au démarrage.

Contrat pour le backend Spring Boot :
  GET  /health
  POST /triage   (multipart : audio et/ou texte)
"""

from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from faadjou_ai.config_utils import set_seed
from faadjou_ai.pipeline import orient

STATE: dict[str, Any] = {}


def _load_models(config: dict[str, Any]) -> None:
    from faadjou_ai.asr import load_asr_model
    from faadjou_ai.llm import build_llm
    from faadjou_ai.translate import WolofToFrenchTranslator

    set_seed(int(config["project"]["seed"]))
    STATE["config"] = config
    STATE["llm"] = build_llm(config)
    STATE["translator"] = WolofToFrenchTranslator.from_config(config)
    model, processor, gen_kwargs = load_asr_model(
        config["api"]["asr_model"],
        use_fp16=bool(config["asr"].get("use_fp16", True)),
    )
    STATE["asr"] = (model, processor, gen_kwargs)
    STATE["sample_rate"] = int(config["audio"]["sample_rate"])


def create_app(config: dict[str, Any], components: dict[str, Any] | None = None) -> FastAPI:
    """components permet de tester l'API sans télécharger les modèles."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if components is None:
            _load_models(config)
        else:
            STATE.clear()
            STATE.update(components)
            STATE["config"] = config
        yield
        STATE.clear()

    app = FastAPI(
        title="Faadjou AI",
        description="Orientation uniquement. Ne remplace pas un médecin.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "models_loaded": "llm" in STATE,
            "disclaimer": (
                "Le système oriente vers un professionnel ; "
                "il ne pose jamais de diagnostic et ne remplace pas un médecin."
            ),
        }

    @app.post("/triage")
    async def triage(
        audio: UploadFile | None = File(default=None),
        texte: str | None = Form(default=None),
        texte_deja_francais: bool = Form(default=False),
    ) -> JSONResponse:
        if audio is None and not (texte and texte.strip()):
            raise HTTPException(status_code=400, detail="Fournir un audio wolof et/ou un texte.")

        wave = None
        french = None
        wolof = None
        if audio is not None:
            from faadjou_ai.audio import load_waveform

            suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(await audio.read())
                tmp_path = tmp.name
            try:
                wave = load_waveform(tmp_path, sample_rate=int(STATE["sample_rate"]))
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=400, detail=f"Audio illisible : {exc}") from exc
        elif texte_deja_francais:
            french = texte.strip()
        else:
            wolof = texte.strip()

        try:
            result = orient(
                config=STATE["config"],
                llm=STATE["llm"],
                translator=STATE.get("translator"),
                asr=STATE.get("asr"),
                french_text=french,
                wolof_text=wolof,
                audio=wave,
                sample_rate=int(STATE.get("sample_rate", 16000)),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        phone = STATE["config"]["triage"].get("emergency_phone", "À RENSEIGNER")
        result["disclaimer"] = (
            "ORIENTATION uniquement — ne remplace pas un médecin. "
            f"En cas de signes graves, contactez les urgences ({phone})."
        )
        return JSONResponse(result)

    return app
