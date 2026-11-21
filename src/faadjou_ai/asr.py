"""Module 2 — reconnaissance vocale (ASR) : chargement, transcription, évaluation.

POURQUOI lire la config du modèle (tokenizer / generation_config) plutôt que
d'imposer un code de langue : chaque checkpoint wolof peut déclarer sa langue
différemment ; une hypothèse erronée dégrade le WER.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from faadjou_ai.metrics import compute_cer, compute_wer

logger = logging.getLogger(__name__)


@dataclass
class AsrEvalResult:
    model_id: str
    split_name: str
    wer: float
    cer: float
    n_examples: int
    mean_inference_sec: float


def _pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _generation_language_kwargs(model: Any, processor: Any) -> dict[str, Any]:
    """Réutilise les réglages langue du modèle ; ne suppose jamais un code.

    Ordre de priorité :
    1. generation_config.lang_to_id / language déjà définis
    2. config.language / config.forced_decoder_ids
    3. rien (le modèle décide)
    """
    kwargs: dict[str, Any] = {}
    gen = getattr(model, "generation_config", None)
    cfg = getattr(model, "config", None)

    language = None
    if gen is not None:
        language = getattr(gen, "language", None)
    if language is None and cfg is not None:
        language = getattr(cfg, "language", None)

    task = None
    if gen is not None:
        task = getattr(gen, "task", None)
    if task is None:
        task = "transcribe"

    # WhisperProcessor.get_decoder_prompt_ids si langue connue du tokenizer
    if language and hasattr(processor, "get_decoder_prompt_ids"):
        try:
            forced = processor.get_decoder_prompt_ids(language=language, task=task)
            kwargs["forced_decoder_ids"] = forced
            logger.info("Langue reprise du modèle : %s (task=%s)", language, task)
            return kwargs
        except Exception as exc:  # noqa: BLE001 — fallback volontaire
            logger.warning(
                "Impossible d'appliquer language=%s via processor (%s). "
                "On laisse le modèle sans contrainte de langue.",
                language,
                exc,
            )

    # forced_decoder_ids déjà dans generation_config
    if gen is not None:
        fdi = getattr(gen, "forced_decoder_ids", None)
        if fdi is not None:
            kwargs["forced_decoder_ids"] = fdi
            logger.info("forced_decoder_ids repris depuis generation_config.")
            return kwargs

    logger.info(
        "Aucune langue explicite dans la config du modèle — génération libre."
    )
    return kwargs


def load_asr_model(
    model_id: str,
    use_fp16: bool = True,
) -> tuple[Any, Any, dict[str, Any]]:
    """Charge modèle + processor ; retourne aussi les kwargs de génération."""
    device = _pick_device()
    dtype = torch.float16 if (use_fp16 and device == "cuda") else torch.float32
    logger.info("Chargement ASR %s sur %s (%s)", model_id, device, dtype)

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    gen_kwargs = _generation_language_kwargs(model, processor)
    return model, processor, gen_kwargs


def release_asr_model(model: Any) -> None:
    """Libère la VRAM entre deux évaluations (sessions Colab limitées)."""
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def transcribe_array(
    model: Any,
    processor: Any,
    audio: np.ndarray | dict[str, Any],
    sample_rate: int = 16000,
    gen_kwargs: dict[str, Any] | None = None,
) -> str:
    """Transcrit un seul extrait (array ou dict datasets Audio)."""
    if isinstance(audio, dict):
        array = np.asarray(audio["array"], dtype=np.float32)
        sample_rate = int(audio.get("sampling_rate", sample_rate))
    else:
        array = np.asarray(audio, dtype=np.float32)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inputs = processor(
        array,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device=device, dtype=dtype)
    generate_kw = dict(gen_kwargs or {})
    with torch.no_grad():
        predicted_ids = model.generate(input_features, **generate_kw)
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return text.strip()


def transcribe_batch(
    model: Any,
    processor: Any,
    audios: list[np.ndarray | dict[str, Any]],
    sample_rate: int = 16000,
    gen_kwargs: dict[str, Any] | None = None,
) -> list[str]:
    """Transcription par lots pour limiter le surcoût GPU."""
    arrays: list[np.ndarray] = []
    for audio in audios:
        if isinstance(audio, dict):
            arrays.append(np.asarray(audio["array"], dtype=np.float32))
            sample_rate = int(audio.get("sampling_rate", sample_rate))
        else:
            arrays.append(np.asarray(audio, dtype=np.float32))

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inputs = processor(
        arrays,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    input_features = inputs.input_features.to(device=device, dtype=dtype)
    generate_kw = dict(gen_kwargs or {})
    with torch.no_grad():
        predicted_ids = model.generate(input_features, **generate_kw)
    texts = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return [t.strip() for t in texts]


def build_asr_pipeline(model_id: str, use_fp16: bool = True) -> Any:
    """Pipeline HF pratique pour l'API / démo (charge une seule fois)."""
    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if (use_fp16 and device == 0) else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
    )


def evaluate_asr_on_dataset(
    model_id: str,
    dataset: Dataset,
    split_name: str,
    batch_size: int = 4,
    use_fp16: bool = True,
    sample_rate: int = 16000,
) -> AsrEvalResult:
    """Évalue un modèle sur un Dataset (colonnes audio, transcription)."""
    model, processor, gen_kwargs = load_asr_model(model_id, use_fp16=use_fp16)
    refs: list[str] = []
    hyps: list[str] = []
    durations: list[float] = []

    n = len(dataset)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = dataset.select(range(start, end))
        audios = [ex["audio"] for ex in batch]
        refs.extend([ex["transcription"] for ex in batch])
        t0 = time.perf_counter()
        try:
            preds = transcribe_batch(
                model, processor, audios, sample_rate=sample_rate, gen_kwargs=gen_kwargs
            )
        except Exception as exc:  # noqa: BLE001 — fallback un par un
            logger.warning("Batch échoué (%s), passage un par un.", exc)
            preds = [
                transcribe_array(
                    model, processor, a, sample_rate=sample_rate, gen_kwargs=gen_kwargs
                )
                for a in audios
            ]
        elapsed = time.perf_counter() - t0
        durations.append(elapsed / max(len(audios), 1))
        hyps.extend(preds)

    result = AsrEvalResult(
        model_id=model_id,
        split_name=split_name,
        wer=compute_wer(refs, hyps),
        cer=compute_cer(refs, hyps),
        n_examples=n,
        mean_inference_sec=float(np.mean(durations)) if durations else 0.0,
    )
    release_asr_model(model)
    return result
