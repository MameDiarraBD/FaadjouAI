"""Traduction wolof → français via NLLB.

POURQUOI NLLB-200-distilled-600M :
- Couvre wol_Latn et fra_Latn officiellement.
- Taille raisonnable pour Colab T4.
- Distillé = plus rapide que le 1.3B / 3.3B pour une démo.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


class WolofToFrenchTranslator:
    """Traducteur wolof (latin) → français (latin)."""

    def __init__(
        self,
        model_id: str = "facebook/nllb-200-distilled-600M",
        src_lang: str = "wol_Latn",
        tgt_lang: str = "fra_Latn",
        max_length: int = 256,
    ) -> None:
        self.model_id = model_id
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Chargement NLLB %s sur %s", model_id, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        # NLLB exige de fixer src_lang sur le tokenizer
        if hasattr(self.tokenizer, "src_lang"):
            self.tokenizer.src_lang = src_lang

    def translate(self, text: str) -> str:
        if not text or not str(text).strip():
            return ""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # id du token de langue cible
        forced_bos = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_length=self.max_length,
            )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "WolofToFrenchTranslator":
        t = config["translate"]
        return cls(
            model_id=t["model_id"],
            src_lang=t["src_lang"],
            tgt_lang=t["tgt_lang"],
            max_length=int(t.get("max_length", 256)),
        )
