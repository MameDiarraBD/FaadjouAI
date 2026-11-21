"""Interface LLM abstraite + backends local (4-bit) et API distante.

POURQUOI une abstraction :
- Sur Colab T4 on charge un petit modèle instruct en 4-bit.
- Si la VRAM manque, on bascule vers une API (clés en variables d'environnement).
Le triage reste identique quel que soit le backend.
"""

from __future__ import annotations

import abc
import logging
from typing import Any

import requests

from faadjou_ai.config_utils import get_env

logger = logging.getLogger(__name__)


class LLMBackend(abc.ABC):
    """Interface commune : une seule méthode generate(prompt) -> str."""

    @abc.abstractmethod
    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        raise NotImplementedError


class LocalTransformersLLM(LLMBackend):
    """Modèle ouvert local via transformers, optionnellement en 4-bit.

    Défaut recommandé : Qwen/Qwen2.5-1.5B-Instruct (licence Apache-2.0).
    Vérifiez toujours la licence du modèle retenu avant soumission concours.
    """

    def __init__(
        self,
        model_id: str,
        load_in_4bit: bool = True,
        max_new_tokens: int = 256,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "Chargement LLM local %s (4bit=%s) sur %s",
            model_id,
            load_in_4bit,
            self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        quant_config = None
        if load_in_4bit and self.device == "cuda":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        if self.device == "cpu":
            self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        import torch

        n_tokens = max_new_tokens or self.max_new_tokens
        # Format chat si le tokenizer le permet (Qwen, etc.)
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Tu es un assistant d'ORIENTATION médicale. "
                        "Tu ne poses JAMAIS de diagnostic. "
                        "Tu réponds UNIQUEMENT en JSON valide."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt

        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=n_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # Ne décoder que la partie générée
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


class RemoteAPILLM(LLMBackend):
    """Appel HTTP générique compatible style OpenAI chat completions.

    Variables d'environnement (jamais dans le code) :
    - FAADJOU_LLM_API_URL
    - FAADJOU_LLM_API_KEY
    """

    def __init__(
        self,
        url: str,
        api_key: str | None,
        model: str | None = None,
        max_new_tokens: int = 256,
    ) -> None:
        if not url:
            raise ValueError(
                "URL API LLM manquante. Définissez FAADJOU_LLM_API_URL."
            )
        self.url = url
        self.api_key = api_key
        self.model = model
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict[str, Any] = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Tu es un assistant d'ORIENTATION médicale. "
                        "Tu ne poses JAMAIS de diagnostic. "
                        "Tu réponds UNIQUEMENT en JSON valide."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_new_tokens or self.max_new_tokens,
            "temperature": 0,
        }
        if self.model:
            payload["model"] = self.model
        resp = requests.post(self.url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # Formats courants
        if "choices" in data:
            return data["choices"][0]["message"]["content"].strip()
        if "content" in data:
            return str(data["content"]).strip()
        if "text" in data:
            return str(data["text"]).strip()
        raise ValueError(f"Réponse API LLM non reconnue : clés={list(data.keys())}")


def build_llm(config: dict[str, Any]) -> LLMBackend:
    """Construit le backend selon config['llm']['backend']."""
    llm_cfg = config["llm"]
    backend = llm_cfg.get("backend", "local")
    if backend == "local":
        local = llm_cfg["local"]
        return LocalTransformersLLM(
            model_id=local["model_id"],
            load_in_4bit=bool(local.get("load_in_4bit", True)),
            max_new_tokens=int(local.get("max_new_tokens", 256)),
        )
    if backend == "api":
        api = llm_cfg["api"]
        url = get_env(api.get("url_env", "FAADJOU_LLM_API_URL"))
        key = get_env(api.get("key_env", "FAADJOU_LLM_API_KEY"))
        return RemoteAPILLM(
            url=url or "",
            api_key=key,
            model=api.get("model"),
            max_new_tokens=int(llm_cfg.get("local", {}).get("max_new_tokens", 256)),
        )
    raise ValueError(f"llm.backend inconnu : {backend}")
