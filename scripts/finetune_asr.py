#!/usr/bin/env python3
"""Fine-tuning Whisper small (wolof) avec Seq2SeqTrainer + reprise checkpoint.

POURQUOI ces choix (défense jury) :
- fp16 + gradient checkpointing + petit batch + accumulation : tenir sur T4 16 Go.
- LR faible : ne pas écraser les poids déjà adaptés au wolof.
- save_steps / save_total_limit : sessions Colab qui se coupent.
- Reprise auto depuis le dernier checkpoint.
- On ne conserve le modèle ajusté que s'il bat le baseline sur les tests.

Usage :
  python scripts/finetune_asr.py --config configs/config.yaml --max_steps 50
  python scripts/finetune_asr.py --push_to_hub   # nécessite HF_TOKEN
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.trainer_utils import get_last_checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faadjou_ai.asr import evaluate_asr_on_dataset, release_asr_model  # noqa: E402
from faadjou_ai.audio import prepare_asr_datasets  # noqa: E402
from faadjou_ai.config_utils import (  # noqa: E402
    ensure_dir,
    get_env,
    load_config,
    resolve_path,
    set_seed,
)
from faadjou_ai.metrics import compute_wer, normalize_text  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("finetune_asr")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tuning ASR Whisper wolof.")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--max_steps", type=int, default=None, help="Essai rapide. Obligatoire sans --full.")
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Borne le nombre d'extraits lus (recommandé : le paquet data/subset).",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Entraînement complet. Dernière étape, après l'évaluation sans entraînement.",
    )
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--output_dir", default=None, help="Ex. chemin Google Drive.")
    p.add_argument("--skip_train", action="store_true", help="Réévalue seulement.")
    return p.parse_args()


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


def prepare_dataset_features(ds, processor, sample_rate: int):
    def _map(batch):
        audio = batch["audio"]
        feats = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        labels = processor.tokenizer(batch["transcription"]).input_ids
        return {"input_features": feats, "labels": labels}

    cols = ds.column_names
    return ds.map(_map, remove_columns=cols, desc="Extraction features Whisper")


def build_wer_metrics(processor):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pred_str = [normalize_text(s) for s in pred_str]
        label_str = [normalize_text(s) for s in label_str]
        return {"wer": compute_wer(label_str, pred_str)}

    return compute_metrics


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    set_seed(int(config["project"]["seed"]))
    if args.max_samples is not None:
        config["audio"]["max_samples"] = args.max_samples
    ft = config["finetune"]
    max_steps = args.max_steps if args.max_steps is not None else ft.get("max_steps")
    if not args.full and not max_steps:
        raise SystemExit(
            "Le fine-tuning est la DERNIÈRE étape, et il doit être borné.\n"
            "Passez --max_steps 50 avec configs/config.subset.yaml, "
            "ou --full seulement après l'évaluation ASR sans entraînement."
        )
    results_dir = ensure_dir(resolve_path(config["paths"]["results_dir"]))
    output_dir = resolve_path(args.output_dir or ft["output_dir"])
    ensure_dir(output_dir)

    splits, medical = prepare_asr_datasets(config)
    base_model = ft["base_model"]
    sr = int(config["audio"]["sample_rate"])

    # --- Baseline avant fine-tuning ---
    baseline_rows = []
    eval_sets = [("test", splits["test"])]
    if medical is not None:
        eval_sets.append(("medical_test", medical))

    logger.info("Évaluation BASELINE (%s)…", base_model)
    for split_name, ds in eval_sets:
        res = evaluate_asr_on_dataset(
            base_model,
            ds,
            split_name,
            batch_size=int(config["asr"].get("eval_batch_size", 4)),
            use_fp16=bool(ft.get("fp16", True)),
            sample_rate=sr,
        )
        baseline_rows.append(
            {
                "etape": "avant",
                "modele": base_model,
                "jeu_de_test": res.split_name,
                "WER": res.wer,
                "CER": res.cer,
                "n_exemples": res.n_examples,
            }
        )

    if not args.skip_train:
        processor = WhisperProcessor.from_pretrained(base_model)
        model = WhisperForConditionalGeneration.from_pretrained(base_model)
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        if ft.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable()
            model.config.use_cache = False

        train_ds = prepare_dataset_features(splits["train"], processor, sr)
        eval_ds = prepare_dataset_features(splits["validation"], processor, sr)
        collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

        max_steps = args.max_steps if args.max_steps is not None else ft.get("max_steps")
        push = args.push_to_hub or bool(ft.get("push_to_hub", False))
        hub_id = ft.get("hub_model_id")
        token = get_env("HF_TOKEN")
        if push and not token:
            raise EnvironmentError(
                "push_to_hub demandé mais HF_TOKEN absent des variables d'environnement."
            )

        targs = Seq2SeqTrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=int(ft["per_device_train_batch_size"]),
            per_device_eval_batch_size=int(ft["per_device_eval_batch_size"]),
            gradient_accumulation_steps=int(ft["gradient_accumulation_steps"]),
            learning_rate=float(ft["learning_rate"]),
            warmup_steps=int(ft.get("warmup_steps", 50)),
            fp16=bool(ft.get("fp16", True)) and torch.cuda.is_available(),
            eval_strategy="steps",
            eval_steps=int(ft.get("eval_steps", 100)),
            save_steps=int(ft.get("save_steps", 100)),
            save_total_limit=int(ft.get("save_total_limit", 3)),
            logging_steps=int(ft.get("logging_steps", 10)),
            predict_with_generate=True,
            generation_max_length=225,
            report_to=[],
            seed=int(config["project"]["seed"]),
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=push,
            hub_model_id=hub_id,
            hub_token=token if push else None,
            max_steps=int(max_steps) if max_steps else -1,
            num_train_epochs=float(ft.get("num_train_epochs", 3)),
        )

        trainer = Seq2SeqTrainer(
            args=targs,
            model=model,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            compute_metrics=build_wer_metrics(processor),
            tokenizer=processor.feature_extractor,
        )

        last_ckpt = get_last_checkpoint(str(output_dir))
        if last_ckpt:
            logger.info("Reprise depuis checkpoint : %s", last_ckpt)
        trainer.train(resume_from_checkpoint=last_ckpt)
        trainer.save_model(str(output_dir / "final"))
        processor.save_pretrained(str(output_dir / "final"))
        if push:
            trainer.push_to_hub()
        release_asr_model(model)

    # --- Évaluation après ---
    ft_model_path = str(output_dir / "final")
    if not Path(ft_model_path).exists():
        ft_model_path = str(output_dir)
    logger.info("Évaluation APRÈS fine-tuning (%s)…", ft_model_path)

    after_rows = []
    for split_name, ds in eval_sets:
        res = evaluate_asr_on_dataset(
            ft_model_path,
            ds,
            split_name,
            batch_size=int(config["asr"].get("eval_batch_size", 4)),
            use_fp16=bool(ft.get("fp16", True)),
            sample_rate=sr,
        )
        after_rows.append(
            {
                "etape": "apres",
                "modele": ft_model_path,
                "jeu_de_test": res.split_name,
                "WER": res.wer,
                "CER": res.cer,
                "n_exemples": res.n_examples,
            }
        )

    compare = pd.DataFrame(baseline_rows + after_rows)

    # Décision : conserver si WER test général strictement meilleur
    before_wer = next(r["WER"] for r in baseline_rows if r["jeu_de_test"] == "test")
    after_wer = next(r["WER"] for r in after_rows if r["jeu_de_test"] == "test")
    keep = after_wer < before_wer
    decision = (
        "CONSERVER le modèle ajusté (WER test amélioré)."
        if keep
        else "NE PAS CONSERVER le modèle ajusté (WER test non amélioré). "
        "Revenir au checkpoint de base."
    )
    logger.info(
        "WER test avant=%.4f après=%.4f → %s", before_wer, after_wer, decision
    )
    compare["decision"] = decision
    compare["conserver_modele_ajuste"] = keep

    out = results_dir / "asr_finetune_compare.csv"
    compare.to_csv(out, index=False)
    logger.info("Écrit : %s", out)
    # Marqueur clair pour le jury
    marker = results_dir / "asr_finetune_decision.txt"
    marker.write_text(
        f"WER_avant={before_wer}\nWER_apres={after_wer}\n{decision}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
