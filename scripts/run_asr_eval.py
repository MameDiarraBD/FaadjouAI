#!/usr/bin/env python3
"""Évalue une liste de modèles ASR (sans entraînement) → results/asr_eval.csv.

Usage :
  python scripts/run_asr_eval.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Permet d'exécuter sans installation editable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faadjou_ai.asr import evaluate_asr_on_dataset  # noqa: E402
from faadjou_ai.audio import prepare_asr_datasets  # noqa: E402
from faadjou_ai.config_utils import ensure_dir, load_config, resolve_path, set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("run_asr_eval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Évaluation ASR wolof (WER/CER).")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Nombre max d'exemples train et test lus (ex. 200). N'ouvre pas tout le corpus.",
    )
    p.add_argument(
        "--models",
        default=None,
        help="Liste de modèles séparés par des virgules (défaut : asr.models).",
    )
    p.add_argument("--batch-size", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    set_seed(int(config["project"]["seed"]))
    if args.max_samples is not None:
        config["audio"]["max_samples"] = args.max_samples

    results_dir = ensure_dir(resolve_path(config["paths"]["results_dir"]))
    splits, medical = prepare_asr_datasets(config)

    if args.models:
        models = [item.strip() for item in args.models.split(",") if item.strip()]
    else:
        models = list(config["asr"]["models"])
    batch_size = int(args.batch_size or config["asr"].get("eval_batch_size", 4))
    use_fp16 = bool(config["asr"].get("use_fp16", True))
    sr = int(config["audio"]["sample_rate"])

    rows = []
    eval_sets = [("test", splits["test"])]
    if medical is not None:
        eval_sets.append(("medical_test", medical))
    else:
        logger.warning("medical_test absent — ligne explicite, sans WER inventé.")
        for model_id in models:
            rows.append(
                {
                    "modele": model_id,
                    "jeu_de_test": "medical_test",
                    "WER": None,
                    "CER": None,
                    "n_exemples": 0,
                    "duree_moyenne_inference_sec": None,
                    "erreur": "medical_test absent (déposez vos phrases dans data/audio/medical_test)",
                }
            )

    for model_id in models:
        for split_name, ds in eval_sets:
            logger.info("Éval %s sur %s (%d ex.)", model_id, split_name, len(ds))
            try:
                res = evaluate_asr_on_dataset(
                    model_id=model_id,
                    dataset=ds,
                    split_name=split_name,
                    batch_size=batch_size,
                    use_fp16=use_fp16,
                    sample_rate=sr,
                )
                rows.append(
                    {
                        "modele": res.model_id,
                        "jeu_de_test": res.split_name,
                        "WER": res.wer,
                        "CER": res.cer,
                        "n_exemples": res.n_examples,
                        "duree_moyenne_inference_sec": res.mean_inference_sec,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Échec %s / %s : %s", model_id, split_name, exc)
                rows.append(
                    {
                        "modele": model_id,
                        "jeu_de_test": split_name,
                        "WER": None,
                        "CER": None,
                        "n_exemples": len(ds),
                        "duree_moyenne_inference_sec": None,
                        "erreur": str(exc),
                    }
                )

    out = results_dir / "asr_eval.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("Écrit : %s", out)


if __name__ == "__main__":
    main()
