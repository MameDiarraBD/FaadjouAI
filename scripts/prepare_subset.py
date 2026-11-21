#!/usr/bin/env python3
"""Produit un petit paquet audio (train + test + phrases médicales).

Le paquet reste sous le plafond d'octets de la config (défaut 190 Mo) pour
être déposé sur Google Drive, Kaggle ou un dataset Hugging Face privé.
Aucun chemin absolu n'est écrit dans le manifeste : seulement des chemins
relatifs à la racine du projet.

Usage :
  python scripts/prepare_subset.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faadjou_ai.config_utils import load_config, resolve_path, set_seed  # noqa: E402
from faadjou_ai.subset_io import sample_sources, write_audio_parquet, write_manifest  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("prepare_subset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sous-ensemble audio portable.")
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def _copy_medical(src: Path, dest: Path) -> dict[str, int]:
    """Copie metadata.csv et les fichiers listés, s'ils existent."""
    meta = src / "metadata.csv"
    if not meta.is_file():
        dest.mkdir(parents=True, exist_ok=True)
        note = dest / "ABSENT.txt"
        note.write_text(
            "Aucune phrase médicale fournie.\n"
            "Déposez metadata.csv (colonnes file, transcription) et les audios "
            "dans ce dossier, puis relancez prepare_subset.py.\n",
            encoding="utf-8",
        )
        return {"n_files": 0, "bytes": 0}
    import pandas as pd

    frame = pd.read_csv(meta)
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    total = 0
    kept_rows = []
    for _, row in frame.iterrows():
        audio = src / str(row["file"])
        if not audio.is_file():
            logger.warning("Audio médical manquant : %s", audio)
            continue
        target = dest / str(row["file"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(audio, target)
        copied += 1
        total += target.stat().st_size
        kept_rows.append({"file": str(row["file"]), "transcription": row["transcription"]})
    pd.DataFrame(kept_rows).to_csv(dest / "metadata.csv", index=False)
    return {"n_files": copied, "bytes": total}


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    seed = int(config["project"]["seed"])
    set_seed(seed)
    subset = config.get("subset", {})
    sources = config["audio"]["source"].get("parquet_sources") or []
    if not sources:
        raise SystemExit("audio.source.parquet_sources est vide dans la config.")

    out = resolve_path(subset.get("output_dir", "data/subset"))
    n_train = int(subset.get("n_train", 210))
    n_test = int(subset.get("n_test", 90))
    max_clip = int(subset.get("max_clip_bytes", 500_000))
    budget = int(float(subset.get("max_total_mb", 190)) * 1024 * 1024)

    train_rows, train_bytes = sample_sources(
        sources, "train", n_train, seed, max_clip_bytes=max_clip, byte_budget=budget
    )
    test_rows, test_bytes = sample_sources(
        sources,
        "test",
        n_test,
        seed + 1,
        max_clip_bytes=max_clip,
        byte_budget=max(0, budget - train_bytes),
    )
    write_audio_parquet(train_rows, out / "train.parquet")
    write_audio_parquet(test_rows, out / "test.parquet")

    medical_src = resolve_path(config["audio"].get("medical_test_dir", "data/audio/medical_test"))
    medical = _copy_medical(medical_src, out / "medical_test")
    total = train_bytes + test_bytes + medical["bytes"]
    manifest = {
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "n_medical": medical["n_files"],
        "bytes_train": train_bytes,
        "bytes_test": test_bytes,
        "bytes_medical": medical["bytes"],
        "bytes_total": total,
        "max_total_bytes": budget,
        "under_budget": total <= budget,
        "seed": seed,
        "output_dir": subset.get("output_dir", "data/subset"),
        "note": (
            "Échantillon borné. Les fichiers test.parquet viennent uniquement "
            "des parquet nommés test-*. Les phrases médicales ne sont pas dans train."
        ),
    }
    write_manifest(out / "MANIFEST.json", manifest)
    logger.info(
        "Paquet %s : train=%d test=%d medical=%d total=%.1f Mo (plafond %.1f Mo)",
        out,
        len(train_rows),
        len(test_rows),
        medical["n_files"],
        total / (1024 * 1024),
        budget / (1024 * 1024),
    )
    if not manifest["under_budget"]:
        raise SystemExit("Le paquet dépasse le plafond. Baissez n_train/n_test ou max_clip_bytes.")
    if len(train_rows) < 10 or len(test_rows) < 10:
        raise SystemExit("Trop peu d'extraits lus. Vérifiez les dossiers parquet de la config.")


if __name__ == "__main__":
    main()
