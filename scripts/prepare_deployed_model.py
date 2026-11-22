#!/usr/bin/env python3
"""Extraire et vérifier l’archive du modèle fine-tuné téléchargée depuis Kaggle."""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "archive",
        nargs="?",
        default=str(ROOT / "faadjou_model_ft.zip"),
        help="Archive ZIP du modèle téléchargée depuis Kaggle.",
    )
    parser.add_argument(
        "--destination",
        default=str(ROOT / "checkpoints" / "whisper-wolof-ft" / "final"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = Path(args.archive)
    destination = Path(args.destination)

    if not archive.is_file():
        raise SystemExit(f"Archive modèle introuvable : {archive}")
    if archive.stat().st_size < 100_000_000:
        raise SystemExit("Archive trop petite pour être un modèle Whisper complet.")

    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad is not None:
            raise SystemExit(f"Archive ZIP corrompue au fichier : {bad}")

        with tempfile.TemporaryDirectory(prefix="faadjou-model-") as temp_name:
            temp_dir = Path(temp_name)
            z.extractall(temp_dir)
            weight_files = list(temp_dir.rglob("*.safetensors"))
            weight_files += list(temp_dir.rglob("pytorch_model.bin"))
            if not weight_files:
                raise SystemExit("Aucun poids de modèle trouvé dans l’archive.")

            model_dir = weight_files[0].parent
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(model_dir, destination)

    required = ("config.json", "preprocessor_config.json", "tokenizer.json")
    missing = [name for name in required if not (destination / name).is_file()]
    if missing:
        print(f"Attention : fichiers possiblement absents : {missing}", file=sys.stderr)

    weights = list(destination.glob("*.safetensors")) + list(destination.glob("pytorch_model.bin"))
    if not weights:
        raise SystemExit("Poids du modèle absents après extraction.")
    print(f"Modèle déployable installé dans : {destination}")


if __name__ == "__main__":
    main()
