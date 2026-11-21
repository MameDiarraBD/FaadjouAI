"""Module 1 — chargement et préparation des données audio ASR.

POURQUOI :
- Deux sources (Hugging Face ou dossier local) pour développer sans vos données
  puis évaluer sur vos enregistrements médicaux.
- Rééchantillonnage 16 kHz mono : format attendu par Whisper et la plupart des ASR.
- Split train/val/test à graine fixée ; le test n'est JAMAIS utilisé en entraînement.
- medical_test : jeu séparé (phrases médicales) pour mesurer la robustesse clinique.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from faadjou_ai.config_utils import resolve_path, set_seed

logger = logging.getLogger(__name__)


def load_waveform(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """Charge un fichier audio, mono, à sample_rate Hz."""
    import librosa

    y, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return y.astype(np.float32)


def _resample_dataset(ds, sample_rate: int):
    """Force le cast Audio vers le sample_rate cible."""
    from datasets import Audio
    """Force le cast Audio vers le sample_rate cible."""
    if "audio" not in ds.column_names:
        raise ValueError("Le dataset doit contenir une colonne 'audio'.")
    return ds.cast_column("audio", Audio(sampling_rate=sample_rate))


def split_parquet_paths(directory: str | Path) -> tuple[list[Path], list[Path]]:
    """Sépare les fichiers train-*.parquet et test-*.parquet d'un dossier.

    Ignore les doublons Windows du type ``train-00006 (1).parquet``.
    POURQUOI : les corpus déjà découpés (train/test) ne doivent pas être
    remélangés, sinon le test officiel fuite dans l'entraînement.
    """
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Dossier parquet introuvable : {root}")
    train: list[Path] = []
    test: list[Path] = []
    for path in sorted(root.glob("*.parquet")):
        if " (" in path.name:
            logger.info("Doublon ignoré : %s", path.name)
            continue
        name = path.name.lower()
        if name.startswith("test"):
            test.append(path)
        else:
            train.append(path)
    if not train and not test:
        raise FileNotFoundError(f"Aucun .parquet dans {root}")
    return train, test


def _standardize_text_column(ds: Dataset, text_column: str) -> Dataset:
    if text_column not in ds.column_names and "transcription" in ds.column_names:
        text_column = "transcription"
    if text_column not in ds.column_names:
        raise ValueError(
            f"Colonne texte '{text_column}' absente. Colonnes : {ds.column_names}"
        )
    if text_column != "transcription":
        ds = ds.rename_column(text_column, "transcription")
    keep = [c for c in ("audio", "transcription") if c in ds.column_names]
    return ds.select_columns(keep)


def load_parquet_sources(
    sources: list[dict[str, Any]],
    sample_rate: int,
    validation_ratio: float,
    seed: int,
) -> DatasetDict:
    """Charge un ou plusieurs dossiers de parquet Hugging Face (audio déjà inclus).

    Chaque source peut filtrer une langue (ex. ne garder que ``wo`` dans le
    dossier wolof-french, qui mélange wolof et français dans le même fichier).
    """
    train_parts: list[Dataset] = []
    test_parts: list[Dataset] = []

    for src in sources:
        root = resolve_path(src["path"])
        text_column = src.get("text_column", "transcription")
        train_files, test_files = split_parquet_paths(root)
        data_files: dict[str, list[str]] = {}
        if train_files:
            data_files["train"] = [str(p) for p in train_files]
        if test_files:
            data_files["test"] = [str(p) for p in test_files]
        logger.info(
            "Parquet %s : %d train, %d test",
            root,
            len(train_files),
            len(test_files),
        )
        from datasets import load_dataset

        loaded = load_dataset("parquet", data_files=data_files)
        lang_col = src.get("language_column")
        lang_keep = src.get("language_keep")

        def _take(split_name: str) -> Dataset | None:
            if split_name not in loaded:
                return None
            part = loaded[split_name]
            if lang_col and lang_keep:
                if lang_col not in part.column_names:
                    raise ValueError(
                        f"{root} : colonne langue '{lang_col}' absente."
                    )
                part = part.filter(lambda row, col=lang_col, keep=lang_keep: row[col] == keep)
            part = _standardize_text_column(part, text_column)
            return _resample_dataset(part, sample_rate)

        train_ds = _take("train")
        test_ds = _take("test")
        if train_ds is not None:
            train_parts.append(train_ds)
        if test_ds is not None:
            test_parts.append(test_ds)

    if not train_parts:
        raise ValueError("Aucun split train dans les dossiers parquet.")

    from datasets import concatenate_datasets

    train_all = concatenate_datasets(train_parts) if len(train_parts) > 1 else train_parts[0]
    test_all = (
        concatenate_datasets(test_parts) if len(test_parts) > 1 else test_parts[0]
        if test_parts
        else None
    )
    # Le test officiel n'entre jamais dans le train. La validation est tirée du train.
    if test_all is None:
        logger.warning("Pas de fichiers test-*.parquet : découpage train/val/test depuis le train.")
        return split_dataset(train_all, 0.8, validation_ratio, 0.1, seed)
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio doit être entre 0 et 1.")
    carved = train_all.train_test_split(test_size=validation_ratio, seed=seed)
    from datasets import DatasetDict

    return DatasetDict(
        {
            "train": carved["train"],
            "validation": carved["test"],
            "test": test_all,
        }
    )


def load_parquet_limited(
    sources: list[dict[str, Any]],
    sample_rate: int,
    validation_ratio: float,
    seed: int,
    max_samples: int,
):
    """Au plus max_samples extraits de train et autant de test, lus en flux."""
    from datasets import Audio, Dataset, DatasetDict

    from faadjou_ai.subset_io import sample_sources

    def _as_dataset(rows: list[dict[str, Any]]) -> Dataset:
        ds = Dataset.from_list(rows)
        return ds.cast_column("audio", Audio(sampling_rate=sample_rate))

    train_rows, _ = sample_sources(sources, "train", max_samples, seed)
    test_rows, _ = sample_sources(sources, "test", max_samples, seed + 1)
    if len(train_rows) < 3:
        raise ValueError(
            f"Pas assez d'extraits train ({len(train_rows)}) pour un sous-échantillon."
        )
    if not test_rows:
        raise ValueError("Aucun extrait test dans les parquet test-*.")
    train_ds = _as_dataset(train_rows)
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio doit être entre 0 et 1.")
    carved = train_ds.train_test_split(test_size=validation_ratio, seed=seed)
    return DatasetDict(
        {
            "train": carved["train"],
            "validation": carved["test"],
            "test": _as_dataset(test_rows),
        }
    )


def load_from_huggingface(
    dataset_id: str,
    split: str = "train",
    sample_rate: int = 16000,
    text_column_candidates: tuple[str, ...] = ("transcription", "text", "sentence"),
) -> Dataset:
    """Charge un corpus HF et normalise les colonnes audio / transcription."""
    if not dataset_id:
        raise ValueError(
            "audio.source.huggingface_id est vide. "
            "Renseignez un identifiant HF ou passez source.type=local."
        )
    from datasets import load_dataset

    raw = load_dataset(dataset_id, split=split)
    text_col = None
    for cand in text_column_candidates:
        if cand in raw.column_names:
            text_col = cand
            break
    if text_col is None:
        raise ValueError(
            f"Aucune colonne de transcription trouvée parmi {text_column_candidates} "
            f"dans {dataset_id}. Colonnes : {raw.column_names}"
        )
    if text_col != "transcription":
        raw = raw.rename_column(text_col, "transcription")
    if "audio" not in raw.column_names:
        raise ValueError(f"Pas de colonne 'audio' dans {dataset_id}.")
    return _resample_dataset(raw, sample_rate)


def load_from_local(
    local_dir: str | Path,
    metadata_csv: str = "metadata.csv",
    sample_rate: int = 16000,
) -> Dataset:
    """Charge un dossier local : metadata.csv avec colonnes file, transcription.

    Chaque 'file' est un chemin relatif au dossier local_dir.
    """
    root = resolve_path(local_dir)
    meta_path = root / metadata_csv
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"metadata.csv introuvable : {meta_path}. "
            "Attendu : colonnes 'file' et 'transcription'."
        )
    df = pd.read_csv(meta_path)
    required = {"file", "transcription"}
    if not required.issubset(df.columns):
        raise ValueError(f"{meta_path} doit contenir les colonnes {required}.")

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        audio_path = root / str(row["file"])
        if not audio_path.is_file():
            logger.warning("Fichier audio manquant, ignoré : %s", audio_path)
            continue
        records.append(
            {
                "audio": str(audio_path),
                "transcription": str(row["transcription"]),
            }
        )
    if not records:
        raise ValueError(f"Aucun fichier audio valide dans {root}.")
    from datasets import Dataset

    ds = Dataset.from_list(records)
    return _resample_dataset(ds, sample_rate)


def maybe_subsample(ds: Dataset, max_samples: int | None, seed: int) -> Dataset:
    """Sous-échantillonne pour un smoke test rapide (ex. 200 exemples)."""
    if max_samples is None or max_samples <= 0 or len(ds) <= max_samples:
        return ds
    return ds.shuffle(seed=seed).select(range(max_samples))


def split_dataset(
    ds: Dataset,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> DatasetDict:
    """Sépare train / validation / test. Le test n'est jamais utilisé en train.

    POURQUOI shuffle + ratios dans la config : reproductibilité du jury.
    """
    total = train_ratio + validation_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Les ratios de split doivent sommer à 1 (obtenu {total}).")
    if len(ds) < 3:
        raise ValueError("Pas assez d'exemples pour train/val/test.")

    from datasets import DatasetDict

    first = ds.train_test_split(test_size=test_ratio, seed=seed)
    remain_ratio = validation_ratio / (train_ratio + validation_ratio)
    second = first["train"].train_test_split(test_size=remain_ratio, seed=seed)
    return DatasetDict(
        {
            "train": second["train"],
            "validation": second["test"],
            "test": first["test"],
        }
    )


def load_medical_test(
    medical_dir: str | Path,
    metadata_csv: str = "metadata.csv",
    sample_rate: int = 16000,
) -> Dataset | None:
    """Charge le jeu medical_test s'il existe, sinon None (pas d'échec silencieux loggé)."""
    root = resolve_path(medical_dir)
    meta = root / metadata_csv
    if not meta.is_file():
        logger.info(
            "medical_test absent (%s) — l'évaluation médicale sera ignorée.",
            meta,
        )
        return None
    return load_from_local(root, metadata_csv=metadata_csv, sample_rate=sample_rate)


def prepare_asr_datasets(config: dict[str, Any]) -> tuple[DatasetDict, Dataset | None]:
    """Point d'entrée : charge la source, sous-échantillonne, split, medical_test."""
    seed = int(config["project"]["seed"])
    set_seed(seed)
    audio_cfg = config["audio"]
    sr = int(audio_cfg["sample_rate"])
    source = audio_cfg["source"]

    if source["type"] == "huggingface":
        ds = load_from_huggingface(
            dataset_id=source.get("huggingface_id") or "",
            split=source.get("huggingface_split", "train"),
            sample_rate=sr,
        )
        ds = maybe_subsample(ds, audio_cfg.get("max_samples"), seed)
        splits_cfg = audio_cfg["splits"]
        splits = split_dataset(
            ds,
            train_ratio=float(splits_cfg["train_ratio"]),
            validation_ratio=float(splits_cfg["validation_ratio"]),
            test_ratio=float(splits_cfg["test_ratio"]),
            seed=seed,
        )
    elif source["type"] == "local":
        ds = load_from_local(
            local_dir=source["local_dir"],
            metadata_csv=source.get("metadata_csv", "metadata.csv"),
            sample_rate=sr,
        )
        ds = maybe_subsample(ds, audio_cfg.get("max_samples"), seed)
        splits_cfg = audio_cfg["splits"]
        splits = split_dataset(
            ds,
            train_ratio=float(splits_cfg["train_ratio"]),
            validation_ratio=float(splits_cfg["validation_ratio"]),
            test_ratio=float(splits_cfg["test_ratio"]),
            seed=seed,
        )
    elif source["type"] == "parquet":
        # Les fichiers test-*.parquet restent le test. On ne les remélange pas.
        max_samples = audio_cfg.get("max_samples")
        if max_samples:
            # Lecture bornée : on n'ouvre pas les gigaoctets du corpus complet.
            splits = load_parquet_limited(
                sources=source["parquet_sources"],
                sample_rate=sr,
                validation_ratio=float(audio_cfg["splits"]["validation_ratio"]),
                seed=seed,
                max_samples=int(max_samples),
            )
        else:
            splits = load_parquet_sources(
                sources=source["parquet_sources"],
                sample_rate=sr,
                validation_ratio=float(audio_cfg["splits"]["validation_ratio"]),
                seed=seed,
            )
    else:
        raise ValueError(f"source.type inconnu : {source['type']}")
    medical = load_medical_test(
        audio_cfg.get("medical_test_dir", "data/audio/medical_test"),
        metadata_csv=audio_cfg.get("medical_test_metadata", "metadata.csv"),
        sample_rate=sr,
    )
    logger.info(
        "Splits : train=%d val=%d test=%d medical_test=%s",
        len(splits["train"]),
        len(splits["validation"]),
        len(splits["test"]),
        len(medical) if medical is not None else "absent",
    )
    return splits, medical
