"""Chargement de la configuration YAML et utilitaires de reproductibilité."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def project_root() -> Path:
    """Racine du dépôt (parent de src/)."""
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Charge configs/config.yaml (ou un chemin fourni).

    POURQUOI centraliser : un seul fichier pour Colab et local, sans dupliquer
    les hyperparamètres dans chaque script.
    """
    if config_path is None:
        config_path = project_root() / "configs" / "config.yaml"
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config introuvable : {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("La config YAML doit être un dictionnaire.")
    return data


def resolve_path(relative: str | Path, root: Path | None = None) -> Path:
    """Résout un chemin relatif par rapport à la racine du projet."""
    root = root or project_root()
    p = Path(relative)
    return p if p.is_absolute() else root / p


def set_seed(seed: int) -> None:
    """Fixe les graines Python / NumPy (et torch si disponible).

    POURQUOI : splits train/val/test et simulation Sentinelle doivent être
    reproductibles pour le jury.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_env(name: str, default: str | None = None) -> str | None:
    """Lit une variable d'environnement (clés / tokens jamais dans le code)."""
    value = os.environ.get(name, default)
    if value is not None and value.strip() == "":
        return default
    return value
