#!/usr/bin/env python3
"""Évalue le pipeline de triage sur tests/scenarios.csv.

Métriques :
- justesse de la spécialité
- rappel urgences vitales (prioritaire) + liste des urgences manquées
- taux de JSON valide
- latence moyenne

Sorties : results/triage_eval.csv, results/triage_errors.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faadjou_ai.config_utils import ensure_dir, load_config, resolve_path, set_seed  # noqa: E402
from faadjou_ai.llm import build_llm  # noqa: E402
from faadjou_ai.metrics import normalize_text  # noqa: E402
from faadjou_ai.pipeline import orient  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("run_triage_eval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Évaluation triage Faadjou Wax.")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument(
        "--scenarios",
        default=None,
        help="CSV scénarios (défaut : paths.scenarios_file ou template).",
    )
    p.add_argument(
        "--use-template",
        action="store_true",
        help="Utiliser tests/scenarios_template.csv (exemples fictifs).",
    )
    return p.parse_args()


def load_scenarios(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Scénarios introuvables : {path}. "
            "Copiez tests/scenarios_template.csv vers tests/scenarios.csv "
            "ou passez --use-template."
        )
    df = pd.read_csv(path)
    required = {"id", "texte_fr", "specialite_attendue", "urgence_attendue"}
    if not required.issubset(df.columns):
        raise ValueError(f"Colonnes requises manquantes : {required - set(df.columns)}")
    return df


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    set_seed(int(config["project"]["seed"]))
    results_dir = ensure_dir(resolve_path(config["paths"]["results_dir"]))

    if args.use_template:
        scen_path = resolve_path(config["paths"]["scenarios_template"])
    elif args.scenarios:
        scen_path = resolve_path(args.scenarios)
    else:
        scen_path = resolve_path(config["paths"]["scenarios_file"])
        if not scen_path.is_file():
            logger.warning(
                "%s absent — repli sur le template fictif.", scen_path
            )
            scen_path = resolve_path(config["paths"]["scenarios_template"])

    scenarios = load_scenarios(scen_path)
    logger.info("Chargement LLM…")
    llm = build_llm(config)

    detail_rows = []
    missed_vital = []
    n_spec_ok = 0
    n_json_ok = 0
    n_vital = 0
    n_vital_hit = 0
    latencies = []

    for _, row in scenarios.iterrows():
        text = str(row["texte_fr"])
        # Note : audio_wolof_optionnel peut être branché plus tard (ASR → translate)
        t0 = time.perf_counter()
        out = orient(config, llm, french_text=text)["triage"]
        latency = time.perf_counter() - t0
        latencies.append(latency)

        pred_spec = out.get("specialite", "")
        pred_urg = out.get("urgence", "")
        json_ok = bool(out.get("json_valide", False))
        if json_ok:
            n_json_ok += 1

        exp_spec = str(row["specialite_attendue"])
        exp_urg = str(row["urgence_attendue"])
        spec_ok = normalize_text(pred_spec) == normalize_text(exp_spec)
        if spec_ok:
            n_spec_ok += 1

        if normalize_text(exp_urg) == "vitale":
            n_vital += 1
            if normalize_text(pred_urg) == "vitale":
                n_vital_hit += 1
            else:
                missed_vital.append(
                    {
                        "id": row["id"],
                        "texte_fr": text,
                        "urgence_attendue": exp_urg,
                        "urgence_predite": pred_urg,
                        "specialite_predite": pred_spec,
                        "signaux_alarme": out.get("signaux_alarme"),
                        "red_flags_detectes": out.get("red_flags_detectes"),
                    }
                )

        detail_rows.append(
            {
                "id": row["id"],
                "specialite_attendue": exp_spec,
                "specialite_predite": pred_spec,
                "specialite_ok": spec_ok,
                "urgence_attendue": exp_urg,
                "urgence_predite": pred_urg,
                "json_valide": json_ok,
                "latence_sec": latency,
                "regles_appliquees": out.get("regles_appliquees", False),
                "fallback": out.get("fallback", False),
            }
        )
        logger.info(
            "%s → spec=%s urg=%s json=%s (%.2fs)",
            row["id"],
            pred_spec,
            pred_urg,
            json_ok,
            latency,
        )

    n = len(scenarios)
    summary = {
        "n_scenarios": n,
        "specialite_accuracy": n_spec_ok / n if n else float("nan"),
        "rappel_urgences_vitales": (n_vital_hit / n_vital) if n_vital else float("nan"),
        "n_urgences_vitales": n_vital,
        "n_urgences_vitales_detectees": n_vital_hit,
        "n_urgences_vitales_manquees": len(missed_vital),
        "taux_json_valide": n_json_ok / n if n else float("nan"),
        "latence_moyenne_sec": sum(latencies) / n if n else float("nan"),
        "note": (
            "Le système ORIENTE et ne remplace pas un médecin. "
            "Scénarios potentiellement fictifs — à valider cliniquement."
        ),
    }

    # Afficher les urgences manquées une par une (exigence cahier des charges)
    if missed_vital:
        logger.warning("=== Urgences vitales MANQUÉES (%d) ===", len(missed_vital))
        for m in missed_vital:
            logger.warning(
                "  [%s] attendu=vitale prédit=%s | %s",
                m["id"],
                m["urgence_predite"],
                m["texte_fr"][:80],
            )
    else:
        logger.info("Aucune urgence vitale manquée sur ce jeu.")

    eval_path = results_dir / "triage_eval.csv"
    err_path = results_dir / "triage_errors.csv"
    pd.DataFrame([summary]).to_csv(eval_path, index=False)
    # errors = détails + manquées
    errors_df = pd.DataFrame(detail_rows)
    if missed_vital:
        missed_df = pd.DataFrame(missed_vital)
        missed_df["type_erreur"] = "urgence_vitale_manquee"
        errors_df = pd.concat([errors_df, missed_df], ignore_index=True, sort=False)
    errors_df.to_csv(err_path, index=False)
    logger.info("Écrit : %s et %s", eval_path, err_path)
    logger.info("Résumé : %s", summary)


if __name__ == "__main__":
    main()
