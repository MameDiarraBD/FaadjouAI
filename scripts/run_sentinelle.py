#!/usr/bin/env python3
"""Pipeline Sentinelle : simulation → météo → détection → évaluation + graphique.

Sorties :
  results/sentinelle_events_SIMULE.csv
  results/sentinelle_injected_spikes_SIMULE.json
  results/sentinelle_eval.csv
  results/sentinelle_pic.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faadjou_ai.config_utils import ensure_dir, load_config, resolve_path, set_seed  # noqa: E402
from faadjou_ai.sentinelle.detect import (  # noqa: E402
    daily_counts,
    detect_alerts,
    evaluate_detection,
)
from faadjou_ai.sentinelle.plot import plot_spike_with_rain  # noqa: E402
from faadjou_ai.sentinelle.simulate import save_simulation, simulate_events  # noqa: E402
from faadjou_ai.sentinelle.weather import (  # noqa: E402
    ATTRIBUTION,
    fetch_all_zones,
    heavy_rain_dates,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("run_sentinelle")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sentinelle — alerte précoce SIMULÉE.")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument(
        "--skip-weather",
        action="store_true",
        help="Ne pas appeler Open-Meteo (pics purement aléatoires).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    set_seed(int(config["project"]["seed"]))
    results_dir = ensure_dir(resolve_path(config["paths"]["results_dir"]))

    rain_peaks = None
    weather = pd.DataFrame()
    if not args.skip_weather:
        logger.info("Téléchargement / cache météo Open-Meteo…")
        logger.info(ATTRIBUTION)
        weather = fetch_all_zones(config)
        if not weather.empty:
            weather_path = results_dir / "sentinelle_weather_SIMULE_context.csv"
            # La météo est réelle (Open-Meteo) ; le nom rappelle le contexte Sentinelle
            weather.to_csv(weather_path, index=False)
            rain_peaks = {}
            for zone in config["sentinelle"]["zones"]:
                rains = heavy_rain_dates(weather, zone, quantile=0.97)
                # Limiter pour coller à n_injected_spikes globalement
                rain_peaks[zone] = rains[:1]  # 1 date forte pluie / zone max
            # Si trop peu, laisser simulate tirer aléatoirement
            if sum(len(v) for v in rain_peaks.values()) == 0:
                rain_peaks = None

    events, spikes = simulate_events(config, rain_peaks=rain_peaks)
    events_path, spikes_path = save_simulation(
        events, spikes, config["paths"]["results_dir"]
    )
    logger.info("Événements SIMULÉS : %s (%d lignes)", events_path, len(events))
    logger.info("Pics injectés : %s (%d)", spikes_path, len(spikes))

    det_cfg = config["sentinelle"]["detection"]
    counts = daily_counts(events)
    alerts = detect_alerts(
        counts,
        rolling_window=int(det_cfg["rolling_window"]),
        k_sigma=float(det_cfg["k_sigma"]),
        min_count_per_zone=int(det_cfg["min_count_per_zone"]),
    )
    alerts_path = results_dir / "sentinelle_alerts_SIMULE.csv"
    alerts.to_csv(alerts_path, index=False)

    from dataclasses import asdict

    spikes_dicts = [asdict(s) for s in spikes]
    metrics = evaluate_detection(alerts, spikes_dicts)
    metrics["attribution_meteo"] = ATTRIBUTION
    eval_path = results_dir / "sentinelle_eval.csv"
    pd.DataFrame([metrics]).to_csv(eval_path, index=False)
    logger.info("Éval Sentinelle (SIMULÉ) : %s", metrics)

    # Graphique du premier pic
    if spikes_dicts:
        pic_path = results_dir / "sentinelle_pic.png"
        # Joindre alertes aux counts pour le scatter
        plot_df = alerts.copy()
        plot_spike_with_rain(plot_df, weather, spikes_dicts[0], pic_path)
        logger.info("Graphique : %s (données événements SIMULÉES)", pic_path)
    else:
        logger.warning("Aucun pic injecté — pas de graphique.")


if __name__ == "__main__":
    main()
