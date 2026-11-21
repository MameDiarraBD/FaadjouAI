"""Simulation d'événements de triage anonymes pour Sentinelle.

POURQUOI simuler :
- Pas de données patients réelles au stade concours.
- On injecte des pics connus (après fortes pluies + décalage) pour évaluer
  la détection. Chaque sortie est marquée SIMULÉ.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from faadjou_ai.config_utils import ensure_dir, resolve_path, set_seed


@dataclass
class InjectedSpike:
    zone: str
    category: str
    rain_date: str
    spike_start: str
    spike_end: str
    multiplier: float
    simulated: bool = True


def _daterange(start: date, months: int) -> list[date]:
    end = start + timedelta(days=int(months * 30.44))
    days: list[date] = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def _seasonal_factor(d: date) -> float:
    """Saisonnalité simple (pics autour de la saison des pluies, ~juin–oct)."""
    # Courbe sinusoïdale centrée sur août (mois 8)
    angle = 2 * np.pi * (d.timetuple().tm_yday - 213) / 365.0
    return 1.0 + 0.35 * np.sin(angle)


def simulate_events(
    config: dict[str, Any],
    rain_peaks: dict[str, list[date]] | None = None,
) -> tuple[pd.DataFrame, list[InjectedSpike]]:
    """Génère des événements SIMULÉS + la liste des pics injectés.

    rain_peaks : dates de fortes pluies par zone (si None, on tire des dates).
    """
    seed = int(config["project"]["seed"])
    set_seed(seed)
    rng = np.random.default_rng(seed)

    scfg = config["sentinelle"]
    zones = list(scfg["zones"])
    categories = list(scfg["symptom_categories"])
    sim = scfg["simulation"]
    start = datetime.strptime(sim["start_date"], "%Y-%m-%d").date()
    days = _daterange(start, int(sim["months"]))
    base = float(sim["base_daily_rate"])
    lag = int(sim["rain_lag_days"])
    mult = float(sim["rain_spike_multiplier"])
    n_spikes = int(sim["n_injected_spikes"])

    # Choisir des pics de pluie injectés si non fournis
    if rain_peaks is None:
        rain_peaks = {z: [] for z in zones}
        candidates = days[60:-60] if len(days) > 150 else days[20:-20]
        for _ in range(n_spikes):
            zone = str(rng.choice(zones))
            rain_day = candidates[int(rng.integers(0, len(candidates)))]
            rain_peaks[zone].append(rain_day)

    spikes: list[InjectedSpike] = []
    for zone, rains in rain_peaks.items():
        for rain_day in rains:
            spike_start = rain_day + timedelta(days=lag)
            spike_end = spike_start + timedelta(days=4)
            cat = str(rng.choice(categories))
            spikes.append(
                InjectedSpike(
                    zone=zone,
                    category=cat,
                    rain_date=rain_day.isoformat(),
                    spike_start=spike_start.isoformat(),
                    spike_end=spike_end.isoformat(),
                    multiplier=mult,
                )
            )

    rows: list[dict[str, Any]] = []
    event_id = 0
    urgences = ["faible", "moyenne", "elevee", "vitale"]
    urgence_p = [0.55, 0.30, 0.12, 0.03]

    for d in days:
        for zone in zones:
            for cat in categories:
                rate = base * _seasonal_factor(d) / len(categories)
                # Appliquer multiplicateur si dans une fenêtre de pic injecté
                for sp in spikes:
                    if (
                        sp.zone == zone
                        and sp.category == cat
                        and sp.spike_start <= d.isoformat() <= sp.spike_end
                    ):
                        rate *= sp.multiplier
                n = int(rng.poisson(max(rate, 0.01)))
                for _ in range(n):
                    event_id += 1
                    rows.append(
                        {
                            "event_id": f"SIM-{event_id:06d}",
                            "date": d.isoformat(),
                            "zone": zone,
                            "categorie_symptomes": cat,
                            "urgence": str(rng.choice(urgences, p=urgence_p)),
                            "SIMULE": True,
                        }
                    )

    df = pd.DataFrame(rows)
    return df, spikes


def save_simulation(
    df: pd.DataFrame,
    spikes: list[InjectedSpike],
    results_dir: str | Path,
) -> tuple[Path, Path]:
    """Écrit les fichiers avec mention SIMULÉ dans le nom."""
    out = ensure_dir(resolve_path(results_dir))
    events_path = out / "sentinelle_events_SIMULE.csv"
    spikes_path = out / "sentinelle_injected_spikes_SIMULE.json"
    df.to_csv(events_path, index=False)
    with spikes_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in spikes], f, ensure_ascii=False, indent=2)
    return events_path, spikes_path
