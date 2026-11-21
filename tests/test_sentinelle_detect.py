"""Tests — détection d'un pic simple sur une série construite."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faadjou_ai.sentinelle.detect import daily_counts, detect_alerts, evaluate_detection


def test_detect_simple_spike() -> None:
    # Baseline ~5, puis pic à 30
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    rows = []
    for i, d in enumerate(dates):
        count = 30 if 25 <= i <= 28 else 5
        for _ in range(count):
            rows.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "zone": "Dakar",
                    "categorie_symptomes": "febrile",
                    "urgence": "moyenne",
                    "SIMULE": True,
                }
            )
    events = pd.DataFrame(rows)
    counts = daily_counts(events)
    alerts = detect_alerts(
        counts, rolling_window=14, k_sigma=2.0, min_count_per_zone=5
    )
    spike_alerts = alerts[
        (alerts["date"] >= "2024-01-26")
        & (alerts["date"] <= "2024-01-29")
        & (alerts["alert"])
    ]
    assert len(spike_alerts) >= 1


def test_evaluate_detection_counts_spike() -> None:
    dates = pd.date_range("2024-03-01", periods=30, freq="D")
    rows = []
    for i, d in enumerate(dates):
        count = 40 if 15 <= i <= 18 else 4
        rows.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "zone": "Thiès",
                "categorie_symptomes": "gastro_intestinal",
                "count": count,
            }
        )
    counts = pd.DataFrame(rows)
    alerts = detect_alerts(counts, rolling_window=10, k_sigma=2.0, min_count_per_zone=5)
    spikes = [
        {
            "zone": "Thiès",
            "category": "gastro_intestinal",
            "rain_date": "2024-03-10",
            "spike_start": "2024-03-16",
            "spike_end": "2024-03-19",
            "multiplier": 3.0,
            "simulated": True,
        }
    ]
    metrics = evaluate_detection(alerts, spikes)
    assert metrics["n_injected_spikes"] == 1
    assert metrics["n_detected"] == 1
    assert "SIMULÉ" in metrics["data_note"]
