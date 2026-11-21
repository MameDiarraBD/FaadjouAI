"""Détection d'anomalies : moyenne mobile + k × écart-type.

POURQUOI cette méthode classique :
- Interprétable devant un jury (pas une boîte noire).
- Paramètres k et fenêtre dans la config.
- Seuil minimal de comptage par zone pour limiter la ré-identification.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def daily_counts(events: pd.DataFrame) -> pd.DataFrame:
    """Comptage quotidien par zone et catégorie (données SIMULÉES attendues)."""
    if events.empty:
        return pd.DataFrame(columns=["date", "zone", "categorie_symptomes", "count"])
    g = (
        events.groupby(["date", "zone", "categorie_symptomes"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    return g


def detect_alerts(
    counts: pd.DataFrame,
    rolling_window: int = 14,
    k_sigma: float = 2.0,
    min_count_per_zone: int = 5,
) -> pd.DataFrame:
    """Alerte si count > rolling_mean + k * rolling_std, et count >= seuil min.

    Le rolling est calculé sur les jours PRÉCÉDENTS (shift 1) pour ne pas
    utiliser le jour courant dans la baseline — sinon le pic s'auto-annule.
    """
    if counts.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "zone",
                "categorie_symptomes",
                "count",
                "baseline_mean",
                "baseline_std",
                "threshold",
                "alert",
            ]
        )

    frames: list[pd.DataFrame] = []
    for (zone, cat), sub in counts.groupby(["zone", "categorie_symptomes"]):
        s = sub.sort_values("date").copy()
        s["date"] = pd.to_datetime(s["date"])
        # Compléter les jours manquants à 0 (uniquement la série count)
        all_dates = pd.date_range(s["date"].min(), s["date"].max(), freq="D")
        series = (
            s.set_index("date")["count"]
            .reindex(all_dates, fill_value=0)
            .astype(float)
        )
        baseline = series.shift(1).rolling(
            window=rolling_window, min_periods=max(3, rolling_window // 2)
        )
        mean = baseline.mean()
        std = baseline.std().fillna(0.0)
        threshold = mean + k_sigma * std
        alert = (series > threshold) & (series >= min_count_per_zone) & mean.notna()
        out = pd.DataFrame(
            {
                "date": all_dates.strftime("%Y-%m-%d"),
                "zone": zone,
                "categorie_symptomes": cat,
                "count": series.to_numpy(),
                "baseline_mean": mean.to_numpy(),
                "baseline_std": std.to_numpy(),
                "threshold": threshold.to_numpy(),
                "alert": alert.to_numpy(),
            }
        )
        frames.append(out)

    return pd.concat(frames, ignore_index=True) if frames else counts


def evaluate_detection(
    alerts: pd.DataFrame,
    spikes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Mesure pics détectés, délai moyen, fausses alertes.

    Un pic injecté est « détecté » s'il existe au moins une alerte dans
    [spike_start, spike_end + 7 jours] pour la même zone et catégorie.
    """
    detected = 0
    delays: list[int] = []
    spike_windows: list[tuple[str, str, str, str]] = []

    for sp in spikes:
        zone, cat = sp["zone"], sp["category"]
        start, end = sp["spike_start"], sp["spike_end"]
        # Fenêtre élargie de 7 jours pour compter une détection tardive
        end_ext = (pd.Timestamp(end) + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        spike_windows.append((zone, cat, start, end_ext))
        mask = (
            (alerts["zone"] == zone)
            & (alerts["categorie_symptomes"] == cat)
            & (alerts["alert"])
            & (alerts["date"] >= start)
            & (alerts["date"] <= end_ext)
        )
        hits = alerts.loc[mask].sort_values("date")
        if len(hits) > 0:
            detected += 1
            first = hits.iloc[0]["date"]
            delay = (pd.Timestamp(first) - pd.Timestamp(start)).days
            delays.append(int(delay))

    # Fausses alertes : alertes hors de toute fenêtre de pic (zone+cat)
    false_alerts = 0
    alert_rows = alerts[alerts["alert"]]
    for _, row in alert_rows.iterrows():
        in_spike = False
        for zone, cat, start, end_ext in spike_windows:
            if (
                row["zone"] == zone
                and row["categorie_symptomes"] == cat
                and start <= row["date"] <= end_ext
            ):
                in_spike = True
                break
        if not in_spike:
            false_alerts += 1

    return {
        "n_injected_spikes": len(spikes),
        "n_detected": detected,
        "detection_rate": (detected / len(spikes)) if spikes else float("nan"),
        "mean_detection_delay_days": float(np.mean(delays)) if delays else float("nan"),
        "n_false_alerts": false_alerts,
        "data_note": "SIMULÉ — événements et pics injectés artificiellement",
    }
