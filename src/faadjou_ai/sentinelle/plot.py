"""Graphiques Sentinelle — toujours annotés SIMULÉ."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from faadjou_ai.config_utils import ensure_dir, resolve_path


def plot_spike_with_rain(
    counts: pd.DataFrame,
    weather: pd.DataFrame,
    spike: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Trace un pic SIMULÉ : comptages symptômes + courbe de pluie."""
    zone = spike["zone"]
    cat = spike["category"]
    out = resolve_path(output_path)
    ensure_dir(out.parent)

    c = counts[
        (counts["zone"] == zone) & (counts["categorie_symptomes"] == cat)
    ].copy()
    c["date"] = pd.to_datetime(c["date"])
    if weather is not None and not weather.empty and "zone" in weather.columns:
        w = weather[weather["zone"] == zone].copy()
        if not w.empty:
            w["date"] = pd.to_datetime(w["date"])
        else:
            w = pd.DataFrame()
    else:
        w = pd.DataFrame()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(c["date"], c["count"], width=1.0, alpha=0.7, label="Comptage symptômes (SIMULÉ)")
    ax1.axvspan(
        pd.Timestamp(spike["spike_start"]),
        pd.Timestamp(spike["spike_end"]),
        color="red",
        alpha=0.2,
        label="Fenêtre pic injecté (SIMULÉ)",
    )
    if "alert" in c.columns:
        alerts = c[c["alert"]]
        if not alerts.empty:
            ax1.scatter(
                alerts["date"],
                alerts["count"],
                color="darkred",
                zorder=5,
                label="Alerte détectée",
            )
    ax1.set_ylabel("Comptage quotidien (SIMULÉ)")
    ax1.set_xlabel("Date")

    ax2 = ax1.twinx()
    if not w.empty:
        ax2.plot(
            w["date"],
            w["precipitation_mm"],
            color="steelblue",
            linewidth=1.5,
            label="Précipitations (Open-Meteo)",
        )
        ax2.set_ylabel("Précipitations (mm)")

    ax1.set_title(
        f"Sentinelle — pic SIMULÉ ({zone} / {cat})\n"
        f"Données d'événements SIMULÉES — pluie : Open-Meteo"
    )

    # Légendes combinées
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
