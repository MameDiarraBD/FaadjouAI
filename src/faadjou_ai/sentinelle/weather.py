"""Météo historique via Open-Meteo (pluie + température).

Attribution : données Open-Meteo (https://open-meteo.com/), licence libre.
API archive : https://archive-api.open-meteo.com

POURQUOI un cache local : sessions Colab qui se coupent + éviter de marteler l'API.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from faadjou_ai.config_utils import ensure_dir, resolve_path

logger = logging.getLogger(__name__)

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
ATTRIBUTION = (
    "Données météo : Open-Meteo Archive API (https://open-meteo.com/), "
    "licence libre. Citation recommandée : Zippenfenig et al., Open-Meteo."
)


def _cache_path(cache_dir: Path, zone: str, start: str, end: str) -> Path:
    safe = zone.replace(" ", "_")
    return cache_dir / f"weather_{safe}_{start}_{end}.json"


def fetch_daily_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    zone: str,
    cache_dir: str | Path,
    timeout: int = 60,
) -> pd.DataFrame:
    """Récupère précipitations et température quotidiennes, avec cache."""
    cache = ensure_dir(resolve_path(cache_dir))
    cpath = _cache_path(cache, zone, start_date, end_date)

    payload: dict[str, Any] | None = None
    if cpath.is_file():
        logger.info("Cache météo hit : %s", cpath)
        with cpath.open(encoding="utf-8") as f:
            payload = json.load(f)
    else:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "precipitation_sum,temperature_2m_mean",
            "timezone": "Africa/Dakar",
        }
        try:
            resp = requests.get(OPEN_METEO_ARCHIVE, params=params, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            with cpath.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except requests.RequestException as exc:
            logger.error(
                "Erreur réseau Open-Meteo pour %s (%s). "
                "Un DataFrame vide sera retourné. Attribution : %s",
                zone,
                exc,
                ATTRIBUTION,
            )
            return pd.DataFrame(
                columns=["date", "zone", "precipitation_mm", "temperature_c", "attribution"]
            )

    daily = payload.get("daily", {}) if payload else {}
    dates = daily.get("time", [])
    precip = daily.get("precipitation_sum", [])
    temp = daily.get("temperature_2m_mean", [])
    rows = []
    for i, d in enumerate(dates):
        rows.append(
            {
                "date": d,
                "zone": zone,
                "precipitation_mm": precip[i] if i < len(precip) else None,
                "temperature_c": temp[i] if i < len(temp) else None,
                "attribution": ATTRIBUTION,
            }
        )
    return pd.DataFrame(rows)


def fetch_all_zones(config: dict[str, Any]) -> pd.DataFrame:
    """Télécharge (ou lit le cache) pour toutes les zones de la config."""
    scfg = config["sentinelle"]
    sim = scfg["simulation"]
    start = sim["start_date"]
    start_d = date.fromisoformat(start)
    end_d = start_d + timedelta(days=int(sim["months"] * 30.44))
    end = end_d.isoformat()
    cache_dir = config["paths"].get("weather_cache_dir", "data/weather_cache")
    coords = scfg["weather"]["latitude_longitude"]

    frames: list[pd.DataFrame] = []
    for zone, latlon in coords.items():
        lat, lon = float(latlon[0]), float(latlon[1])
        frames.append(
            fetch_daily_weather(lat, lon, start, end, zone=zone, cache_dir=cache_dir)
        )
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    logger.info(ATTRIBUTION)
    return df


def heavy_rain_dates(
    weather_df: pd.DataFrame,
    zone: str,
    quantile: float = 0.95,
) -> list[date]:
    """Dates de fortes pluies (quantile élevé) pour une zone."""
    sub = weather_df[weather_df["zone"] == zone].dropna(subset=["precipitation_mm"])
    if sub.empty:
        return []
    thr = float(sub["precipitation_mm"].quantile(quantile))
    heavy = sub[sub["precipitation_mm"] >= thr]
    return [date.fromisoformat(d) for d in heavy["date"].tolist()]
