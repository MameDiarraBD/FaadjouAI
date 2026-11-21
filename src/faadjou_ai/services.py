"""Interfaces remplaçables pour les services externes de Faadjou.

Ces classes isolent la reconnaissance vocale, les réservations et les
événements pour permettre un branchement ultérieur aux vraies API.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import date, time
from typing import Any


class SpeechRecognitionService(abc.ABC):
    @abc.abstractmethod
    def recognize(self, audio_bytes: bytes | None = None) -> str:
        """Retourne une transcription. Implémentation réelle à brancher plus tard."""
        raise NotImplementedError


class DemoSpeechRecognitionService(SpeechRecognitionService):
    """Démo locale sans accès à un modèle ASR réel."""

    def recognize(self, audio_bytes: bytes | None = None) -> str:
        if audio_bytes:
            return "TODO_WOLOF"
        return ""


@dataclass(frozen=True)
class Practitioner:
    name: str
    specialty: str
    location: str
    next_slot: str
    fictitious: bool = True


class ReservationService(abc.ABC):
    @abc.abstractmethod
    def list_practitioners(self, specialty: str) -> list[Practitioner]:
        raise NotImplementedError

    @abc.abstractmethod
    def create_reservation(
        self,
        practitioner: Practitioner,
        appointment_date: date,
        appointment_time: time,
    ) -> dict[str, Any]:
        raise NotImplementedError


class DemoReservationService(ReservationService):
    """Réservation fictive : aucune donnée réelle et aucun paiement."""

    PRACTITIONERS = (
        Practitioner("Dr. A. Fictif", "medecin_generaliste", "Dakar — Démo", "Aujourd'hui"),
        Practitioner("Dr. B. Fictif", "pediatrie", "Thiès — Démo", "Demain"),
        Practitioner("Dr. C. Fictif", "urgences", "Saint-Louis — Démo", "Disponible"),
    )

    def list_practitioners(self, specialty: str) -> list[Practitioner]:
        matches = [p for p in self.PRACTITIONERS if p.specialty == specialty]
        return matches or list(self.PRACTITIONERS[:2])

    def create_reservation(
        self,
        practitioner: Practitioner,
        appointment_date: date,
        appointment_time: time,
    ) -> dict[str, Any]:
        return {
            "status": "SIMULÉ",
            "practitioner": practitioner.name,
            "date": appointment_date.isoformat(),
            "time": appointment_time.strftime("%H:%M"),
            "payment": "Aucun paiement — démonstration",
            "real": False,
        }


class EventService(abc.ABC):
    @abc.abstractmethod
    def create(self, event: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class DemoEventService(EventService):
    """Service local de stockage en mémoire pour le prototype."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def create(self, event: dict[str, Any]) -> dict[str, Any]:
        stored = dict(event)
        stored["status"] = "SIMULÉ"
        self.events.append(stored)
        return stored
