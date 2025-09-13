# src/econ_calendar.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime, timezone
import pandas as pd


@dataclass
class EconEvent:
    when: datetime          # UTC
    impact: str             # "low" | "medium" | "high"
    name: str


def _to_utc(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def load_events_csv(path: str) -> List[EconEvent]:
    """
    CSV attendu: colonnes = when, impact, name
      - when: ISO8601 (timezone optionnelle)
      - impact: low|medium|high (case-insensitive)
      - name: texte
    """
    df = pd.read_csv(path)
    df["when"] = pd.to_datetime(df["when"], utc=True, errors="coerce")
    df = df.dropna(subset=["when"])
    df["impact"] = df["impact"].astype(str).str.lower().str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    evts = [
        EconEvent(when=row["when"].to_pydatetime(), impact=row["impact"], name=row["name"])
        for _, row in df.iterrows()
    ]
    evts.sort(key=lambda e: e.when)
    return evts


def build_high_impact_index(events: List[EconEvent]) -> List[EconEvent]:
    return [e for e in events if e.impact == "high"]


def minutes_to_event(now_utc: datetime, evt_time_utc: datetime) -> int:
    return int((evt_time_utc - now_utc).total_seconds() // 60)


def minutes_since_event(now_utc: datetime, evt_time_utc: datetime) -> int:
    return int((now_utc - evt_time_utc).total_seconds() // 60)


def nearest_high_events(
    now_utc: datetime, highs: List[EconEvent], ptr: int
) -> Tuple[Optional[int], Optional[int], Optional[EconEvent], int]:
    """
    Renvoie (mins_to_next_high, mins_since_last_high, next_evt, new_ptr).
    ptr est un index "curseur" pour éviter les recherches coûteuses à chaque barre.
    """
    n = len(highs)
    if n == 0:
        return None, None, None, ptr

    # Avance le curseur tant que l'événement est passé
    while ptr < n and highs[ptr].when <= now_utc:
        ptr += 1

    # Prochain à venir
    mins_to_next = None
    next_evt = None
    if ptr < n:
        next_evt = highs[ptr]
        mins_to_next = minutes_to_event(now_utc, next_evt.when)

    # Dernier passé
    prev_idx = max(0, ptr - 1)
    mins_since_last = None
    if highs[prev_idx].when <= now_utc:
        mins_since_last = minutes_since_event(now_utc, highs[prev_idx].when)

    return mins_to_next, mins_since_last, next_evt, ptr