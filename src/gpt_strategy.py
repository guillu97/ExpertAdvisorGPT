# src/gpt_strategy.py
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional, Literal, Any, Dict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import OpenAI

load_dotenv()

Decision = Literal["buy", "sell", "flat"]


class StrategyOutput(BaseModel):
    decision: Decision = Field(default="flat")
    # IMPORTANT: sl_points / tp_points = DISTANCE EN PRIX (même unité que close),
    # pas en "points" MT5. Exemple EURUSD: 0.00050 = 5 pips.
    sl_points: float = Field(default=0.0, ge=0.0)
    tp_points: float = Field(default=0.0, ge=0.0)
    reason: str = Field(default="")
    # Confiance globale (0..1) sur l’edge de la décision.
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    # Facultatif: mini diagnostic pour les logs/analyses
    playbook: Optional[Dict[str, Any]] = None

    def safe(self) -> "StrategyOutput":
        # décision valide
        if self.decision not in ("buy", "sell", "flat"):
            self.decision = "flat"
        # distances >= 0
        try:
            self.sl_points = float(self.sl_points)
            self.tp_points = float(self.tp_points)
        except Exception:
            self.sl_points, self.tp_points = 0.0, 0.0
        if self.sl_points < 0:
            self.sl_points = 0.0
        if self.tp_points < 0:
            self.tp_points = 0.0
        # clamp confiance
        try:
            c = float(self.confidence)
        except Exception:
            c = 0.5
        self.confidence = max(0.0, min(1.0, c))
        # playbook = dict simple ou None
        if self.playbook is not None and not isinstance(self.playbook, dict):
            self.playbook = None
        return self


SYSTEM_PROMPT = (
    "Tu es un sélecteur de trades ultra-léger. Réponds STRICTEMENT par un JSON unique.\n"
    "Objectif: décider {buy|sell|flat} et calibrer des distances SL/TP en PRIX via l’ATR, de façon robuste.\n"
    "\n"
    "RÈGLES DURES (→ 'flat' immédiat):\n"
    " - 'cooldown_active' True ou 'kill_switch_active' True.\n"
    " - 'atr' absent/non valide OU 'spread_price' >= 0.6*'atr'.\n"
    " - 'vol_rel' ou 'atr_ratio' absents/non valides.\n"
    " - Si 'event_window_active'==True (no-trade news) → flat.\n"
    " - Si fourni: 'session_allowed'==False.\n"
    "\n"
    "SEUILS DYNAMIQUES:\n"
    " - Utilise en priorité 'vol_rel_min_dyn' / 'atr_ratio_min_dyn' s'ils existent, sinon 'vol_rel_min'/'atr_ratio_min'.\n"
    " - Si tout est absent, par défaut vol_rel_min=1.10 et atr_ratio_min=1.00.\n"
    "\n"
    "PRÉFÉRENCES SOUPLES (scoring, pas de if/else rigides):\n"
    " * BUY: indices positifs = 'trend_buy_ok', 'supertrend_dir'=='up', 'adx14' ≥ 'adx_min', 'slope_sma20'>0, 'slope_sma100'>=0,\n"
    "        'dist_sma20'>0, 'session'∈{london_open,newyork}, bonus si 'is_session_overlap'==1, 'htf_bias'>=0.\n"
    " * SELL: indices positifs = 'trend_sell_ok', 'supertrend_dir'=='down', 'adx14' ≥ 'adx_min', 'slope_sma20'<0, 'slope_sma100'<=0,\n"
    "         'dist_sma20'<0, mêmes bonus de session, 'htf_bias'<=0.\n"
    " * Pénalités: RSI extrême (évite BUY si 'rsi14'>72 ; évite SELL si 'rsi14'<28), 'atr_slope' contre la position,\n"
    "   'spread_penalty' élevé, ou vol/ATR juste au seuil.\n"
    " * Pondère par 'session_weight' si présent (0..~1.1) et utilise 'sl_mult_hint_range'/'tp_rr_hint_range' si fournis.\n"
    "\n"
    "PRUDENCE ADAPTATIVE (loss_pressure 0..3):\n"
    " - <1.0 → seuils faciles (un léger edge suffit).\n"
    " - 1.0–2.0 → seuils modérés (demander un edge net).\n"
    " - ≥2.0 → très conservateur: exige edge fort ET alignement tendance+session, sinon 'flat'.\n"
    "Calcule mentalement score_buy et score_sell à partir des indices ci-dessus; décide selon le meilleur score et un seuil de séparation\n"
    "adaptatif (plus 'loss_pressure' est haut, plus le seuil est grand). Si scores proches → 'flat'.\n"
    "\n"
    "CALIBRAGE SL/TP (distances en PRIX):\n"
    " - sl_mult dans ~[1.7, 2.3] (plus haut si loss_pressure élevé et/ou volatilité nerveuse). Base 1.8.\n"
    " - tp_rr dans ~[2.0, 2.8] (peut monter vers 3.0 si 'vol_rel'>1.6 ou 'atr_ratio'>1.6; proche des minima → 1.8–2.2).\n"
    " - sl_points = max(sl_mult*atr, 3*spread_price, 1.2*stop_level_min).\n"
    " - tp_points = max(tp_rr*sl_points, 1.5*atr).\n"
    " - Les distances doivent rester réalistes (pas > ~5*ATR pour SL, ni RR < 1.6).\n"
    "\n"
    "SORTIE JSON OBLIGATOIRE (pas de texte hors JSON).\n"
    "Schéma minimal: {\"decision\":\"buy|sell|flat\",\"sl_points\":number>=0,\"tp_points\":number>=0,\n"
    "                 \"reason\":string,\"confidence\":0..1}\n"
    "Tu peux OPTIONNELLEMENT ajouter 'playbook' (objet court) avec: {\"score_buy\":.., \"score_sell\":..,\n"
    " \"sl_mult\":.., \"tp_rr\":.., \"vol_min_used\":.., \"atr_min_used\":..}.\n"
)


def _bound(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _get_float(d: dict, k: str, default: float = 0.0) -> float:
    try:
        v = float(d.get(k, default))
        if not (v == v):  # NaN check
            return default
        return v
    except Exception:
        return default


def _fallback_decision(features: dict, reason: str) -> StrategyOutput:
    """
    Fallback scoring + SL/TP robustes quand l'API n'est pas dispo.
    - Scoring souple (BUY vs SELL) avec seuil adaptatif selon loss_pressure
    - SL/TP bornés et réalistes
    - playbook inclus
    """
    # --- lecture des features avec garde-fous ---
    close = _get_float(features, "close", 0.0)
    sma20 = _get_float(features, "sma20", close)
    sma100 = _get_float(features, "sma100", sma20)
    atr = _get_float(features, "atr", 0.0005)
    atr = max(atr, 0.00025)
    spread_price = _get_float(features, "spread_price", 0.0)
    stop_level_min = _get_float(features, "stop_level_min", 0.0)
    rsi = _get_float(features, "rsi14", 50.0)
    vol_rel = _get_float(features, "vol_rel", 1.0)
    atr_ratio = _get_float(features, "atr_ratio", 1.0)
    atr_slope = _get_float(features, "atr_slope", 0.0)

    lp = _get_float(features, "loss_pressure", 0.0)
    lp = _bound(lp, 0.0, 3.0)

    # News no-trade window
    evt_active = bool(features.get("event_window_active", False))
    if evt_active:
        return StrategyOutput(decision="flat", reason="fallback: event_window_active", confidence=0.25).safe()

    # Hard stop spread vs ATR
    if spread_price >= 0.6 * atr:
        return StrategyOutput(decision="flat", reason="fallback: spread>0.6*ATR", confidence=0.2).safe()

    trend_buy_ok = bool(features.get("trend_buy_ok"))
    trend_sell_ok = bool(features.get("trend_sell_ok"))
    slope20 = _get_float(features, "slope_sma20", 0.0)
    slope100 = _get_float(features, "slope_sma100", 0.0)
    dist20 = _get_float(features, "dist_sma20", 0.0)
    session = str(features.get("session", "tokyo") or "tokyo")
    overlap = int(features.get("is_session_overlap", 0) or 0)
    session_weight = _get_float(features, "session_weight", 0.6)
    htf_bias = _get_float(features, "htf_bias", 0.0)
    spread_penalty = _get_float(features, "spread_penalty", spread_price / max(atr, 1e-12))

    # Seuils dynamiques/hints
    vol_min = _get_float(features, "vol_rel_min", 1.10)
    atr_min = _get_float(features, "atr_ratio_min", 1.00)
    vol_min = _get_float(features, "vol_rel_min_dyn", vol_min)
    atr_min = _get_float(features, "atr_ratio_min_dyn", atr_min)

    # Scoring souple
    score_buy = 0.0
    score_sell = 0.0

    # Tendance + session
    if trend_buy_ok:
        score_buy += 1.0
    if trend_sell_ok:
        score_sell += 1.0
    if session in ("london_open", "newyork"):
        score_buy += 0.3 * session_weight
        score_sell += 0.3 * session_weight
    if overlap:
        score_buy += 0.1
        score_sell += 0.1

    # Slopes/distances
    if slope20 > 0:
        score_buy += 0.4
    if slope20 < 0:
        score_sell += 0.4
    if slope100 >= 0:
        score_buy += 0.2
    if slope100 <= 0:
        score_sell += 0.2
    if dist20 > 0:
        score_buy += 0.25
    if dist20 < 0:
        score_sell += 0.25

    # Volatilité par rapport aux seuils
    if vol_rel >= vol_min:
        score_buy += 0.25
        score_sell += 0.25
    if atr_ratio >= atr_min:
        score_buy += 0.25
        score_sell += 0.25

    # RSI extrêmes
    if rsi > 72:
        score_buy -= 0.4
    if rsi < 28:
        score_sell -= 0.4

    # ATR slope contre la position
    if atr_slope < 0:
        score_buy -= 0.1
    if atr_slope > 0:
        score_sell -= 0.1

    # HTF bias
    if htf_bias > 0:
        score_buy += 0.2
    if htf_bias < 0:
        score_sell += 0.2

    # Coût microstructure
    score_buy -= 0.15 * spread_penalty
    score_sell -= 0.15 * spread_penalty

    # Seuil de séparation adaptatif
    sep = 0.05 + 0.15 * _bound(lp, 0.0, 2.0) + (
        0.05 if session not in ("london_open", "newyork") and not overlap else 0.0
    )

    # Décision brute + confidence
    edge = score_buy - score_sell
    decision: Decision = "flat"
    if edge > sep:
        decision = "buy"
    elif edge < -sep:
        decision = "sell"
    else:
        decision = "flat"

    # Confiance: normalisation douce
    conf = _bound(abs(edge) / (1.8 + 0.6 * lp), 0.05, 0.95)
    conf *= (0.95 - 0.15 * (lp / 3.0))  # prudence avec lp
    if decision == "flat":
        conf = max(0.05, 0.35 - 0.1 * lp)

    # SL/TP: hints + bornes
    lo_sl, hi_sl = 1.7, 2.3
    if isinstance(features.get("sl_mult_hint_range"), (list, tuple)) and len(features["sl_mult_hint_range"]) == 2:
        try:
            lo_sl = float(features["sl_mult_hint_range"][0])
            hi_sl = float(features["sl_mult_hint_range"][1])
        except Exception:
            pass
    lo_rr, hi_rr = 2.0, 2.8
    if isinstance(features.get("tp_rr_hint_range"), (list, tuple)) and len(features["tp_rr_hint_range"]) == 2:
        try:
            lo_rr = float(features["tp_rr_hint_range"][0])
            hi_rr = float(features["tp_rr_hint_range"][1])
        except Exception:
            pass

    # sl_mult augmente avec lp et avec vol/atr_ratio élevés
    vol_boost = 0.15 if (vol_rel > 1.5 or atr_ratio > 1.5) else 0.0
    sl_mult = _bound(1.8 + 0.15 * min(lp, 2.0) + vol_boost, lo_sl, hi_sl)

    # tp_rr augmente avec vol/atr_ratio, baisse si proche des minima
    rr_base = 2.2 + (0.4 if (vol_rel > 1.6 or atr_ratio > 1.6) else 0.0) - (
        0.2 if (vol_rel < vol_min + 0.05 or atr_ratio < atr_min + 0.05) else 0.0
    )
    tp_rr = _bound(rr_base, lo_rr, max(hi_rr, 3.0))

    # Distances
    sl_points = max(sl_mult * atr, 3.0 * max(spread_price, 0.0), 1.2 * max(stop_level_min, 0.0))
    sl_points = min(sl_points, 5.0 * atr)  # garde-fou SL
    tp_points = max(tp_rr * sl_points, 1.5 * atr)

    if decision == "flat":
        sl_points = 0.0
        tp_points = 0.0

    playbook = {
        "score_buy": round(score_buy, 3),
        "score_sell": round(score_sell, 3),
        "edge": round(edge, 3),
        "sep": round(sep, 3),
        "sl_mult": round(sl_mult, 3),
        "tp_rr": round(tp_rr, 3),
        "vol_min_used": round(vol_min, 3),
        "atr_min_used": round(atr_min, 3),
        "session": session,
        "overlap": int(overlap),
    }

    return StrategyOutput(
        decision=decision,
        sl_points=sl_points,
        tp_points=tp_points,
        reason=reason,
        confidence=conf,
        playbook=playbook,
    ).safe()


def _parse_to_strategy(txt: str) -> StrategyOutput:
    # supporte éventuels code fences
    try:
        payload = json.loads(txt)
    except Exception:
        start = txt.find("{")
        end = txt.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Pas d'objet JSON trouvé")
        payload = json.loads(txt[start:end + 1])

    def _getnum(key: str, default: float = 0.0) -> float:
        try:
            v = float(payload.get(key, default))
            if not (v == v):
                return default
            return v
        except Exception:
            return default

    conf = _getnum("confidence", 0.5)
    slp = _getnum("sl_points", 0.0)
    tpp = _getnum("tp_points", 0.0)
    reason = str(payload.get("reason", ""))

    # playbook facultatif
    pb = payload.get("playbook")
    if not isinstance(pb, dict):
        pb = None

    return StrategyOutput(
        decision=str(payload.get("decision", "flat")),
        sl_points=slp,
        tp_points=tpp,
        reason=reason,
        confidence=conf,
        playbook=pb,
    ).safe()


def gpt_decide(
    features: dict,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = None,
    max_retries: int = 4,
) -> StrategyOutput:
    """
    features attendues (exemples non exhaustifs):
      - close, sma20, sma100, sma200, atr, atr_ratio, atr_slope
      - dist_sma20, dist_sma100, dist_sma200, slope_sma20, slope_sma100, slope_sma200
      - hour, session (tokyo/london_open/newyork), is_session_overlap
      - spread_price, stop_level_min, freeze_level_min
      - loss_pressure (0–3), loss_streak, win_streak, trades_in_day, cooldown_active, kill_switch_active
      - trend_buy_ok, trend_sell_ok, atr_min_hint
      - vol_rel, vol_rel_min (facultatif), atr_ratio_min (facultatif)
      - adx14, adx_min (facultatif), supertrend_dir ('up'/'down'/'flat') (facultatif)
      - max_trades_per_day (facultatif, défaut 6)
      - Hints facultatifs: sl_mult_hint_range [lo,hi], tp_rr_hint_range [lo,hi],
        session_weight, htf_bias, spread_penalty, vol_rel_min_dyn, atr_ratio_min_dyn
      - News: event_window_active, minutes_to_next_high, minutes_since_last_high, no_trade_before_high_min, no_trade_after_high_min
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("gpt_decider")

    # --- config & client ---
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return _fallback_decision(features, "fallback: clé API absente")

    model = model or os.environ.get("OPENAI_MODEL") or "gpt-5-nano"
    base = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    timeout = timeout if timeout is not None else float(os.getenv("GPT_DECIDE_TIMEOUT", "30"))
    client = OpenAI(api_key=api_key, base_url=base, timeout=timeout)

    # --- Normalisation / defaults utiles pour le prompt ---
    if "atr_min_hint" not in features and "atr" in features:
        try:
            features["atr_min_hint"] = max(0.00025, 0.5 * float(features["atr"]))
        except Exception:
            features["atr_min_hint"] = 0.00025

    features.setdefault("vol_rel_min", 1.10)
    features.setdefault("atr_ratio_min", 1.00)
    features.setdefault("max_trades_per_day", 6)
    features.setdefault("adx_min", 20)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Décide parmi {buy,sell,flat}. Utilise EXCLUSIVEMENT ces features JSON et respecte les règles:\n"
                f"{json.dumps(features, ensure_ascii=False)}"
            ),
        },
    ]

    # --- Appels avec retries & backoff, forçage JSON si possible ---
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            start = time.time()
            try:
                cmp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
            except Exception as e_rf:
                logger.debug(f"response_format JSON non supporté, retry sans: {e_rf}")
                cmp = client.chat.completions.create(model=model, messages=messages)

            txt = cmp.choices[0].message.content or ""
            out = _parse_to_strategy(txt)
            logger.info(f"[Chat] OK en {time.time()-start:.2f}s (attempt {attempt})")
            return out
        except Exception as e:
            last_error = e
            wait = min(8, 1.5 ** (attempt - 1))  # petit backoff
            logger.warning(f"Tentative {attempt}/{max_retries} échec: {e} → sleep {wait:.2f}s")
            time.sleep(wait)

    return _fallback_decision(features, f"fallback: {type(last_error).__name__}")


if __name__ == "__main__":
    # Petit test manuel
    features = {
        "close": 1.1050, "sma20": 1.1045, "sma100": 1.1030, "sma200": 1.1020,
        "atr": 0.00045, "atr_ratio": 1.2, "vol_rel": 1.4, "atr_slope": 0.0,
        "trend_buy_ok": True, "trend_sell_ok": False,
        "loss_pressure": 0.8, "loss_streak": 0, "recent_pnl_sum": 0.0, "trades_in_day": 0,
        "session": "london_open", "is_session_overlap": 1, "rsi14": 55,
        "slope_sma20": 0.0001, "slope_sma100": 0.0, "dist_sma20": 0.0003,
        "spread_price": 0.0001, "stop_level_min": 0.0, "cooldown_active": False,
        "supertrend_dir": "up", "adx14": 23.0, "adx_min": 20,
        "kill_switch_active": False,
        # Hints & extras
        "session_weight": 1.05, "htf_bias": 0.5, "spread_penalty": 0.1,
        "sl_mult_hint_range": [1.8, 2.2], "tp_rr_hint_range": [2.0, 2.8],
        "vol_rel_min": 1.10, "atr_ratio_min": 1.00,
        # News
        "event_window_active": False
    }
    res = gpt_decide(features)
    print(json.dumps(res.model_dump(), indent=2, ensure_ascii=False))