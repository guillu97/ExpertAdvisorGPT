from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional, Literal

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

    def safe(self) -> "StrategyOutput":
        if self.decision not in ("buy", "sell", "flat"):
            self.decision = "flat"
        if self.sl_points < 0:
            self.sl_points = 0.0
        if self.tp_points < 0:
            self.tp_points = 0.0
        return self


SYSTEM_PROMPT = (
    "Tu es un sélecteur de trades ultra-léger. Réponds STRICTEMENT par un JSON unique.\n"
    "Règles:\n"
    "1) Respecte les DRAPEAUX de tendance fournis:\n"
    "   - 'trend_buy_ok' doit être True pour retourner 'buy'.\n"
    "   - 'trend_sell_ok' doit être True pour retourner 'sell'.\n"
    "   Sinon, réponds 'flat'.\n"
    "2) 'sl_points' / 'tp_points' = DISTANCES en PRIX (même unité que 'close'), pas des points MT5.\n"
    "3) Qualité de marché (gating):\n"
    "   - Seuils de volatilité: utilise 'vol_rel_min' et 'atr_ratio_min' si présents dans les features, sinon par défaut vol_rel_min=1.2 et atr_ratio_min=1.15.\n"
    "     Si 'vol_rel' < vol_rel_min OU 'atr_ratio' < atr_ratio_min → réponds 'flat'.\n"
    "   - Session: privilégie Londres/New York. Si 'session' ∉ {'london_open','newyork'} ET 'is_session_overlap'==0 → réponds 'flat'.\n"
    "   - Direction/impulsion: pour 'buy' exige 'slope_sma20'>0, 'slope_sma100'>=0 et 'dist_sma20'>0; pour 'sell' exige 'slope_sma20'<0, 'slope_sma100'<=0 et 'dist_sma20'<0. Sinon → 'flat'.\n"
    "   - Extrêmes RSI: évite d’acheter si 'rsi14'>72 et de vendre si 'rsi14'<28 (dans ce cas → 'flat').\n"
    "   - Spread: si 'spread_price' >= 0.6*'atr' → 'flat' (coût trop élevé). Sinon maintiens SL ≥ 3*spread_price et ≥ 'stop_level_min'.\n"
    "   - Si 'atr' < 'atr_min_hint' (si fourni) ou si 'cooldown_active' est True → 'flat'.\n"
    "4) Gestion du risque (sizing ATR):\n"
    "   - Base: sl_mult=1.8, tp_rr=2.2 (donc tp_mult≈sl_mult*tp_rr).\n"
    "   - Ajuste par régime: si 'vol_rel'>1.6 ou 'atr_ratio'>1.6 → tp_rr=2.6~3.0; si proche des seuils → tp_rr=1.8~2.2.\n"
    "   - Niveaux minimum: sl_points=max(sl_mult*atr, 3*spread_price, 1.2*stop_level_min). tp_points=max(tp_rr*sl_points, 1.5*atr).\n"
    "5) Prudence adaptative 'loss_pressure' (0→3):\n"
    "   - ≥2.0: réponds 'flat' SAUF si (tendance OK + gating vol OK + gating session OK + conditions de direction §3 OK).\n"
    "           Si tu n’es pas 'flat', élargis légèrement: sl_mult=2.0~2.3 et garde tp_rr>=2.0.\n"
    "   - 1.0–2.0: modérément conservateur (sl_mult ≈ 1.9~2.1; tp_rr ≈ 2.0~2.4).\n"
    "   - <1.0: normal (sl_mult ≈ 1.7~1.9; tp_rr ≈ 2.2~2.8).\n"
    "   - Si 'loss_streak'≥2 OU 'recent_pnl_sum'<0 et 'loss_pressure'≥1.5 → 'flat' sauf si tout est au vert (vol+session+direction).\n"
    "6) Fréquence: si 'trades_in_day' ≥ 6 → réponds 'flat'.\n"
    "7) Cohérence JSON uniquement: N'utilise AUCUNE info hors des features. Pas de texte hors JSON, pas de commentaires.\n"
    'Schéma JSON: {"decision":"buy|sell|flat","sl_points":number>=0,"tp_points":number>=0,"reason":string}\n'
)


def _fallback_decision(features: dict, reason: str) -> StrategyOutput:
    """
    Fallback simple, conscient de loss_pressure:
      - respecte trend_*_ok
      - si loss_pressure >= 2 et contexte pas très fort, renvoie flat
      - sinon SL/TP basés sur ATR, élargis en fonction de loss_pressure
    """
    close = features.get("close")
    sma20 = features.get("sma20")
    atr = float(features.get("atr", 0.0005) or 0.0005)
    atr = max(atr, 0.00025)

    lp = float(features.get("loss_pressure", 0.0) or 0.0)
    lp = max(0.0, min(3.0, lp))
    widen = 1.0 + 0.3 * lp  # élargit SL/TP doucement
    strong_trend_buy = bool(features.get("trend_buy_ok"))
    strong_trend_sell = bool(features.get("trend_sell_ok"))

    try:
        if close is not None and sma20 is not None:
            close = float(close); sma20 = float(sma20)
            # Si très prudents et pas de fort signal → flat
            if lp >= 2.0 and not (strong_trend_buy or strong_trend_sell):
                return StrategyOutput(decision="flat", sl_points=0.0, tp_points=0.0, reason=reason).safe()

            if strong_trend_buy and close > sma20:
                return StrategyOutput(
                    decision="buy",
                    sl_points=1.5 * atr * widen,
                    tp_points=3.0 * atr * widen,
                    reason=reason
                ).safe()
            if strong_trend_sell and close < sma20:
                return StrategyOutput(
                    decision="sell",
                    sl_points=1.5 * atr * widen,
                    tp_points=3.0 * atr * widen,
                    reason=reason
                ).safe()
    except Exception:
        pass
    return StrategyOutput(decision="flat", sl_points=0.0, tp_points=0.0, reason=reason).safe()


def _parse_to_strategy(txt: str) -> StrategyOutput:
    try:
        payload = json.loads(txt)
    except Exception:
        start = txt.find("{"); end = txt.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Pas d'objet JSON trouvé")
        payload = json.loads(txt[start:end+1])

    return StrategyOutput(
        decision=payload.get("decision", "flat"),
        sl_points=float(payload.get("sl_points", 0.0)),
        tp_points=float(payload.get("tp_points", 0.0)),
        reason=str(payload.get("reason", "")),
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
      - loss_pressure (0–3), loss_streak, win_streak, trades_in_day, cooldown_active
      - trend_buy_ok, trend_sell_ok, atr_min_hint
      - vol_rel, vol_rel_min (facultatif), atr_ratio_min (facultatif)
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("gpt_decider")

    # --- config & client ---
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return _fallback_decision(features, "fallback: clé API absente")

    model = model or os.environ.get("OPENAI_MODEL") or "gpt-5-nano"
    base = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    # timeout param: priorité à l'argument, sinon env, sinon 30s
    timeout = (
        timeout
        if timeout is not None
        else float(os.getenv("GPT_DECIDE_TIMEOUT", "30"))
    )
    client = OpenAI(api_key=api_key, base_url=base, timeout=timeout)

    # --- Normalisation / defaults utiles pour le prompt ---
    # indice de min ATR
    if "atr_min_hint" not in features and "atr" in features:
        try:
            features["atr_min_hint"] = max(0.00025, 0.5 * float(features["atr"]))
        except Exception:
            features["atr_min_hint"] = 0.00025

    # seuils de volatilité si absents
    features.setdefault("vol_rel_min", 1.2)
    features.setdefault("atr_ratio_min", 1.15)

    # messages
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
            # Essai 1: réponse JSON forcée (si modèle le supporte)
            try:
                cmp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
            except Exception as e_rf:
                # Si le modèle ne supporte pas response_format, on retente sans
                logger.debug(f"response_format JSON non supporté, retry sans: {e_rf}")
                cmp = client.chat.completions.create(
                    model=model,
                    messages=messages
                )

            txt = cmp.choices[0].message.content or ""
            out = _parse_to_strategy(txt)
            logger.info(f"[Chat] OK en {time.time()-start:.2f}s (attempt {attempt})")
            return out
        except Exception as e:
            last_error = e
            wait = min(8, 1.5 ** (attempt - 1))  # petit backoff exponentiel
            logger.warning(f"Tentative {attempt}/{max_retries} échec: {e} → sleep {wait:.2f}s")
            time.sleep(wait)

    return _fallback_decision(features, f"fallback: {type(last_error).__name__}")


if __name__ == "__main__":
    features = {
        "close": 1.1050, "sma20": 1.1045, "sma100": 1.1030, "atr": 0.00045,
        "trend_buy_ok": True, "loss_pressure": 2.2, "vol_rel": 1.5, "atr_ratio": 1.2,
        "session": "london_open", "is_session_overlap": 1, "slope_sma20": 0.0001,
        "slope_sma100": 0.0, "dist_sma20": 0.0003, "spread_price": 0.0001,
        "stop_level_min": 0.0, "cooldown_active": False, "trades_in_day": 0, "rsi14": 55,
    }
    res = gpt_decide(features)
    print(json.dumps(res.model_dump(), indent=2, ensure_ascii=False))
