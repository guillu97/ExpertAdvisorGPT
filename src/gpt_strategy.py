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
    "1) Respecte les DRAPEAUX de tendance fournis par l'utilisateur:\n"
    "   - 'trend_buy_ok' doit être True pour retourner 'buy'.\n"
    "   - 'trend_sell_ok' doit être True pour retourner 'sell'.\n"
    "   Sinon, réponds 'flat'.\n"
    "2) 'sl_points' / 'tp_points' = distances en PRIX (même unité que 'close').\n"
    "3) Donne des niveaux cohérents avec l'ATR et le régime de volatilité. Évite les stops trop serrés.\n"
    "4) Si 'atr' < 'atr_min_hint' (si fourni) ou si 'cooldown_active' est True, privilégie 'flat'.\n"
    "5) Vérifie la volatilité:\n"
    "   - Si 'vol_rel' < 1.1 ou 'atr_ratio' < 1.0 → réponds 'flat' (marché trop calme).\n"
    "6) Utilise 'loss_pressure' (0 à 3) pour ajuster l’agressivité:\n"
    "   - >=2: privilégie 'flat' sauf si tous les drapeaux sont favorables; SL plus large, TP plus ambitieux.\n"
    "   - 1–2: modérément conservateur (SL/TP légèrement plus grands que la normale).\n"
    "   - <1: comportement normal.\n"
    "7) 'loss_streak' et 'win_streak' peuvent préciser le contexte, mais 'loss_pressure' prime.\n"
    "8) N'utilise aucune info hors des features. Pas de texte hors JSON, pas de commentaires.\n"
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
    timeout: float = 15.0,
    max_retries: int = 3,
) -> StrategyOutput:
    """
    features attendues (exemples non exhaustifs):
      - close, sma20, sma100, sma200, atr, atr_ratio, atr_slope, vol_mean_14
      - dist_sma20, dist_sma100, dist_sma200, slope_sma20, slope_sma100, slope_sma200
      - m15_sma20, h1_sma20, h4_sma20 (ou flags m15_trend, h1_trend…)
      - hour, session (tokyo/london/ny), is_session_overlap
      - spread_price, stop_level_min, freeze_level_min
      - loss_pressure (0–3), loss_streak, win_streak, trades_in_day, cooldown_active
      - trend_buy_ok, trend_sell_ok, atr_min_hint
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("gpt_decider")

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return _fallback_decision(features, "fallback: clé API absente")

    model = model or os.environ.get("OPENAI_MODEL") or "gpt-5-nano"
    base = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    client = OpenAI(api_key=api_key, base_url=base, timeout=timeout)

    # Petite normalisation & hints utiles pour le prompting
    if "atr_min_hint" not in features and "atr" in features:
        features["atr_min_hint"] = max(0.00025, 0.5 * float(features["atr"]))

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

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            start = time.time()
            cmp = client.chat.completions.create(model=model, messages=messages)
            txt = cmp.choices[0].message.content or ""
            out = _parse_to_strategy(txt)
            logger.info(f"[Chat] OK en {time.time()-start:.2f}s (attempt {attempt})")
            return out
        except Exception as e:
            last_error = e
            logger.warning(f"Tentative {attempt}/{max_retries} échec: {e}")
            if attempt < max_retries:
                time.sleep(2)
            else:
                break

    return _fallback_decision(features, f"fallback: {type(last_error).__name__}")


if __name__ == "__main__":
    features = {
        "close": 1.1050, "sma20": 1.1045, "sma100": 1.1030, "atr": 0.00045,
        "trend_buy_ok": True, "loss_pressure": 2.2
    }
    res = gpt_decide(features)
    print(json.dumps(res.model_dump(), indent=2, ensure_ascii=False))
