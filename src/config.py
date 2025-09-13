# src/config.py
from __future__ import annotations

import os
from typing import Optional

# --- Pydantic v2 → field_validator ; fallback v1 → validator(pre=True) ---
try:  # v2
    from pydantic import BaseModel, Field, field_validator
    _PydVer = 2
except Exception:  # v1
    from pydantic import BaseModel, Field, validator as _validator  # type: ignore
    _PydVer = 1

    def field_validator(*fields, mode: str = "before", **kwargs):
        """
        Minimal shim to emulate pydantic v2 `field_validator` with v1's `validator(pre=True)`.
        Usage stays the same in the rest of the file.
        """
        pre = (mode == "before")

        def _decorator(fn):
            return _validator(*fields, pre=pre, **kwargs)(fn)  # type: ignore
        return _decorator


# ----------------------------- helpers -----------------------------
def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name, default)
    if v is None:
        return v
    # retire guillemets et espaces parasites
    v = v.strip().strip('"').strip("'")
    return v

def _env_int(name: str, default: int) -> int:
    v = _env_str(name, None)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    v = _env_str(name, None)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default

def _env_bool(name: str, default: bool) -> bool:
    v = _env_str(name, None)
    if v is None:
        return default
    v = v.lower()
    return v in ("1", "true", "yes", "on")

# ------------------------- Settings model --------------------------
class Settings(BaseModel):
    # OpenAI
    openai_api_key: str = Field(default_factory=lambda: _env_str("OPENAI_API_KEY", "") or "")
    openai_base_url: str = Field(default_factory=lambda: _env_str("OPENAI_BASE_URL", "https://api.openai.com/v1") or "https://api.openai.com/v1")
    openai_model: str = Field(default_factory=lambda: _env_str("OPENAI_MODEL", "gpt-5-nano") or "gpt-5-nano")

    # MT5 credentials
    mt5_login: Optional[int] = Field(default_factory=lambda: _env_int("MT5_LOGIN", 0) or None)
    mt5_password: str = Field(default_factory=lambda: _env_str("MT5_PASSWORD", "") or "")
    mt5_server: str = Field(default_factory=lambda: _env_str("MT5_SERVER", "") or "")
    mt5_path: Optional[str] = Field(default_factory=lambda: _env_str("MT5_PATH", None))

    # Optional Linux bridge
    mt5_host: str = Field(default_factory=lambda: _env_str("MT5_HOST", "localhost") or "localhost")
    mt5_port: int = Field(default_factory=lambda: _env_int("MT5_PORT", 8001))

    # Strategy / risk knobs (valeurs par défaut raisonnables)
    risk_pct_per_trade: float = Field(default_factory=lambda: _env_float("ACCOUNT_RISK_PER_TRADE", 0.004))  # 0.4%/trade
    max_daily_dd_pct: float = Field(default_factory=lambda: _env_float("MAX_DAILY_DD_PCT", 0.02))          # -2% jour
    commission_per_lot: float = Field(default_factory=lambda: _env_float("COMMISSION_PER_LOT", 7.0))       # $/lot RT
    adx_min: float = Field(default_factory=lambda: _env_float("ADX_MIN", 20.0))
    supertrend_period: int = Field(default_factory=lambda: _env_int("SUPERTREND_PERIOD", 10))
    supertrend_mult: float = Field(default_factory=lambda: _env_float("SUPERTREND_MULT", 3.0))

    # Convenience / backtest hints
    timeframe: str = Field(default_factory=lambda: _env_str("BT_TIMEFRAME", "M5") or "M5")
    lookback_days: int = Field(default_factory=lambda: _env_int("LOOKBACK_DAYS", 7))
    top_symbols: int = Field(default_factory=lambda: _env_int("TOP_SYMBOLS", 8))

    @field_validator("timeframe", mode="before")
    def _norm_tf(cls, v: str) -> str:
        if v is None:
            return "M5"
        v = str(v).strip().strip('"').strip("'").upper()
        # autoriser M1/M5/M15/M30/H1/H4/D1/W1/MN1
        ok = {"M1","M2","M3","M4","M5","M6","M10","M12","M15","M20","M30",
              "H1","H2","H3","H4","H6","H8","H12",
              "D1","W1","MN1"}
        if v not in ok:
            raise ValueError(f"timeframe invalide: {v}")
        return v


def load_settings() -> Settings:
    """Charge les settings depuis l'environnement, avec normalisation."""
    return Settings()


# ---------------------- timeframe_to_mt5 ---------------------------
def timeframe_to_mt5(tf: str):
    """
    Retourne un identifiant de timeframe compatible avec notre client MT5.
    - Si MetaTrader5 est dispo → renvoie la constante MT5.
    - Sinon → renvoie simplement la chaîne upper (le client sait mapper les strings).
    """
    tf_u = (tf or "").strip().strip('"').strip("'").upper()
    # mapping pour string fallback
    allowed = {
        "M1","M2","M3","M4","M5","M6","M10","M12","M15","M20","M30",
        "H1","H2","H3","H4","H6","H8","H12","D1","W1","MN1"
    }
    if tf_u not in allowed:
        raise ValueError(f"Timeframe inconnu: {tf}")

    # Essaye d'utiliser les constantes MetaTrader5 si possible
    try:
        import MetaTrader5 as mt5
        mapping = {
            "M1": mt5.TIMEFRAME_M1, "M2": mt5.TIMEFRAME_M2, "M3": mt5.TIMEFRAME_M3, "M4": mt5.TIMEFRAME_M4,
            "M5": mt5.TIMEFRAME_M5, "M6": mt5.TIMEFRAME_M6, "M10": mt5.TIMEFRAME_M10, "M12": mt5.TIMEFRAME_M12,
            "M15": mt5.TIMEFRAME_M15, "M20": mt5.TIMEFRAME_M20, "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1, "H2": mt5.TIMEFRAME_H2, "H3": mt5.TIMEFRAME_H3, "H4": mt5.TIMEFRAME_H4,
            "H6": mt5.TIMEFRAME_H6, "H8": mt5.TIMEFRAME_H8, "H12": mt5.TIMEFRAME_H12,
            "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1,
        }
        return mapping[tf_u]
    except Exception:
        # Pas de module MT5 → retourne la string, notre client sait gérer.
        return tf_u
