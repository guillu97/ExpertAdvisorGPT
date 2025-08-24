from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError
from pydantic import field_validator
from typing import Optional
import os


class Settings(BaseModel):
	openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
	openai_base_url: Optional[str] = Field(default=None, alias="OPENAI_BASE_URL")
	openai_model: str = Field(default=os.getenv("OPENAI_MODEL", "gpt-5"), alias="OPENAI_MODEL")

	mt5_login: Optional[int] = Field(default=None, alias="MT5_LOGIN")
	mt5_password: Optional[str] = Field(default=None, alias="MT5_PASSWORD")
	mt5_server: Optional[str] = Field(default=os.getenv("MT5_SERVER", "ICMarketsSC-Demo"), alias="MT5_SERVER")
	mt5_path: Optional[str] = Field(default=os.getenv("MT5_PATH"), alias="MT5_PATH")

	account_risk_per_trade: float = Field(default=float(os.getenv("ACCOUNT_RISK_PER_TRADE", 0.01)))
	max_leverage: float = Field(default=float(os.getenv("MAX_LEVERAGE", 30)))
	max_open_trades: int = Field(default=int(os.getenv("MAX_OPEN_TRADES", 5)))

	timeframe: str = Field(default=os.getenv("TIMEFRAME", "M5"))
	lookback_days: int = Field(default=int(os.getenv("LOOKBACK_DAYS", 7)))
	top_symbols: int = Field(default=int(os.getenv("TOP_SYMBOLS", 8)))

	@field_validator("timeframe")
	@classmethod
	def validate_timeframe(cls, v: str) -> str:
		allowed = {"M1", "M5", "M15", "M30", "H1", "H4", "D1"}
		if v not in allowed:
			raise ValueError(f"TIMEFRAME invalide: {v}. Choisir parmi {allowed}.")
		return v


def timeframe_to_mt5(timeframe: str):
	"""Convertit une chaîne (ex: 'M5') en constante MetaTrader5.TIMEFRAME_*."""
	try:
		import MetaTrader5 as mt5
	except Exception:  # pragma: no cover
		return None

	mapping = {
		"M1": mt5.TIMEFRAME_M1,
		"M5": mt5.TIMEFRAME_M5,
		"M15": mt5.TIMEFRAME_M15,
		"M30": mt5.TIMEFRAME_M30,
		"H1": mt5.TIMEFRAME_H1,
		"H4": mt5.TIMEFRAME_H4,
		"D1": mt5.TIMEFRAME_D1,
	}
	return mapping.get(timeframe)


def load_settings() -> Settings:
	"""Charge la configuration depuis l'environnement (dotenv recommandé)."""
	try:
		from dotenv import load_dotenv
		# IMPORTANT: override=True pour que .env remplace les variables existantes
		load_dotenv(override=True)
	except Exception:
		pass

	env_map = {k: v for k, v in os.environ.items()}
	try:
		return Settings.model_validate(env_map)
	except ValidationError as e:
		raise RuntimeError(f"Configuration invalide: {e}")
