from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Dict

import pandas as pd

from .mt5_client import MT5Client
from .config import timeframe_to_mt5


def compute_features(df: pd.DataFrame) -> dict:
	if df.empty:
		return {}
	df = df.copy()
	df["sma20"] = df["close"].rolling(20).mean()
	df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
	last = df.iloc[-1]
	return {
		"close": float(last["close"]),
		"sma20": float(last.get("sma20", last["close"])),
		"atr": float(last.get("atr", 0.0)),
	}


def load_recent_m5(client: MT5Client, symbol: str, days: int, timeframe_code) -> pd.DataFrame:
	end = datetime.now(timezone.utc)
	start = end - timedelta(days=days)
	return client.fetch_ohlcv(symbol, timeframe_code, start, end)


def build_feature_map(client: MT5Client, symbols: List[str], timeframe: str, lookback_days: int) -> Dict[str, dict]:
	code = timeframe_to_mt5(timeframe)
	features: Dict[str, dict] = {}
	for s in symbols:
		df = load_recent_m5(client, s, lookback_days, code)
		features[s] = compute_features(df)
	return features
