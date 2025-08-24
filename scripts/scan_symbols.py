from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Dict
import pandas as pd

# Ajouter le dossier racine du projet au sys.path pour permettre l'import de src/
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from src.config import load_settings, timeframe_to_mt5
from src.mt5_client import MT5Client
from src.symbols import list_all_symbols, rank_symbols_by_spread_and_volatility


def main():
	settings = load_settings()
	client = MT5Client(settings.mt5_login, settings.mt5_password, settings.mt5_server, settings.mt5_path)
	client.connect()

	all_syms = list_all_symbols()
	code = timeframe_to_mt5("M5")
	end = datetime.now(timezone.utc)
	start = end - timedelta(days=7)

	# Collecter les derniers closings pour volatilité
	recent_closes: Dict[str, list[float]] = {}
	for s in all_syms[:100]:  # limiter un peu pour vitesse
		df = client.fetch_ohlcv(s, code, start, end)
		if df.empty:
			continue
		recent_closes[s] = [float(x) for x in df["close"].tail(200).tolist()]

	client.shutdown()

	ranked = rank_symbols_by_spread_and_volatility(list(recent_closes.keys()), recent_closes, max_symbols=settings.top_symbols)
	df = pd.DataFrame(ranked, columns=["symbol", "score"]) if ranked else pd.DataFrame()
	print(df.to_string(index=False) if not df.empty else "Aucun symbole classé.")

	out_path = os.path.join(PROJECT_ROOT, "scan_symbols.csv")
	df.to_csv(out_path, index=False)
	print(f"Résultats sauvegardés: {out_path}")


if __name__ == "__main__":
	main()


