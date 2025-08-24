from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Dict

# Ajouter le dossier racine du projet au sys.path pour permettre l'import de src/
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from src.config import load_settings, timeframe_to_mt5
from src.mt5_client import MT5Client
from src.symbols import list_all_symbols


def pick_symbols_available(all_symbols: List[str], preferred: List[str], limit: int) -> List[str]:
	ordered = [s for p in preferred for s in all_symbols if s.startswith(p)]
	if not ordered:
		ordered = all_symbols
	return ordered[:limit]


def summarize_volumes(df: pd.DataFrame) -> dict:
	if df.empty:
		return {"bars": 0, "tick_volume_sum": 0, "tick_volume_avg": 0.0}
	return {
		"bars": int(len(df)),
		"tick_volume_sum": int(df["tick_volume"].sum()),
		"tick_volume_avg": float(df["tick_volume"].mean()),
	}


def main():
	settings = load_settings()
	client = MT5Client(settings.mt5_login, settings.mt5_password, settings.mt5_server, settings.mt5_path)
	client.connect()

	all_syms = list_all_symbols()
	preferred = [
		"EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD",
		"US500", "NAS100", "DE40", "UK100",
	]
	syms = pick_symbols_available(all_syms, preferred, settings.top_symbols)

	print(f"Test volumes M5: {len(syms)} symboles sur {settings.lookback_days} jours")
	code = timeframe_to_mt5("M5")
	end = datetime.now(timezone.utc)
	start = end - timedelta(days=settings.lookback_days)

	rows: List[dict] = []
	for s in syms:
		df = client.fetch_ohlcv(s, code, start, end)
		sumv = summarize_volumes(df)
		rows.append({"symbol": s, **sumv})

	client.shutdown()

	df_out = pd.DataFrame(rows)
	if df_out.empty:
		print("Aucune donnée récupérée.")
		return

	df_sorted = df_out.sort_values(by=["tick_volume_sum", "tick_volume_avg"], ascending=False)
	print(df_sorted.to_string(index=False))

	# Sauvegarde CSV rapide pour inspection éventuelle
	out_path = os.path.join(PROJECT_ROOT, "mt5_volumes_m5.csv")
	df_sorted.to_csv(out_path, index=False)
	print(f"Résultats sauvegardés: {out_path}")


if __name__ == "__main__":
	main()


