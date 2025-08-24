from __future__ import annotations

import os
import sys

# Ajouter le dossier racine du projet au sys.path pour permettre l'import de src/
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from src.config import load_settings
from src.mt5_client import MT5Client


def main():
	settings = load_settings()
	client = MT5Client(settings.mt5_login, settings.mt5_password, settings.mt5_server, settings.mt5_path)
	client.connect()

	symbol = os.getenv("TEST_SYMBOL", "EURUSD")
	volume = float(os.getenv("TEST_VOLUME", "0.01"))
	action = os.getenv("TEST_ACTION", "buy")

	import MetaTrader5 as mt5
	info = mt5.symbol_info(symbol)
	if not info:
		print(f"Symbole non trouvé: {symbol}")
		client.shutdown()
		return

	price = info.ask if action == "buy" else info.bid
	# SL/TP à 100 points par défaut
	sl = price - 100 * info.point if action == "buy" else price + 100 * info.point
	tp = price + 100 * info.point if action == "buy" else price - 100 * info.point

	res = client.place_market_order(symbol, action, volume, sl=sl, tp=tp, comment="test-0.01")
	print(res)
	client.shutdown()


if __name__ == "__main__":
	main()
