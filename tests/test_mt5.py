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
	connected = False
	try:
		connected = client.connect()
		acct = client.get_account_info()
		if not acct:
			print("Connecté mais aucune info compte.")
			return
		print("Connexion MT5: OK")
		print(f"Login: {acct.login}")
		print(f"Serveur: {settings.mt5_server}")
		print(f"Balance: {acct.balance}")
		print(f"Equity: {acct.equity}")
		print(f"Marge: {acct.margin}")
		print(f"Marge libre: {acct.margin_free}")
	finally:
		if connected:
			client.shutdown()


if __name__ == "__main__":
	main()
