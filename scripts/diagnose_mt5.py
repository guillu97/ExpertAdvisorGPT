from __future__ import annotations

import os
import sys

# Ajouter le dossier racine du projet au sys.path pour permettre l'import de src/
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from src.config import load_settings


def log(line: str):
	print(line, flush=True)
	with open("mt5_diag.txt", "a", encoding="utf-8") as f:
		f.write(line + "\n")


def main():
	# Reset log file
	try:
		open("mt5_diag.txt", "w", encoding="utf-8").close()
	except Exception:
		pass

	try:
		import MetaTrader5 as mt5
	except Exception as e:
		log(f"Import MetaTrader5 a échoué: {e}")
		return

	settings = load_settings()
	log(f"Serveur: {settings.mt5_server}")
	log(f"Login: {settings.mt5_login}")
	log(f"MT5_PATH: {settings.mt5_path or '(auto)'}")

	ok = mt5.initialize(path=settings.mt5_path)
	log(f"initialize()= {ok} last_error= {mt5.last_error()}")
	if not ok:
		return

	if settings.mt5_login and settings.mt5_password and settings.mt5_server:
		ok_login = mt5.login(settings.mt5_login, password=settings.mt5_password, server=settings.mt5_server)
		log(f"login()= {ok_login} last_error= {mt5.last_error()}")
	else:
		log("Login/pass/server manquants, tentative de connexion au terminal déjà connecté.")

	acct = mt5.account_info()
	log(f"account_info= {acct}")

	mt5.shutdown()
	log("shutdown() fait.")


if __name__ == "__main__":
	main()
