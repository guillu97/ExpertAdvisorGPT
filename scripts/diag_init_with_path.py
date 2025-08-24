from __future__ import annotations

import os
import sys

# Ajouter le dossier racine du projet au sys.path pour permettre l'import de src/
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from src.config import load_settings


def main():
	try:
		import MetaTrader5 as mt5
	except Exception as e:
		print(f"Import MetaTrader5 a échoué: {e}")
		return

	settings = load_settings()
	login = settings.mt5_login or 0
	password = settings.mt5_password or ""
	server = settings.mt5_server or ""
	# Priorité à l'argument CLI, sinon MT5_PATH de l'env, sinon settings
	path_arg = sys.argv[1] if len(sys.argv) > 1 else None
	path_env = os.getenv("MT5_PATH")
	path = path_arg or path_env or settings.mt5_path

	print(f"Serveur: {server}")
	print(f"Login: {login}")
	print(f"MT5_PATH utilisé: {path or '(auto)'}")

	ok = mt5.initialize(path=path, login=login, password=password, server=server)
	print(f"initialize(login)= {ok} last_error= {mt5.last_error()}")
	if not ok:
		return

	acct = mt5.account_info()
	print(f"account_info= {acct}")

	mt5.shutdown()
	print("shutdown() fait.")


if __name__ == "__main__":
	main()


