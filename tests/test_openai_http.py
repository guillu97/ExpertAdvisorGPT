from __future__ import annotations

import os
import sys

# Ajouter le dossier racine du projet au sys.path pour permettre l'import de src/
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

import httpx
from dotenv import load_dotenv


def main():
	# Charger .env pour exposer OPENAI_API_KEY dans l'env
	try:
		load_dotenv(override=True)
	except Exception:
		pass
	key = os.getenv("OPENAI_API_KEY", "")
	base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
	url = base.rstrip("/") + "/v1/models"
	print(f"KEY_SET={bool(key)} BASE={base}")
	try:
		r = httpx.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=10)
		print("STATUS=", r.status_code)
		print("BODY=", (r.text or "")[:200])
	except Exception as e:
		print("HTTPX_ERR=", type(e).__name__, str(e)[:200])
		# Tentative sans vérification SSL (diagnostic uniquement)
		try:
			r = httpx.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=10, verify=False)
			print("STATUS_NV=", r.status_code)
			print("BODY_NV=", (r.text or "")[:200])
		except Exception as e2:
			print("HTTPX_ERR_NV=", type(e2).__name__, str(e2)[:200])


if __name__ == "__main__":
	main()


