from __future__ import annotations

import os
import sys

# Ajouter le dossier racine du projet au sys.path pour permettre l'import de src/
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from src.config import load_settings
from src.gpt_strategy import gpt_decide


def main():
	settings = load_settings()
	model = os.getenv("TEST_MODEL", settings.openai_model)
	features = {"close": 1.2345, "sma20": 1.2340, "atr": 0.0005}
	print("Testing gpt_decide with:", features, "model=", model)
	out = gpt_decide(features, api_key=settings.openai_api_key, model=model, base_url=settings.openai_base_url)
	print("Decision=", out.decision, "SLpts=", out.sl_points, "TPpts=", out.tp_points, "reason=", out.reason)


if __name__ == "__main__":
	main()
