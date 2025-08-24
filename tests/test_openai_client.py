from __future__ import annotations

import os
import sys

# Ajouter le dossier racine du projet au sys.path pour permettre l'import de src/
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from src.config import load_settings
from src.gpt_strategy import get_client


def main():
	settings = load_settings()
	model = sys.argv[1] if len(sys.argv) > 1 else os.getenv("TEST_MODEL", settings.openai_model)
	client = get_client(settings.openai_api_key, settings.openai_base_url)
	if client is None:
		print("Client OpenAI indisponible (fallback) — vérifiez OPENAI_API_KEY / certificats.")
		return
	try:
		resp = client.responses.create(
			model=model,
			input=[
				{"role": "user", "content": [{"type": "input_text", "text": "ping"}]},
			],
			max_output_tokens=32,
		)
		text = getattr(resp, "output_text", None) or ""
		if not text:
			# fallback chat.completions
			resp2 = client.chat.completions.create(
				model=model,
				messages=[
					{"role": "system", "content": "Réponds en un mot."},
					{"role": "user", "content": "ping"},
				],
			)
			text = resp2.choices[0].message.content or ""
		print(f"OK: {text[:50]}")
	except Exception as e:
		print(f"Erreur OpenAI: {e}")


if __name__ == "__main__":
	main()
