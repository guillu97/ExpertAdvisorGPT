from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import List

from .config import load_settings, timeframe_to_mt5
from .mt5_client import MT5Client
from .symbols import list_all_symbols
from .opportunity_scanner import build_feature_map
from .gpt_strategy import gpt_decide
from .risk import compute_volume_for_risk


def pick_symbols(symbols: List[str], max_symbols: int) -> List[str]:
	# Heuristique simple: garder FX majors et indices courants si présents
	preferred_prefixes = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD", "US500", "NAS100", "DE40", "UK100")
	ordered = [s for p in preferred_prefixes for s in symbols if s.startswith(p)]
	if not ordered:
		ordered = symbols
	return ordered[:max_symbols]


def run_once():
	settings = load_settings()
	client = MT5Client(settings.mt5_login, settings.mt5_password, settings.mt5_server, settings.mt5_path)
	client.connect()

	all_symbols = list_all_symbols()
	symbols = pick_symbols(all_symbols, settings.top_symbols)

	features_map = build_feature_map(client, symbols, settings.timeframe, settings.lookback_days)

	account = client.get_account_info()
	balance = float(account.balance) if account else 0.0
	free_margin = float(getattr(account, "margin_free", 0.0)) if account else 0.0
	equity = float(getattr(account, "equity", balance)) if account else balance

	# Limite de positions ouvertes
	open_positions = client.get_positions() or []
	if len(open_positions) >= settings.max_open_trades:
		print(f"Nombre de positions ouvertes ({len(open_positions)}) >= max autorisé ({settings.max_open_trades}), on saute ce cycle.")
		client.shutdown()
		return

	# Calcul de l'exposition nominale actuelle (approximation): somme(prix_ouverture * contract_size * volume)
	try:
		import MetaTrader5 as mt5
	except Exception:
		mt5 = None  # type: ignore

	current_exposure_value = 0.0
	if mt5 is not None:
		for pos in (client.get_positions() or []):
			info_pos = mt5.symbol_info(pos.symbol)
			if info_pos and info_pos.trade_contract_size > 0:
				current_exposure_value += float(pos.price_open) * float(info_pos.trade_contract_size) * float(pos.volume)

	for s in symbols:
		feat = features_map.get(s, {})
		if not feat:
			continue
		decision = gpt_decide(feat, api_key=settings.openai_api_key, model=settings.openai_model, base_url=settings.openai_base_url)
		if decision.decision == "flat":
			continue

		# Sizing: SL points -> distance en prix via point
		import MetaTrader5 as mt5
		info = mt5.symbol_info(s)
		if not info or info.point <= 0:
			continue
		entry_price = info.ask if decision.decision == "buy" else info.bid
		stop_price = entry_price - decision.sl_points * info.point if decision.decision == "buy" else entry_price + decision.sl_points * info.point
		volume = compute_volume_for_risk(s, balance, settings.account_risk_per_trade, entry_price, stop_price)
		if volume <= 0:
			continue

		# Vérification levier/couverture: éviter de dépasser settings.max_leverage
		contract_size = float(getattr(info, "trade_contract_size", 0.0) or 0.0)
		if equity > 0 and contract_size > 0:
			proposed_nominal = float(entry_price) * contract_size * float(volume)
			new_total_nominal = current_exposure_value + proposed_nominal
			leverage_used = new_total_nominal / equity
			if leverage_used > float(getattr(settings, "max_leverage", 30)):
				print(f"Levier dépassé si on prend {s}: {leverage_used:.2f}x > max {settings.max_leverage}x. On saute.")
				continue

		# Vérification marge requise vs marge libre
		# Utiliser ORDER_CALC_MODE_MARGIN / order_calc_margin
		order_type = mt5.ORDER_TYPE_BUY if decision.decision == "buy" else mt5.ORDER_TYPE_SELL
		calc_margin = mt5.order_calc_margin(order_type, s, volume, entry_price)
		if calc_margin is None:
			print(f"Impossible de calculer la marge requise pour {s}")
			continue
		if free_margin <= 0 or calc_margin > free_margin:
			print(f"Marge libre insuffisante: requise {calc_margin:.2f} > libre {free_margin:.2f} sur {s}")
			continue

		# Vérifier encore la limite de positions avant d'envoyer
		open_positions = client.get_positions() or []
		if len(open_positions) >= settings.max_open_trades:
			print("Limite de positions atteinte, stop.")
			break

		# Envoi ordre
		sl = stop_price
		tp = entry_price + decision.tp_points * info.point if decision.decision == "buy" else entry_price - decision.tp_points * info.point
		res = client.place_market_order(s, decision.decision, volume, sl=sl, tp=tp, comment=f"gpt:{decision.reason[:20]}")
		print(s, decision, volume, res)

	client.shutdown()


def main_loop():
	# Boucle simple: exécuter toutes les 5 minutes
	while True:
		start = time.time()
		try:
			run_once()
		except Exception as e:
			print("Erreur:", e)
			time.sleep(5)
		continue_delay = max(0, 300 - (time.time() - start))
		time.sleep(continue_delay)


if __name__ == "__main__":
	main_loop()
