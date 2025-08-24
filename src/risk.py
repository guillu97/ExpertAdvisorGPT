from __future__ import annotations

from typing import Optional

import MetaTrader5 as mt5


def clamp_lot(symbol_info, desired_volume: float) -> float:
	min_lot = symbol_info.volume_min
	max_lot = symbol_info.volume_max
	step = symbol_info.volume_step

	# Arrondir au step
	steps = round((desired_volume - min_lot) / step)
	rounded = min_lot + max(0, steps) * step
	return float(max(min(rounded, max_lot), min_lot))


def compute_volume_for_risk(symbol: str, balance: float, risk_fraction: float, entry_price: float, stop_price: float) -> float:
	"""Calcule un volume (lots) pour risquer une fraction du solde.

	Approche: utilise tick_value/tick_size afin d'estimer la valeur par point.
	"""
	symbol_info = mt5.symbol_info(symbol)
	if not symbol_info:
		return 0.0

	distance_price = abs(entry_price - stop_price)
	if distance_price <= 0:
		return 0.0

	# Valeur monétaire par 'point' pour 1 lot
	point_value_money_per_lot = (symbol_info.trade_tick_value / symbol_info.trade_tick_size) * symbol_info.point

	if point_value_money_per_lot <= 0:
		return 0.0

	risk_money = balance * risk_fraction
	points_distance = distance_price / symbol_info.point
	if points_distance <= 0:
		return 0.0

	# volume = risque / (distance_points * valeur_par_point_par_lot)
	desired_volume = risk_money / (points_distance * point_value_money_per_lot)
	return clamp_lot(symbol_info, desired_volume)
