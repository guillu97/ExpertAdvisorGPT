from __future__ import annotations

from typing import List, Tuple
import statistics

import MetaTrader5 as mt5


def list_all_symbols() -> List[str]:
	syms = mt5.symbols_get()
	if not syms:
		return []
	return [s.name for s in syms if getattr(s, "visible", True)]


def estimate_spread_points(symbol: str) -> float:
	info = mt5.symbol_info(symbol)
	if not info:
		return float("inf")
	spread_points = (info.spread if info.spread > 0 else (info.ask - info.bid) / info.point)
	return float(spread_points)


def rank_symbols_by_spread_and_volatility(symbols: List[str], recent_closes: dict[str, list[float]], weight_spread: float = 0.6, weight_vol: float = 0.4, max_symbols: int = 12) -> List[Tuple[str, float]]:
	scores = []
	for s in symbols:
		spread = estimate_spread_points(s)
		closes = recent_closes.get(s, [])
		vol = statistics.pstdev(closes[-50:]) if len(closes) >= 20 else 0.0
		# Score plus bas est meilleur (spread faible, vol élevée -> bonus)
		score = weight_spread * spread - weight_vol * vol
		scores.append((s, score))
	return sorted(scores, key=lambda x: x[1])[:max_symbols]
