from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime, timedelta, timezone

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

# sous windows
#import MetaTrader5 as mt5

# sous linux
from pymt5linux import MetaTrader5
mt5 = MetaTrader5(host="localhost", port=8001)

@dataclass
class OrderResult:
	order_id: Optional[int]
	success: bool
	message: str


class MT5Client:
	def __init__(self, login: Optional[int], password: Optional[str], server: Optional[str], terminal_path: Optional[str] = None):
		self.login = login
		self.password = password
		self.server = server
		self.terminal_path = terminal_path
		self._connected = False
	
	def _normalize_win_path(self, p: Optional[str]) -> Optional[str]:
		if not p:
			return None
		# /opt/wineprefix/drive_c/... -> C:\...
		if p.startswith("/opt/wineprefix/drive_c/"):
			p = "C:\\" + p.split("/opt/wineprefix/drive_c/")[1]
		# slashes -> backslashes
		return p.replace("/", "\\")

	@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
	def connect(self) -> bool:
		p = self._normalize_win_path(self.terminal_path)
		candidates = []
		# 1) chemin fourni (normalisé)
		if p:
			candidates.append(("provided", p))
		# 2) chemin par défaut d'install MT5
		candidates.append(("default", r"C:\Program Files\MetaTrader 5\terminal64.exe"))
		# 3) sans path (s’appuie sur install/terminal déjà lancé)
		candidates.append(("none", None))

		last_err = None
		for label, cand in candidates:
			if cand:
				ok = mt5.initialize(path=cand, login=self.login, password=self.password, server=self.server)
			else:
				ok = mt5.initialize(login=self.login, password=self.password, server=self.server)
			if ok:
				self._connected = True
				return True
			last_err = mt5.last_error()

		# fallback: initialize simple puis login (si identifiants fournis)
		if not mt5.initialize():
			raise RuntimeError(f"MT5 initialize a échoué: {last_err or mt5.last_error()}")
		if self.login and self.password and self.server:
			if not mt5.login(self.login, password=self.password, server=self.server):
				raise RuntimeError(f"MT5 login a échoué: {mt5.last_error()}")

		self._connected = True
		return True

	def shutdown(self) -> None:
		if self._connected:
			mt5.shutdown()
			self._connected = False

	def ensure_symbol(self, symbol: str) -> bool:
		info = mt5.symbol_info(symbol)
		if info is None:
			return False
		if not info.visible:
			mt5.symbol_select(symbol, True)
		return True

	def get_account_info(self):
		return mt5.account_info()

	def get_positions(self):
		return mt5.positions_get()

	def get_orders(self):
		return mt5.orders_get()

	def fetch_ohlcv(self, symbol: str, timeframe, start: datetime, end: datetime) -> pd.DataFrame:
		# 1) convertir timeframe -> constante MT5
		if isinstance(timeframe, str):
			tf_map = {
				"M1": mt5.TIMEFRAME_M1, "M2": mt5.TIMEFRAME_M2, "M3": mt5.TIMEFRAME_M3, "M4": mt5.TIMEFRAME_M4,
				"M5": mt5.TIMEFRAME_M5, "M6": mt5.TIMEFRAME_M6, "M10": mt5.TIMEFRAME_M10, "M12": mt5.TIMEFRAME_M12,
				"M15": mt5.TIMEFRAME_M15, "M20": mt5.TIMEFRAME_M20, "M30": mt5.TIMEFRAME_M30,
				"H1": mt5.TIMEFRAME_H1, "H2": mt5.TIMEFRAME_H2, "H3": mt5.TIMEFRAME_H3, "H4": mt5.TIMEFRAME_H4,
				"H6": mt5.TIMEFRAME_H6, "H8": mt5.TIMEFRAME_H8, "H12": mt5.TIMEFRAME_H12,
				"D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1,
			}
			tf = tf_map.get(timeframe.upper())
		else:
			tf = timeframe
		if tf is None or not isinstance(tf, int):
			raise ValueError(f"Timeframe invalide pour MetaTrader5: {timeframe!r}")

		# 2) datetimes naïfs (UTC) pour compatibilité MT5
		if start.tzinfo is not None:
			start = start.astimezone(timezone.utc).replace(tzinfo=None)
		if end.tzinfo is not None:
			end = end.astimezone(timezone.utc).replace(tzinfo=None)

		# 3) symbole dispo
		if not self.ensure_symbol(symbol):
			raise RuntimeError(f"Symbole introuvable: {symbol}")

		# 4) appel MT5
		rates = mt5.copy_rates_range(symbol, tf, start, end)
		if rates is None:
			raise RuntimeError(f"copy_rates_range a renvoyé None: {mt5.last_error()}")
		if len(rates) == 0:
			return pd.DataFrame()

		df = pd.DataFrame(rates)
		df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
		df.set_index("time", inplace=True)
		return df[["open", "high", "low", "close", "tick_volume"]]


	def place_market_order(self, symbol: str, action: str, volume: float, sl: Optional[float] = None, tp: Optional[float] = None, comment: str = "") -> OrderResult:
		if action not in {"buy", "sell"}:
			return OrderResult(order_id=None, success=False, message="Action invalide (buy/sell)")
		if not self.ensure_symbol(symbol):
			return OrderResult(order_id=None, success=False, message=f"Symbole non sélectionné: {symbol}")

		symbol_info = mt5.symbol_info(symbol)
		price = symbol_info.ask if action == "buy" else symbol_info.bid

		request = {
			"action": mt5.TRADE_ACTION_DEAL,
			"symbol": symbol,
			"volume": float(volume),
			"type": mt5.ORDER_TYPE_BUY if action == "buy" else mt5.ORDER_TYPE_SELL,
			"price": price,
			"sl": sl if sl else 0.0,
			"tp": tp if tp else 0.0,
			"deviation": 20,
			"magic": 20250823,
			"comment": comment,
			"type_time": mt5.ORDER_TIME_GTC,
			"type_filling": mt5.ORDER_FILLING_IOC,
		}
		result = mt5.order_send(request)
		if result is None:
			return OrderResult(order_id=None, success=False, message=f"order_send None: {mt5.last_error()}")
		if result.retcode != mt5.TRADE_RETCODE_DONE:
			return OrderResult(order_id=None, success=False, message=f"retcode {result.retcode}: {result.comment}")
		return OrderResult(order_id=result.order, success=True, message="OK")
