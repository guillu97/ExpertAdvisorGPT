from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime, timedelta, timezone

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

import MetaTrader5 as mt5


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

	@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
	def connect(self) -> bool:
		# Tentative 1: initialize avec identifiants
		ok = mt5.initialize(path=self.terminal_path, login=self.login, password=self.password, server=self.server)
		if not ok:
			# Tentative 2: initialize simple, puis login
			if not mt5.initialize(path=self.terminal_path):
				raise RuntimeError(f"MT5 initialize a échoué: {mt5.last_error()}")
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
		if not self.ensure_symbol(symbol):
			raise RuntimeError(f"Symbole introuvable: {symbol}")
		rates = mt5.copy_rates_range(symbol, timeframe, start, end)
		if rates is None or len(rates) == 0:
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
