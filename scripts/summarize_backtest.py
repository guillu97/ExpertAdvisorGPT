from __future__ import annotations

import os
import sys
import pandas as pd


def summarize(csv_path: str) -> None:
	if not os.path.exists(csv_path):
		print(f"Fichier introuvable: {csv_path}")
		return
	df = pd.read_csv(csv_path)
	if df.empty:
		print("CSV vide")
		return
	# Nettoyage colonnes attendues
	if "pnl_cum" not in df.columns:
		df["pnl_cum"] = df["pnl"].cumsum()
	trades = len(df)
	win = int((df["pnl"] > 0).sum())
	loss = int((df["pnl"] < 0).sum())
	flat = trades - win - loss
	win_rate = (win / trades) * 100 if trades else 0.0
	total_pnl = float(df["pnl"].sum())
	avg_pnl = float(df["pnl"].mean()) if trades else 0.0
	max_cum = float(df["pnl_cum"].max())
	min_cum = float(df["pnl_cum"].min())
	max_drawdown = float((df["pnl_cum"].cummax() - df["pnl_cum"]).max())
	# Fallback ratio
	reason_col = df.get("reason")
	fallback_ratio = None
	if reason_col is not None:
		fallback_ratio = float(reason_col.str.contains("fallback", na=False).mean()) * 100.0
		fallback_count = int(reason_col.str.contains("fallback", na=False).sum())
		print(f"Fallback décisions: {fallback_count}/{trades} ({fallback_ratio:.1f}%)")

	print(f"Trades: {trades}")
	print(f"Win/Loss/Flat: {win}/{loss}/{flat}  (Win rate: {win_rate:.1f}%)")
	print(f"PnL total: {total_pnl:.5f}  PnL moyen/trade: {avg_pnl:.6f}")
	print(f"PnL cumulé max: {max_cum:.5f}  Min: {min_cum:.5f}  Max DD: {max_drawdown:.5f}")


if __name__ == "__main__":
	csv = os.path.join(os.path.dirname(__file__), os.pardir, "backtest_last_month.csv")
	csv = os.path.abspath(csv)
	summarize(csv)


