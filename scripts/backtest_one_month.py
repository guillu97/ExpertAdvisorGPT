# src/backtest_one_month.py
from __future__ import annotations

import os
import sys
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from datetime import datetime as dt
import pandas as pd
import re
import dataclasses

# --- Ajouter le dossier racine du projet au sys.path pour permettre l'import de src/ ---
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import load_settings, timeframe_to_mt5
from src.mt5_client import MT5Client
from src.backtest import simulate_trading


# ----------------- Utilitaires -----------------
def _slug(s: str) -> str:
    s = str(s)
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)


def _parse_bool_env(name: str, default: bool = True) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def _get_logging_level() -> int:
    level_str = os.getenv("BT_LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_str, logging.INFO)


def _first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def _compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if df.empty:
        return out

    # cumul & dd
    df = df.copy()
    if "pnl_cum" not in df.columns:
        df["pnl_cum"] = df["pnl"].cumsum()

    # max drawdown sur la courbe de pnl_cum
    roll_max = df["pnl_cum"].cummax()
    dd = roll_max - df["pnl_cum"]
    out["max_drawdown"] = float(dd.max()) if not dd.empty else 0.0

    # win/loss
    wins = df["pnl"] > 0
    losses = df["pnl"] < 0
    n_trades = len(df)
    n_w = int(wins.sum())
    n_l = int(losses.sum())
    out["trades"] = n_trades
    out["wins"] = n_w
    out["losses"] = n_l
    out["win_rate"] = (n_w / n_trades) if n_trades else 0.0

    avg_win = df.loc[wins, "pnl"].mean() if n_w else 0.0
    avg_loss = -df.loc[losses, "pnl"].mean() if n_l else 0.0  # positif
    out["avg_win"] = float(avg_win or 0.0)
    out["avg_loss"] = float(avg_loss or 0.0)
    out["profit_factor"] = (df.loc[wins, "pnl"].sum() / max(1e-12, -df.loc[losses, "pnl"].sum())) if n_l else float("inf")
    out["expectancy"] = out["win_rate"] * out["avg_win"] - (1.0 - out["win_rate"]) * out["avg_loss"]

    # equity finale (si présent)
    if "equity_after" in df.columns and not df["equity_after"].isna().all():
        out["equity_final"] = float(df["equity_after"].dropna().iloc[-1])
    else:
        out["equity_final"] = None

    return out


def _print_metrics(df: pd.DataFrame):
    m = _compute_metrics(df)
    if not m:
        print("Aucune métrique calculable.")
        return
    print(
        "== Résumé =="
        f"\nTrades: {m['trades']}"
        f"\nWin rate: {m['win_rate']*100:.1f}%  (W={m['wins']} / L={m['losses']})"
        f"\nAvg win: {m['avg_win']:.2f}   Avg loss: {m['avg_loss']:.2f}"
        f"\nProfit factor: {m['profit_factor']:.2f}"
        f"\nExpectancy/trade: {m['expectancy']:.2f}"
        f"\nMax drawdown (pnl_cum): {m['max_drawdown']:.2f}"
        f"\nEquity finale: {m.get('equity_final', None)}"
    )


# ----------------- Script principal -----------------
def main():
    # ---------- Chargement settings & connexion MT5 ----------
    settings = load_settings()
    client = MT5Client(settings.mt5_login, settings.mt5_password, settings.mt5_server, settings.mt5_path)
    client.connect()

    try:
        # ---------- Paramètres généraux ----------
        # Arguments CLI: [1]=model (opt), [2]=days (opt), [3]=YYYY-MM-DD (opt)
        symbol = os.getenv("BT_SYMBOL", "EURUSD")
        timeframe = os.getenv("BT_TIMEFRAME", "M5")

        # Modèle: priorité CLI -> BT_MODEL -> settings
        model_arg: Optional[str] = sys.argv[1] if len(sys.argv) > 1 else None
        model = model_arg or os.getenv("BT_MODEL") or settings.openai_model

        # Jours de backtest: priorité CLI -> BT_DAYS -> 30
        try:
            days_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None
        except Exception:
            days_arg = None
        days = days_arg if days_arg is not None else int(os.getenv("BT_DAYS", "30"))
        if days <= 0:
            days = 30

        # Date cible optionnelle (format YYYY-MM-DD)
        date_arg: Optional[str] = sys.argv[3] if len(sys.argv) > 3 else os.getenv("BT_DATE")
        if date_arg:
            try:
                base_day = dt.strptime(date_arg, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except Exception:
                base_day = datetime.now(timezone.utc) - timedelta(days=days)
            end = base_day + timedelta(days=1)
            start = base_day
        else:
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=days)

        # --- Option: forcer une période explicite via env ---
        bt_start_env = os.getenv("BT_START")  # ex: 2025-07-01
        bt_end_env   = os.getenv("BT_END")    # ex: 2025-08-01 (exclu)
        if bt_start_env and bt_end_env:
            try:
                start = dt.strptime(bt_start_env, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                end   = dt.strptime(bt_end_env,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except Exception:
                pass

        # ---------- Paramètres avancés pour simulate_trading ----------
        # Spread/commission/ATR/logs
        use_spread = _parse_bool_env("BT_USE_SPREAD", True)
        manual_spread_price_env = float(os.getenv("BT_SPREAD_PRICE", "0") or 0)  # 0 => auto
        manual_spread_price = manual_spread_price_env if manual_spread_price_env > 0 else None

        # Commission par LOT (priorité: BT_COMMISSION_PER_LOT -> settings.commission_per_lot -> BT_COMMISSION -> 0)
        commission_per_lot_env = os.getenv("BT_COMMISSION_PER_LOT")
        commission_per_lot = _first_not_none(
            float(commission_per_lot_env) if commission_per_lot_env else None,
            getattr(settings, "commission_per_lot", None),
            float(os.getenv("BT_COMMISSION", "0.0")),
            0.0,
        )

        atr_len = int(os.getenv("BT_ATR_LEN", "14"))
        adx_min = float(os.getenv("BT_ADX_MIN", "20"))
        supertrend_len = int(os.getenv("BT_SUPERTREND_LEN", "10"))
        supertrend_mult = float(os.getenv("BT_SUPERTREND_MULT", "3"))

        # Risk & DD control (priorité: env -> settings -> défaut)
        starting_balance = float(_first_not_none(
            os.getenv("BT_BALANCE"),
            os.getenv("STARTING_BALANCE"),
            getattr(settings, "starting_balance", None),
            15000,
        ))
        risk_pct_per_trade = float(_first_not_none(
            os.getenv("BT_RISK_PCT"),
            getattr(settings, "risk_pct_per_trade", None),
            0.004,
        ))
        max_daily_dd_pct = float(_first_not_none(
            os.getenv("BT_MAX_DAILY_DD"),
            getattr(settings, "max_daily_dd_pct", None),
            0.02,
        ))
        max_trades_per_day = int(_first_not_none(
            os.getenv("BT_MAX_TRADES_DAY"),
            getattr(settings, "max_trades_per_day", None),
            6,
        ))

        log_level = _get_logging_level()

        # Progress (borne pour éviter le spam)
        try:
            progress_pct = int(os.getenv("BT_PROGRESS_PCT", "5"))
        except Exception:
            progress_pct = 5
        progress_pct = max(1, min(25, progress_pct))

        # Dossier de sortie
        out_root = os.getenv("BT_OUTDIR") or os.path.join(PROJECT_ROOT, "backtests")

        # --------- News gating (optionnel) ----------
        events_csv_path = os.getenv("BT_EVENTS_CSV")  # ex: ./data/events_eu_us_2025Q3.csv
        try:
            no_trade_before_high_min = int(os.getenv("BT_NO_TRADE_BEFORE_HIGH_MIN", "15"))
        except Exception:
            no_trade_before_high_min = 15
        try:
            no_trade_after_high_min = int(os.getenv("BT_NO_TRADE_AFTER_HIGH_MIN", "10"))
        except Exception:
            no_trade_after_high_min = 10

        # Petit récap des paramètres de run
        print(
            "[BT] Run params:",
            f"symbol={symbol}",
            f"timeframe={timeframe}",
            f"period={start}→{end}",
            f"model={model}",
            f"atr_len={atr_len}",
            f"adx_min={adx_min}",
            f"supertrend=({supertrend_len},{supertrend_mult})",
            f"use_spread={use_spread}",
            f"manual_spread={manual_spread_price or 'auto'}",
            f"commission_per_lot={commission_per_lot}",
            f"balance={starting_balance}",
            f"risk_pct={risk_pct_per_trade}",
            f"max_daily_dd_pct={max_daily_dd_pct}",
            f"max_trades_per_day={max_trades_per_day}",
            f"progress_log_every_pct={progress_pct}",
            f"outdir={out_root}",
            f"events_csv={events_csv_path or 'none'}",
            f"news_window=[-{no_trade_before_high_min}m,+{no_trade_after_high_min}m]",
        )

        # ---------- Lancement du backtest principal ----------
        common_kwargs = dict(
            api_key=settings.openai_api_key,
            model=model,
            atr_len=atr_len,
            use_spread=use_spread,
            manual_spread_price=manual_spread_price,
            commission_per_lot=commission_per_lot,
            log_level=log_level,
            # Risk & guards
            starting_balance=starting_balance,
            risk_pct_per_trade=risk_pct_per_trade,
            max_daily_dd_pct=max_daily_dd_pct,
            max_trades_per_day=max_trades_per_day,
            # Filtres / trend
            adx_min=adx_min,
            use_supertrend=True,
            supertrend_period=supertrend_len,
            supertrend_mult=supertrend_mult,
            progress_log_every_pct=progress_pct,
            # News
            events_csv_path=events_csv_path,
            no_trade_before_high_min=no_trade_before_high_min,
            no_trade_after_high_min=no_trade_after_high_min,
        )

        trades = simulate_trading(
            client, symbol, timeframe, start, end, **common_kwargs
        )

        # ---------- Fallback si aucune donnée (week-end, trous, etc.) ----------
        if not trades:
            try:
                _ = timeframe_to_mt5(timeframe)  # valide le TF
                for shift_days in range(1, 11):
                    end2 = end - timedelta(days=shift_days)
                    start2 = end2 - timedelta(days=days)
                    print(f"[BT] Fenêtre vide, tentative décalée de {shift_days}j: {start2} -> {end2}")
                    trades = simulate_trading(
                        client, symbol, timeframe, start2, end2, **common_kwargs
                    )
                    if trades:
                        start, end = start2, end2  # pour l’export
                        break
            except Exception as e:
                print(f"[BT] Fallback période échoué: {e}")

        # ---------- Arrêt si aucun trade ----------
        if not trades:
            print("Aucun trade simulé (données insuffisantes ou stratégie flat).")
            return

        # ---------- Résumé & export ----------
        df = pd.DataFrame([dataclasses.asdict(t) for t in trades])
        df["pnl_cum"] = df["pnl"].cumsum()

        # tail preview
        print(df.tail(10).to_string(index=False))
        _print_metrics(df)

        # Nom de fichier plus informatif et horodaté (UTC)
        run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        period_tag = f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
        nb_trades = len(df)

        symbol_slug = _slug(symbol)
        timeframe_slug = _slug(timeframe)
        model_slug = _slug(model)

        yyyy = start.strftime("%Y")
        yyyymm = start.strftime("%Y%m")

        base_dir = os.path.join(out_root, symbol_slug, timeframe_slug, model_slug, yyyy, yyyymm)
        os.makedirs(base_dir, exist_ok=True)

        out_name = f"backtest_{period_tag}__{run_ts}__n{nb_trades}.csv"
        out_path = os.path.join(base_dir, out_name)

        df.to_csv(out_path, index=False)
        print(f"Résultats sauvegardés: {out_path}")

    finally:
        # ---------- Shutdown propre du client MT5 ----------
        client.shutdown()


if __name__ == "__main__":
    main()