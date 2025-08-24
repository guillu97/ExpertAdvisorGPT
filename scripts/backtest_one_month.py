from __future__ import annotations

import os
import sys
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from datetime import datetime as dt
import pandas as pd
import re


# --- Ajouter le dossier racine du projet au sys.path pour permettre l'import de src/ ---
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import load_settings, timeframe_to_mt5
from src.mt5_client import MT5Client
from src.backtest import simulate_trading


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


def main():
    # ---------- Chargement settings & connexion MT5 ----------
    settings = load_settings()
    client = MT5Client(settings.mt5_login, settings.mt5_password, settings.mt5_server, settings.mt5_path)
    client.connect()

    try:
        # ---------- Paramètres généraux ----------
        symbol = os.getenv("BT_SYMBOL", "EURUSD")
        timeframe = os.getenv("BT_TIMEFRAME", "M5")

        # Modèle: priorité à l'argument CLI, sinon variable d'env BT_MODEL, sinon settings
        model_arg: Optional[str] = sys.argv[1] if len(sys.argv) > 1 else None
        model = model_arg or os.getenv("BT_MODEL") or settings.openai_model

        # Jours de backtest: argument CLI #2, sinon env BT_DAYS, sinon 30
        days_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None
        days = days_arg if days_arg is not None else int(os.getenv("BT_DAYS", "30"))

        # Date cible optionnelle (format YYYY-MM-DD) en arg #3 ou env
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

        # --- Option: forcer une période explicite par variables d'env ---
        bt_start_env = os.getenv("BT_START")  # ex: 2025-08-18
        bt_end_env   = os.getenv("BT_END")    # ex: 2025-08-23 (exclu)
        if bt_start_env and bt_end_env:
            start = dt.strptime(bt_start_env, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end   = dt.strptime(bt_end_env,   "%Y-%m-%d").replace(tzinfo=timezone.utc)

        # ---------- Paramètres avancés pour simulate_trading ----------
        # Spread/commission/ATR/logs
        use_spread = _parse_bool_env("BT_USE_SPREAD", True)
        # exemple: 0.00010 ~ 1 pip EURUSD ; si 0 → tentera de lire MT5
        manual_spread_price_env = float(os.getenv("BT_SPREAD_PRICE", "0") or 0)
        manual_spread_price = manual_spread_price_env if manual_spread_price_env > 0 else None
        commission_per_trade = float(os.getenv("BT_COMMISSION", "0.0"))
        atr_len = int(os.getenv("BT_ATR_LEN", "14"))
        log_level = _get_logging_level()

        # Progress bar (logs) — % de progression (0 pour désactiver)
        progress_pct = int(os.getenv("BT_PROGRESS_PCT", "5"))

        # (facultatif) autres kwargs si tu as étendu simulate_trading
        # cooldown_bars = int(os.getenv("BT_COOLDOWN", "3"))
        # adx_min = float(os.getenv("BT_ADX_MIN", "20"))

        # Petit récap des paramètres de run
        print(
            "[BT] Run params:",
            f"symbol={symbol}",
            f"timeframe={timeframe}",
            f"period={start}→{end}",
            f"model={model}",
            f"atr_len={atr_len}",
            f"use_spread={use_spread}",
            f"manual_spread={manual_spread_price or 'auto'}",
            f"commission={commission_per_trade}",
            f"progress_log_every_pct={progress_pct}",
        )

        # ---------- Lancement du backtest principal ----------
        trades = simulate_trading(
            client, symbol, timeframe, start, end,
            api_key=settings.openai_api_key,
            model=model,
            atr_len=atr_len,
            use_spread=use_spread,
            manual_spread_price=manual_spread_price,
            commission_per_trade=commission_per_trade,
            log_level=log_level,
            # cooldown_bars=cooldown_bars,
            # adx_min=adx_min,
            progress_log_every_pct=progress_pct,  # <<<<<< barre de progression via logs
        )

        # ---------- Fallback si aucune donnée (week-end, jour férié, etc.) ----------
        if not trades:
            try:
                _ = timeframe_to_mt5(timeframe)  # vérifie que le timeframe est valide
                for shift_days in range(1, 11):
                    end2 = end - timedelta(days=shift_days)
                    start2 = end2 - timedelta(days=days)
                    print(f"[BT] Fenêtre initiale vide, tentative avec décalage {shift_days}j: {start2} -> {end2}")
                    trades = simulate_trading(
                        client, symbol, timeframe, start2, end2,
                        api_key=settings.openai_api_key,
                        model=model,
                        atr_len=atr_len,
                        use_spread=use_spread,
                        manual_spread_price=manual_spread_price,
                        commission_per_trade=commission_per_trade,
                        log_level=log_level,
                        # cooldown_bars=cooldown_bars,
                        # adx_min=adx_min,
                        progress_log_every_pct=progress_pct,
                    )
                    if trades:
                        start, end = start2, end2  # pour nommer le fichier correctement ensuite
                        break
            except Exception:
                pass

        # ---------- Arrêt si aucun trade ----------
        if not trades:
            print("Aucun trade simulé (données insuffisantes ou stratégie flat).")
            return

        # ---------- Résumé & export ----------
        import dataclasses
        df = pd.DataFrame([dataclasses.asdict(t) for t in trades])
        # métriques simples
        df["pnl_cum"] = df["pnl"].cumsum()
        print(df.tail(10).to_string(index=False))
        print(f"Trades: {len(df)}, PnL total: {df['pnl'].sum():.5f}, PnL max cumulé: {df['pnl_cum'].max():.5f}")

        # Nom de fichier plus informatif et horodaté
        run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")  # horodatage UTC compact
        period_tag = f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
        nb_trades = len(df)

        # pièces pour le chemin
        symbol_slug = _slug(symbol)
        timeframe_slug = _slug(timeframe)
        model_slug = _slug(model)
        # arborescence: backtests/<symbol>/<timeframe>/<model>/<yyyy>/<yyyymm>/
        yyyy = start.strftime("%Y")
        yyyymm = start.strftime("%Y%m")
        base_dir = os.path.join(PROJECT_ROOT, "backtests", symbol_slug, timeframe_slug, model_slug, yyyy, yyyymm)

        os.makedirs(base_dir, exist_ok=True)

        # nom de fichier: backtest_<period>__<ts>__n<nb>.csv
        out_name = f"backtest_{period_tag}__{run_ts}__n{nb_trades}.csv"
        out_path = os.path.join(base_dir, out_name)

        df.to_csv(out_path, index=False)
        print(f"Résultats sauvegardés: {out_path}")

    finally:
        # ---------- Shutdown propre du client MT5 ----------
        client.shutdown()


if __name__ == "__main__":
    main()
