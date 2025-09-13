# src/backtest.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Tuple
import logging
import math
import pandas as pd

from .mt5_client import MT5Client
from .config import timeframe_to_mt5
from .gpt_strategy import gpt_decide, StrategyOutput
from .econ_calendar import load_events_csv, build_high_impact_index, nearest_high_events

import time
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Trade:
    time: datetime
    symbol: str
    action: str
    entry: float
    sl: Optional[float]
    tp: Optional[float]
    exit: Optional[float]
    pnl: float = 0.0              # PnL en devise du compte
    reason: str = ""
    exit_reason: str = ""
    volume_lots: float = 0.0
    equity_before: float = 0.0
    equity_after: float = 0.0


# ---------- Indicateurs utilitaires ----------
def _compute_atr_true_range(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"]; low = df["low"]; prev_close = df["close"].shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def _slope(series: pd.Series, lookback: int = 5) -> pd.Series:
    return (series - series.shift(lookback)) / max(1, lookback)


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0); down = -delta.clip(upper=0.0)
    ma_up = up.rolling(n).mean(); ma_down = down.rolling(n).mean()
    rs = ma_up / (ma_down.replace(0, float("inf")))
    return 100 - (100 / (1 + rs))


def _vol_rel(close: pd.Series, n_short: int = 20, n_long: int = 100) -> pd.Series:
    short = close.pct_change().rolling(n_short).std()
    long = close.pct_change().rolling(n_long).std()
    return (short / (long.replace(0, float("inf"))))


# --- Wilder ADX (+DI/-DI) ---
def _adx_components(df: pd.DataFrame, n: int = 14):
    high = df["high"]; low = df["low"]; close = df["close"]
    up_move = high.diff(); down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move
    tr1 = (high - low).abs(); tr2 = (high - close.shift()).abs(); tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean().replace(0, 1e-12)
    plus_di = 100 * (plus_dm.ewm(alpha=1/n, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/n, adjust=False).mean() / atr)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-12)) * 100
    adx = dx.ewm(alpha=1/n, adjust=False).mean()
    return plus_di, minus_di, adx


# --- Supertrend ---
def _supertrend(df: pd.DataFrame, atr: pd.Series, period: int = 10, mult: float = 3.0):
    hl2 = (df["high"] + df["low"]) / 2.0
    atr_s = atr if atr is not None else _compute_atr_true_range(df, n=period)
    basic_ub = hl2 + mult * atr_s
    basic_lb = hl2 - mult * atr_s

    final_ub = basic_ub.copy(); final_lb = basic_lb.copy()
    st = pd.Series(index=df.index, dtype=float)
    dirn = pd.Series(index=df.index, dtype=int)

    dirn.iloc[0] = 1
    st.iloc[0] = final_lb.iloc[0]

    for i in range(1, len(df)):
        cprev = df["close"].iloc[i-1]
        # Final bands with carry
        final_ub.iloc[i] = basic_ub.iloc[i] if cprev <= final_ub.iloc[i-1] else min(basic_ub.iloc[i], final_ub.iloc[i-1])
        final_lb.iloc[i] = basic_lb.iloc[i] if cprev >= final_lb.iloc[i-1] else max(basic_lb.iloc[i], final_lb.iloc[i-1])

        # Direction switch
        if df["close"].iloc[i] > final_ub.iloc[i-1]:
            dirn.iloc[i] = 1
        elif df["close"].iloc[i] < final_lb.iloc[i-1]:
            dirn.iloc[i] = -1
        else:
            dirn.iloc[i] = dirn.iloc[i-1]

        st.iloc[i] = final_lb.iloc[i] if dirn.iloc[i] == 1 else final_ub.iloc[i]

    return st, dirn


def _exit_price_long(row: pd.Series, sl: float, tp: float) -> Tuple[Optional[float], Optional[str]]:
    o, h, l = float(row["open"]), float(row["high"]), float(row["low"])
    if o <= sl: return o, "SL(gap@open)"
    if o >= tp: return o, "TP(gap@open)"
    hit_tp = h >= tp; hit_sl = l <= sl
    if hit_tp and hit_sl: return (tp if (tp - o) <= (o - sl) else sl), "TP/SL(closest_to_open)"
    if hit_tp: return tp, "TP"
    if hit_sl: return sl, "SL"
    return None, None


def _exit_price_short(row: pd.Series, sl: float, tp: float) -> Tuple[Optional[float], Optional[str]]:
    o, h, l = float(row["open"]), float(row["high"]), float(row["low"])
    if o >= sl: return o, "SL(gap@open)"
    if o <= tp: return o, "TP(gap@open)"
    hit_sl = h >= sl; hit_tp = l <= tp
    if hit_sl and hit_tp: return (tp if (o - tp) <= (sl - o) else sl), "TP/SL(closest_to_open)"
    if hit_sl: return sl, "SL"
    if hit_tp: return tp, "TP"
    return None, None


def _is_finite(x: float) -> bool:
    try:
        return (x is not None) and (not math.isnan(float(x))) and (not math.isinf(float(x)))
    except Exception:
        return False


def _bars_per_day(timeframe: str) -> int:
    tf = timeframe.upper()
    if   tf == "M1":  return 1440
    elif tf == "M5":  return 288
    elif tf == "M15": return 96
    elif tf == "M30": return 48
    elif tf == "H1":  return 24
    elif tf == "H4":  return 6
    elif tf == "D1":  return 1
    return 288


def _tf_to_minutes(tf: str) -> int:
    tf = tf.upper().strip()
    if tf.startswith("M"):
        return int(tf[1:])
    if tf == "H1": return 60
    if tf == "H2": return 120
    if tf == "H3": return 180
    if tf == "H4": return 240
    if tf == "D1": return 1440
    return 5


def simulate_trading(
    client: MT5Client,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    api_key: Optional[str],
    model: str = "gpt-5-nano",
    *,
    atr_len: int = 14,
    min_bars: int = 220,
    use_spread: bool = True,
    manual_spread_price: Optional[float] = None,
    commission_per_trade: float = 0.0,            # backward compat (ignored if commission_per_lot set)
    commission_per_lot: Optional[float] = None,   # NEW: per-lot commission
    log_level: int = logging.INFO,

    # Robustesse / filtres
    use_sma200_filter: bool = True,
    atr_min_threshold: float = 0.00025,
    min_sl_atr_mult: float = 1.5,
    min_tp_atr_mult: float = 2.5,
    cooldown_bars_after_any_exit: int = 2,
    cooldown_bars_after_sl: int = 5,

    # Volatilité
    vol_rel_min: float = 1.10,
    atr_ratio_min: float = 1.00,

    # MTF
    use_mtf: bool = True,
    mtf_symbols_same: bool = True,
    tf_m15: str = "M15",
    tf_h1: str = "H1",
    tf_h4: str = "H4",

    # Prudence adaptative
    loss_max: int = 3,
    loss_relief_win: float = 1.5,
    loss_decay_bars: int = 100,
    day_reset: bool = False,

    # Décision cadence
    progress_log_every_pct: int = 5,

    # --- Nouveaux paramètres Risk & Trend ---
    starting_balance: float = 15000.0,
    risk_pct_per_trade: float = 0.004,
    max_daily_dd_pct: float = 0.02,
    max_trades_per_day: int = 6,
    adx_min: float = 20.0,
    use_supertrend: bool = True,
    supertrend_period: int = 10,
    supertrend_mult: float = 3.0,

    # --- News / calendrier ---
    events_csv_path: Optional[str] = None,
    no_trade_before_high_min: int = 15,
    no_trade_after_high_min: int = 10,
) -> List[Trade]:
    logging.basicConfig(level=log_level, format="[BT] %(message)s")
    logger = logging.getLogger("backtest")
    logger.info("SIM_VERSION=2025-09-01e")  # tag de version pour tracer les runs

    # ------------ Cadence de décision via ENV -----------------
    decision_tf_env = (os.getenv("BT_DECISION_TF") or "").upper().strip()
    decision_every_bars_env = os.getenv("BT_DECISION_EVERY_BARS")
    decision_stride: Optional[int] = None
    base_minutes = _tf_to_minutes(timeframe)

    if decision_every_bars_env:
        try:
            n = int(decision_every_bars_env)
            if n >= 1:
                decision_stride = n
                logger.info(f"Cadence décision: toutes les {decision_stride} barres (BT_DECISION_EVERY_BARS).")
        except Exception:
            logger.warning("BT_DECISION_EVERY_BARS invalide, ignoré.")

    if decision_stride is None and decision_tf_env and decision_tf_env not in ("OFF", "NONE"):
        dec_minutes = _tf_to_minutes(decision_tf_env)
        if dec_minutes >= base_minutes and (dec_minutes % base_minutes == 0):
            decision_stride = dec_minutes // base_minutes
            logger.info(f"Cadence décision: TF base {timeframe} ({base_minutes}m) → décision {decision_tf_env} ({dec_minutes}m), stride={decision_stride} barres.")
        else:
            logger.warning(f"BT_DECISION_TF={decision_tf_env} incompatible avec TF base {timeframe} → cadence désactivée.")

    # ------------- Confidence gating & sizing (ENV) -------------
    try:
        min_conf_buy = float(os.getenv("BT_MIN_CONF_BUY", "0.0"))
    except Exception:
        min_conf_buy = 0.0
    try:
        min_conf_sell = float(os.getenv("BT_MIN_CONF_SELL", "0.0"))
    except Exception:
        min_conf_sell = 0.0
    conf_size_floor = max(0.0, min(1.0, float(os.getenv("BT_CONF_SIZE_FLOOR", "0.5"))))
    conf_size_mode = (os.getenv("BT_CONF_SIZE_MODE", "linear") or "linear").lower()  # linear | square

    # ----------------------------------------------------------

    _ = timeframe_to_mt5(timeframe)  # just to validate
    rates = client.fetch_ohlcv(symbol, timeframe, start, end)
    if rates.empty or len(rates) < min_bars:
        print(f"[BT] Pas de données suffisantes pour {symbol} ({len(rates)} barres)")
        return []

    df = rates.copy()
    close = df["close"]
    df["sma20"] = _sma(close, 20)
    df["sma100"] = _sma(close, 100)
    df["sma200"] = _sma(close, 200)
    df["atr"] = _compute_atr_true_range(df, n=atr_len)

    # ADX / DI
    plus_di, minus_di, adx = _adx_components(df, n=14)
    df["plus_di14"] = plus_di
    df["minus_di14"] = minus_di
    df["adx14"] = adx

    # Supertrend
    if use_supertrend:
        st, dirn = _supertrend(df, df["atr"], period=supertrend_period, mult=supertrend_mult)
        df["supertrend"] = st
        df["supertrend_dir"] = dirn  # 1 up, -1 down
    else:
        df["supertrend"] = float("nan")
        df["supertrend_dir"] = 0

    df["slope_sma20"] = _slope(df["sma20"], 5)
    df["slope_sma100"] = _slope(df["sma100"], 10)
    df["slope_sma200"] = _slope(df["sma200"], 20)
    df["dist_sma20"] = close - df["sma20"]
    df["dist_sma100"] = close - df["sma100"]
    df["dist_sma200"] = close - df["sma200"]
    df["rsi14"] = _rsi(close, 14)
    df["vol_rel"] = _vol_rel(close, 20, 100)
    df["atr_ratio"] = df["atr"] / (df["atr"].rolling(100).mean().replace(0, float("inf")))
    df["atr_slope"] = _slope(df["atr"], 10)
    df["ret_1"] = close.pct_change()
    df["ret_12"] = close.pct_change(12)

    idx = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
    hours = idx.hour
    df["hour"] = hours
    df["session"] = pd.cut(
        hours, bins=[-1, 7, 12, 20, 24], labels=["tokyo", "london_open", "newyork", "asia_late"],
    )
    hours = df.index.tz_convert("UTC").hour
    df["is_session_overlap"] = ((hours >= 12) & (hours < 16)).astype(int)

    # --- MTF (optionnel) ---
    mtf = {}
    if use_mtf:
        try:
            for tf_name in [tf_m15, tf_h1, tf_h4]:
                tf_code = timeframe_to_mt5(tf_name)
                r = client.fetch_ohlcv(symbol if mtf_symbols_same else symbol, tf_code, start, end)
                if not r.empty:
                    r[f"sma20_{tf_name.lower()}"] = _sma(r["close"], 20)
                    mtf[tf_name] = r
        except Exception:
            rule_map = {"M15": "15T", "H1": "1H", "H4": "4H"}
            for tf_name, rule in rule_map.items():
                r = df.resample(rule).agg({"open":"first","high":"max","low":"min","close":"last"})
                r[f"sma20_{tf_name.lower()}"] = _sma(r["close"], 20)
                mtf[tf_name] = r

    # --- High-impact news (optionnel) ---
    high_events = []
    if events_csv_path:
        try:
            evts = load_events_csv(events_csv_path)
            high_events = build_high_impact_index(evts)
            logger.info(f"News: {len(high_events)} événements HIGH chargés depuis {events_csv_path}")
        except Exception as e:
            logger.warning(f"News: échec chargement {events_csv_path}: {e}")
    _evt_ptr = 0  # curseur pour chercher le prochain événement (linéaire & rapide)

    # --- Market micro / costs from MT5 ---
    spread_price = 0.0; stop_level_min = 0.0; freeze_level_min = 0.0
    tick_value = None; tick_size = None; min_lot = 0.01; lot_step = 0.01; max_lot = 100.0
    try:
        import MetaTrader5 as mt5
    except Exception:
        from pymt5linux import MetaTrader5 as mt5
        mt5 = mt5(host="localhost", port=8001)

    if use_spread:
        if manual_spread_price is not None and manual_spread_price > 0:
            spread_price = float(manual_spread_price)
        else:
            try:
                si = mt5.symbol_info(symbol)
                if si:
                    if getattr(si, "spread", 0) and si.spread > 0 and getattr(si, "point", 0):
                        spread_price = float(si.spread) * float(si.point)
                    stop_level_min = float(getattr(si, "stops_level", 0) or 0) * float(si.point)
                    freeze_level_min = float(getattr(si, "freeze_level", 0) or 0) * float(si.point)
            except Exception:
                pass

    try:
        si = mt5.symbol_info(symbol)
        if si:
            tick_value = float(getattr(si, "trade_tick_value", 0.0) or 0.0) or None
            tick_size = float(getattr(si, "trade_tick_size", 0.0) or 0.0) or None
            min_lot = float(getattr(si, "volume_min", 0.01) or 0.01)
            lot_step = float(getattr(si, "volume_step", 0.01) or 0.01)
            max_lot = float(getattr(si, "volume_max", 100.0) or 100.0)
    except Exception:
        pass

    # commissions
    commission_per_lot_eff = commission_per_lot if commission_per_lot is not None else commission_per_trade

    logger.info(
        "Paramètres: atr_len=%d, use_spread=%s, spread=%.8f, commission_per_lot=%.4f, "
        "use_sma200_filter=%s, atr_min=%.6f, minSL=%.2f*ATR, minTP=%.2f*ATR, cooldown_any=%d, cooldown_sl=%d, adx_min=%.1f, news_before=%d, news_after=%d",
        atr_len, use_spread, spread_price, commission_per_lot_eff,
        use_sma200_filter, atr_min_threshold, min_sl_atr_mult, min_tp_atr_mult, cooldown_bars_after_any_exit, cooldown_bars_after_sl, adx_min,
        no_trade_before_high_min, no_trade_after_high_min
    )

    trades: List[Trade] = []
    in_position = False
    last_exit_bar_idx: Optional[int] = None
    last_exit_was_sl: bool = False
    loss_streak = 0
    win_streak = 0
    day_trade_count = {}

    loss_pressure: float = 0.0
    last_event_bar_idx: Optional[int] = None
    last_day = None

    # Equity & DD control
    equity = float(starting_balance)
    day_start_equity = equity

    window_days = 7
    window_bars = window_days * _bars_per_day(timeframe)

    logger.info(f"Début simulation {symbol} {timeframe} de {start} à {end} modèle={model}")

    loop_start = max(min_bars, 220)
    loop_end = len(df) - 2
    total_iters = max(1, loop_end - loop_start + 1)
    progress_step = max(1, int(progress_log_every_pct)) if progress_log_every_pct > 0 else 0
    last_logged_pct = -1
    t0 = time.perf_counter()

    for idx_i in range(loop_start, len(df) - 1):
        # progress log
        if progress_step:
            done = idx_i - loop_start + 1
            pct = int(done * 100 / total_iters)
            if pct >= last_logged_pct + progress_step:
                elapsed = time.perf_counter() - t0
                per_iter = elapsed / max(1, done)
                remain = total_iters - done
                eta_s = int(remain * per_iter)
                mm, ss = divmod(eta_s, 60)
                logger.info(f"Progress {pct:3d}% ({done}/{total_iters}) | ETA ~ {mm:02d}:{ss:02d}")
                last_logged_pct = pct

        row = df.iloc[idx_i]
        bar_time = df.index[idx_i + 1]
        bar_time_utc = bar_time.tz_convert("UTC") if bar_time.tzinfo else bar_time.tz_localize("UTC")
        day_key = bar_time.date()
        next_row = df.iloc[idx_i + 1]
        day_trade_count.setdefault(day_key, 0)

        # Reset day start equity quand le jour change
        if (last_day is not None) and (day_key != last_day):
            day_start_equity = equity
        last_day = day_key

        # Decay loss pressure
        if last_event_bar_idx is not None and loss_pressure > 0 and loss_decay_bars > 0:
            bars_since = idx_i - last_event_bar_idx
            if bars_since >= loss_decay_bars:
                steps = bars_since // loss_decay_bars
                if steps > 0:
                    loss_pressure *= (0.5 ** steps)
                    if loss_pressure < 1e-6:
                        loss_pressure = 0.0
                    last_event_bar_idx += steps * loss_decay_bars

        # vérifs indicateurs basiques
        try:
            _ = float(row["sma20"]); _ = float(row["sma100"]); _ = float(row["sma200"])
            atr_now = float(row["atr"]); _ = float(row["close"])
        except Exception:
            continue

        # EXIT management first (si en position, on ne cherche QUE la sortie)
        if in_position:
            last_trade = trades[-1]
            if last_trade.action == "buy":
                ex, rs = _exit_price_long(row, last_trade.sl, last_trade.tp)
                if ex is not None:
                    adj = ex - (spread_price / 2.0) if use_spread else ex
                    pnl_ticks = (adj - last_trade.entry) / (tick_size or 1.0)
                    trade_pnl = pnl_ticks * (tick_value or 1.0) * last_trade.volume_lots - commission_per_lot_eff * last_trade.volume_lots
                    last_trade.exit = adj
                    last_trade.pnl = trade_pnl
                    last_trade.exit_reason = rs
                    equity += trade_pnl
                    last_trade.equity_after = equity
                    in_position = False; last_exit_bar_idx = idx_i; last_event_bar_idx = idx_i
                    last_exit_was_sl = rs.startswith("SL")
                    if trade_pnl < 0: loss_streak += 1; win_streak = 0; loss_pressure = min(loss_pressure + 1.0, float(loss_max))
                    else: win_streak += 1; loss_streak = 0; loss_pressure = max(0.0, loss_pressure - loss_relief_win)
                    logger.info(f"EXIT BUY {rs} @ {adj:.5f} pnl={trade_pnl:.2f} eq={equity:.2f} (loss_pressure={loss_pressure:.2f})")
            else:
                ex, rs = _exit_price_short(row, last_trade.sl, last_trade.tp)
                if ex is not None:
                    adj = ex + (spread_price / 2.0) if use_spread else ex
                    pnl_ticks = (last_trade.entry - adj) / (tick_size or 1.0)
                    trade_pnl = pnl_ticks * (tick_value or 1.0) * last_trade.volume_lots - commission_per_lot_eff * last_trade.volume_lots
                    last_trade.exit = adj
                    last_trade.pnl = trade_pnl
                    last_trade.exit_reason = rs
                    equity += trade_pnl
                    last_trade.equity_after = equity
                    in_position = False; last_exit_bar_idx = idx_i; last_event_bar_idx = idx_i
                    last_exit_was_sl = rs.startswith("SL")
                    if trade_pnl < 0: loss_streak += 1; win_streak = 0; loss_pressure = min(loss_pressure + 1.0, float(loss_max))
                    else: win_streak += 1; loss_streak = 0; loss_pressure = max(0.0, loss_pressure - loss_relief_win)
                    logger.info(f"EXIT SELL {rs} @ {adj:.5f} pnl={trade_pnl:.2f} eq={equity:.2f} (loss_pressure={loss_pressure:.2f})")

            # Toujours continuer à la barre suivante si on est encore en position
            if in_position:
                continue

        # À partir d'ici, on est FLAT (pas de nouvelle entrée si ATR très faible)
        if (not _is_finite(atr_now)) or (atr_now < atr_min_threshold):
            continue

        # cooldown entre trades après une sortie
        if last_exit_bar_idx is not None:
            bars_since_exit = idx_i - last_exit_bar_idx
            needed = cooldown_bars_after_sl if last_exit_was_sl else cooldown_bars_after_any_exit
            if bars_since_exit < needed:
                continue

        # Garde-fou journalier (hard cap si besoin)
        trades_today = day_trade_count.get(day_key, 0)
        if trades_today >= max_trades_per_day:
            continue

        # cadence (si configurée)
        if decision_stride is not None:
            bar_index = (idx_i - loop_start + 1)
            if bar_index % decision_stride != 0:
                continue

        # ----------------- NEWS NO-TRADE WINDOW -----------------
        minutes_to_next_high = None
        minutes_since_last_high = None
        next_high_name = ""
        event_window_active = False
        if high_events:
            m_to, m_since, next_evt, _evt_ptr = nearest_high_events(bar_time_utc.to_pydatetime(), high_events, _evt_ptr)
            minutes_to_next_high = m_to
            minutes_since_last_high = m_since
            if next_evt:
                next_high_name = next_evt.name
            if (m_to is not None and m_to <= int(no_trade_before_high_min)) or (m_since is not None and m_since <= int(no_trade_after_high_min)):
                event_window_active = True
        # --------------------------------------------------------

        # Quality gating
        vol_rel_now = float(df.iloc[idx_i]["vol_rel"]) if _is_finite(df.iloc[idx_i]["vol_rel"]) else float("nan")
        atr_ratio_now = float(df.iloc[idx_i]["atr_ratio"]) if _is_finite(df.iloc[idx_i]["atr_ratio"]) else float("nan")
        if (not _is_finite(vol_rel_now)) or (not _is_finite(atr_ratio_now)):
            continue
        if (vol_rel_now < float(vol_rel_min)) or (atr_ratio_now < float(atr_ratio_min)):
            continue

        # Kill-switch quotidien (pas de nouvelles entrées)
        kill_switch = (equity - day_start_equity) / max(1e-9, day_start_equity) <= -max_daily_dd_pct
        if kill_switch:
            continue

        # Contexte / features
        start_idx = max(0, idx_i - window_bars)
        ctx = df.iloc[start_idx:idx_i]
        if len(ctx) < 50:
            continue

        sma20 = float(row["sma20"]); sma100 = float(row["sma100"]); sma200 = float(row["sma200"])
        adx_now = float(row["adx14"]) if _is_finite(row["adx14"]) else 0.0
        st_dir = int(row["supertrend_dir"]) if use_supertrend else 0

        trend_buy_ok = (sma20 > sma100)
        trend_sell_ok = (sma20 < sma100)
        if use_sma200_filter:
            trend_buy_ok = trend_buy_ok and (sma100 > sma200) and (sma20 > sma200)
            trend_sell_ok = trend_sell_ok and (sma100 < sma200) and (sma20 < sma200)
        if use_supertrend:
            trend_buy_ok = trend_buy_ok and (st_dir == 1)
            trend_sell_ok = trend_sell_ok and (st_dir == -1)
        # ADX filter
        trend_buy_ok = trend_buy_ok and (adx_now >= adx_min)
        trend_sell_ok = trend_sell_ok and (adx_now >= adx_min)

        # PnL récent
        trades_last_n = trades[-10:]
        last_n_pnl = sum(t.pnl for t in trades_last_n if t.exit is not None)

        # ----- HTF bias (M15/H1/H4) -----
        htf_bias = 0.0
        if mtf:
            def _bias_from(mdf):
                # dernier point <= bar_time
                try:
                    j = mdf.index.get_indexer([bar_time], method="pad")[0]
                    if j < 0:
                        return 0.0
                except Exception:
                    j = len(mdf) - 1
                c = float(mdf["close"].iloc[j])
                s = float(mdf.filter(like="sma20").iloc[j, 0])
                return 1.0 if (c > s) else (-1.0 if c < s else 0.0)
            try:
                htf_bias = 0.0
                if "M15" in mtf: htf_bias += 0.5 * _bias_from(mtf["M15"])
                if "H1"  in mtf: htf_bias += 0.8 * _bias_from(mtf["H1"])
                if "H4"  in mtf: htf_bias += 1.2 * _bias_from(mtf["H4"])
            except Exception:
                htf_bias = 0.0

        # Poids de session & pénalité spread
        session_weight = 1.05 if str(df.iloc[idx_i]["session"]) in ("london_open", "newyork") else 0.85
        spread_penalty = (spread_price / max(atr_now, 1e-12)) if spread_price and atr_now else 0.0

        features = {
            "close": float(row["close"]),
            "sma20": sma20, "sma100": sma100, "sma200": sma200,
            "dist_sma20": float(df.iloc[idx_i]["dist_sma20"]),
            "dist_sma100": float(df.iloc[idx_i]["dist_sma100"]),
            "dist_sma200": float(df.iloc[idx_i]["dist_sma200"]),
            "slope_sma20": float(df.iloc[idx_i]["slope_sma20"]),
            "slope_sma100": float(df.iloc[idx_i]["slope_sma100"]),
            "slope_sma200": float(df.iloc[idx_i]["slope_sma200"]),
            "atr": atr_now, "atr_ratio": float(df.iloc[idx_i]["atr_ratio"]), "atr_slope": float(df.iloc[idx_i]["atr_slope"]),
            "vol_rel": float(df.iloc[idx_i]["vol_rel"]), "rsi14": float(df.iloc[idx_i]["rsi14"]),
            "ret_1": float(df.iloc[idx_i]["ret_1"]), "ret_12": float(df.iloc[idx_i]["ret_12"]),
            "hour": int(df.iloc[idx_i]["hour"]), "session": str(df.iloc[idx_i]["session"]),
            "is_session_overlap": int(df.iloc[idx_i]["is_session_overlap"]),
            "spread_price": float(spread_price),
            "stop_level_min": float(stop_level_min),
            "freeze_level_min": float(freeze_level_min),
            # ADX/Supertrend
            "adx14": adx_now, "plusDI": float(df.iloc[idx_i]["plus_di14"]), "minusDI": float(df.iloc[idx_i]["minus_di14"]),
            "supertrend_dir": ("up" if st_dir == 1 else ("down" if st_dir == -1 else "flat")),
            "adx_min": float(adx_min),
            # htf & session
            "htf_bias": float(htf_bias),
            "session_weight": float(session_weight),
            "spread_penalty": float(spread_penalty),
            # comportement
            "loss_streak": int(loss_streak),
            "win_streak": int(win_streak),
            "recent_pnl_sum": float(last_n_pnl),
            "trades_in_day": int(trades_today),
            "cooldown_active": False,
            "loss_pressure": float(round(loss_pressure, 3)),
            "kill_switch_active": False,
            # news
            "event_window_active": bool(event_window_active),
            "minutes_to_next_high": int(minutes_to_next_high) if minutes_to_next_high is not None else None,
            "minutes_since_last_high": int(minutes_since_last_high) if minutes_since_last_high is not None else None,
            "no_trade_before_high_min": int(no_trade_before_high_min),
            "no_trade_after_high_min": int(no_trade_after_high_min),
            "next_high_event": next_high_name or "",
            # filtres impératifs
            "trend_buy_ok": bool(trend_buy_ok),
            "trend_sell_ok": bool(trend_sell_ok),
            # seuils de vol transmis au modèle
            "vol_rel_min": float(vol_rel_min),
            "atr_ratio_min": float(atr_ratio_min),
            # hints SL/TP
            "sl_mult_hint_range": [1.7, 2.3],
            "tp_rr_hint_range": [2.0, 2.8],
            # info cadence
            "decision_stride": int(decision_stride) if decision_stride else 1,
            "decision_tf_env": decision_tf_env or "",
        }

        # Sécurité : si pour une raison X on est revenu en position, on n'entre pas
        if in_position:
            logger.error("Invariant breach: in_position=True avant décision — skip.")
            continue

        decision: StrategyOutput = gpt_decide(features, api_key=api_key, model=model)

        # Confidence gating
        conf = float(getattr(decision, "confidence", 0.5) or 0.5)
        if decision.decision == "buy" and conf < min_conf_buy:
            continue
        if decision.decision == "sell" and conf < min_conf_sell:
            continue

        if decision.decision not in ("buy", "sell"):
            continue
        if decision.decision == "buy" and not trend_buy_ok:
            continue
        if decision.decision == "sell" and not trend_sell_ok:
            continue

        # Entrée au prochain open
        raw_entry = float(next_row["open"])
        sl_pts = max(0.0, float(decision.sl_points))
        tp_pts = max(0.0, float(decision.tp_points))
        sl_min = max(sl_pts, min_sl_atr_mult * atr_now, 3 * (spread_price or 0.0), float(stop_level_min))
        tp_min = max(tp_pts, min_tp_atr_mult * atr_now)

        if decision.decision == "buy":
            entry = raw_entry + (spread_price / 2.0) if use_spread else raw_entry
            sl = entry - sl_min; tp = entry + tp_min
            sl_dist = entry - sl
        else:
            entry = raw_entry - (spread_price / 2.0) if use_spread else raw_entry
            sl = entry + sl_min; tp = entry - tp_min
            sl_dist = sl - entry

        if not (_is_finite(sl) and _is_finite(tp) and _is_finite(entry)):
            continue
        if decision.decision == "buy" and (sl >= entry or tp <= entry):
            continue
        if decision.decision == "sell" and (sl <= entry or tp >= entry):
            continue

        # --- Confidence-based sizing ---
        if conf_size_mode == "square":
            size_factor = max(conf_size_floor, min(1.0, conf * conf))
        else:
            size_factor = max(conf_size_floor, min(1.0, conf))

        # --- Risk-based position sizing ---
        if not tick_value or not tick_size or tick_size <= 0:
            vol_lots = 1.0 * size_factor
        else:
            risk_amt = equity * float(max(0.0, risk_pct_per_trade)) * size_factor
            ticks_to_sl = max(1.0, sl_dist / tick_size)
            cost_per_lot_at_sl = ticks_to_sl * tick_value
            if cost_per_lot_at_sl <= 0:
                continue
            vol_lots = risk_amt / cost_per_lot_at_sl
            # align to broker constraints
            steps = max(1, int(vol_lots / (lot_step or 0.01)))
            vol_lots = steps * (lot_step or 0.01)
            vol_lots = max(min_lot, min(max_lot, vol_lots))
            if vol_lots < (min_lot - 1e-9):
                continue

        # Double check invariant juste avant append
        if in_position:
            logger.error("Invariant breach: tentative d'entrée alors qu'on est déjà en position — skip.")
            continue

        # Enrichir la reason avec la confiance & sizing
        reason_txt = decision.reason or ""
        reason_txt = (reason_txt + f"; conf={conf:.2f}; size×={size_factor:.2f}").strip()

        # Enregistrer le trade (equity_before)
        trade = Trade(
            time=df.index[idx_i + 1].to_pydatetime(),
            symbol=symbol,
            action=decision.decision,
            entry=entry, sl=sl, tp=tp, exit=None, reason=reason_txt,
            volume_lots=float(round(vol_lots, 4)),
            equity_before=float(round(equity, 2)),
            equity_after=float(round(equity, 2)),
        )
        trades.append(trade)
        in_position = True
        day_trade_count[day_key] += 1

        logger.info(
            f"ENTER {decision.decision.upper()} @ {entry:.5f} sl={sl:.5f} tp={tp:.5f} vol={vol_lots:.2f} "
            f"(ATR={atr_now:.6f}, loss_pressure={loss_pressure:.2f}, conf={conf:.2f}, size×={size_factor:.2f}, stride={decision_stride or 1}) "
            f"reason={reason_txt[:140]} | eq={equity:.2f}"
        )

    # ---------- EOD: mass-close de TOUT trade encore ouvert ----------
    if len(trades) > 0:
        eod_price = float(df.iloc[-1]["close"])
        for t in reversed(trades):
            if t.exit is not None:
                break
            if t.action == "buy":
                adj = eod_price - (spread_price / 2.0) if use_spread else eod_price
                pnl_ticks = (adj - t.entry) / (tick_size or 1.0)
            else:
                adj = eod_price + (spread_price / 2.0) if use_spread else eod_price
                pnl_ticks = (t.entry - adj) / (tick_size or 1.0)
            trade_pnl = pnl_ticks * (tick_value or 1.0) * (t.volume_lots or 1.0) - commission_per_lot_eff * (t.volume_lots or 1.0)
            t.exit = adj
            t.pnl = trade_pnl
            t.exit_reason = "EOD(mass-close)"
            equity += trade_pnl
            t.equity_after = equity
            logging.info(f"EXIT {t.action.upper()} EOD(mass-close) @ {adj:.5f} pnl={trade_pnl:.2f} eq={equity:.2f}")

    logging.info(f"Fin simulation {symbol}: trades={len(trades)} | equity_final={equity:.2f}")
    return trades