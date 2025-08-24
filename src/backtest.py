from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Tuple
import logging
import math
import pandas as pd

from .mt5_client import MT5Client
from .config import timeframe_to_mt5
from .gpt_strategy import gpt_decide, StrategyOutput

import time  # <<< pour ETA
import os  # <<< pour ENV

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
    pnl: float = 0.0
    reason: str = ""
    exit_reason: str = ""


# ---------- Indicateurs utilitaires ----------
def _compute_atr_true_range(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"]; low = df["low"]; prev_close = df["close"].shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def _slope(series: pd.Series, lookback: int = 5) -> pd.Series:
    # pente simple: différence moyenne par barre (approx)
    return (series - series.shift(lookback)) / max(1, lookback)

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0); down = -delta.clip(upper=0.0)
    ma_up = up.rolling(n).mean(); ma_down = down.rolling(n).mean()
    rs = ma_up / (ma_down.replace(0, float("inf")))
    return 100 - (100 / (1 + rs))

def _vol_rel(close: pd.Series, n_short: int = 20, n_long: int = 100) -> pd.Series:
    # volatilité relative: ratio des stds
    short = close.pct_change().rolling(n_short).std()
    long = close.pct_change().rolling(n_long).std()
    return (short / (long.replace(0, float("inf"))))

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
    return x is not None and not (math.isnan(x) or math.isinf(x))

def _bars_per_day(timeframe: str) -> int:
    tf = timeframe.upper()
    if   tf == "M1":  return 1440
    elif tf == "M5":  return 288
    elif tf == "M15": return 96
    elif tf == "M30": return 48
    elif tf == "H1":  return 24
    elif tf == "H4":  return 6
    elif tf == "D1":  return 1
    # fallback
    return 288  # M5 par défaut

def _tf_to_minutes(tf: str) -> int:
    tf = tf.upper().strip()
    if tf.startswith("M"):
        return int(tf[1:])
    if tf == "H1": return 60
    if tf == "H2": return 120
    if tf == "H3": return 180
    if tf == "H4": return 240
    if tf == "D1": return 1440
    # fallback raisonnable
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
    commission_per_trade: float = 0.0,
    log_level: int = logging.INFO,

    # Robustesse
    use_sma200_filter: bool = True,
    atr_min_threshold: float = 0.00025,
    min_sl_atr_mult: float = 1.5,
    min_tp_atr_mult: float = 2.5,
    cooldown_bars_after_any_exit: int = 2,
    cooldown_bars_after_sl: int = 5,
    
    # --- Filtres de volatilité ---
    vol_rel_min: float = 1.10,     # ratio std courte/longue (>=1.10 ~ marché actif)
    atr_ratio_min: float = 1.00,   # ATR / ATR_moy_100 (>=1.00 ~ ATR non anémié)

    # Multi-timeframe
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

    # Progress logs
    progress_log_every_pct: int = 5,
) -> List[Trade]:
    logging.basicConfig(level=log_level, format="[BT] %(message)s")
    logger = logging.getLogger("backtest")

    # ------------ Nouveau: cadence de décision via ENV -----------------
    # BT_DECISION_TF = "M15" par ex., ou "OFF" / non défini pour désactiver
    # BT_DECISION_EVERY_BARS = "3" pour ne décider que toutes les N barres
    decision_tf_env = (os.getenv("BT_DECISION_TF") or "").upper().strip()
    decision_every_bars_env = os.getenv("BT_DECISION_EVERY_BARS")

    decision_stride: Optional[int] = None  # nombre de barres entre 2 décisions
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

    # -------------------------------------------------------------------

    code = timeframe_to_mt5(timeframe)
    rates = client.fetch_ohlcv(symbol, code, start, end)
    if rates.empty or len(rates) < min_bars:
        print(f"[BT] Pas de données suffisantes pour {symbol} ({len(rates)} barres)")
        return []

    df = rates.copy()
    close = df["close"]
    df["sma20"] = _sma(close, 20)
    df["sma100"] = _sma(close, 100)
    df["sma200"] = _sma(close, 200)
    df["atr"] = _compute_atr_true_range(df, n=atr_len)

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

    # --- MTF (pooled comme avant) ---
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

    spread_price = 0.0; stop_level_min = 0.0; freeze_level_min = 0.0
    if use_spread:
        if manual_spread_price is not None and manual_spread_price > 0:
            spread_price = float(manual_spread_price)
        else:
            try:
                import MetaTrader5 as mt5
                si = mt5.symbol_info(symbol)
                if si and si.point:
                    if getattr(si, "spread", 0) and si.spread > 0:
                        spread_price = float(si.spread) * float(si.point)
                    stop_level_min = float(getattr(si, "stops_level", 0) or 0) * float(si.point)
                    freeze_level_min = float(getattr(si, "freeze_level", 0) or 0) * float(si.point)
            except Exception:
                pass

    logger.info(
        "Paramètres: atr_len=%d, use_spread=%s, spread=%.8f, commission=%.8f, "
        "use_sma200_filter=%s, atr_min=%.6f, minSL=%.2f*ATR, minTP=%.2f*ATR, cooldown_any=%d, cooldown_sl=%d",
        atr_len, use_spread, spread_price, commission_per_trade,
        use_sma200_filter, atr_min_threshold, min_sl_atr_mult, min_tp_atr_mult,
        cooldown_bars_after_any_exit, cooldown_bars_after_sl
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
        day_key = df.index[idx_i + 1].date()
        next_row = df.iloc[idx_i + 1]
        day_trade_count.setdefault(day_key, 0)

        # resets prudence/decay (inchangé)
        cur_day = day_key
        if last_day is not None and cur_day != last_day and day_reset:
            loss_streak = 0
            win_streak = 0
            loss_pressure *= 0.5
            if loss_pressure < 1e-6:
                loss_pressure = 0.0
        last_day = cur_day
        if last_event_bar_idx is not None and loss_pressure > 0 and loss_decay_bars > 0:
            bars_since = idx_i - last_event_bar_idx
            if bars_since >= loss_decay_bars:
                steps = bars_since // loss_decay_bars
                if steps > 0:
                    loss_pressure *= (0.5 ** steps)
                    if loss_pressure < 1e-6:
                        loss_pressure = 0.0
                    last_event_bar_idx += steps * loss_decay_bars

        # vérifs indicateurs
        try:
            _ = float(row["sma20"]); _ = float(row["sma100"]); _ = float(row["sma200"])
            atr_now = float(row["atr"]); _ = float(row["close"])
        except Exception:
            continue
        if not _is_finite(atr_now) or atr_now < atr_min_threshold:
            # même si ATR faible on continue de gérer les EXIT
            if in_position:
                last_trade = trades[-1]
                if last_trade.action == "buy":
                    ex, rs = _exit_price_long(row, last_trade.sl, last_trade.tp)
                    if ex is not None:
                        adj = ex - (spread_price / 2.0) if use_spread else ex
                        last_trade.exit = adj
                        last_trade.pnl = (adj - last_trade.entry) - commission_per_trade
                        last_trade.exit_reason = rs
                        in_position = False; last_exit_bar_idx = idx_i; last_event_bar_idx = idx_i
                        if last_trade.pnl < 0: loss_streak += 1; win_streak = 0; loss_pressure = min(loss_pressure + 1.0, float(loss_max))
                        else: win_streak += 1; loss_streak = 0; loss_pressure = max(0.0, loss_pressure - loss_relief_win)
                else:
                    ex, rs = _exit_price_short(row, last_trade.sl, last_trade.tp)
                    if ex is not None:
                        adj = ex + (spread_price / 2.0) if use_spread else ex
                        last_trade.exit = adj
                        last_trade.pnl = (last_trade.entry - adj) - commission_per_trade
                        last_trade.exit_reason = rs
                        in_position = False; last_exit_bar_idx = idx_i; last_event_bar_idx = idx_i
                        if last_trade.pnl < 0: loss_streak += 1; win_streak = 0; loss_pressure = min(loss_pressure + 1.0, float(loss_max))
                        else: win_streak += 1; loss_streak = 0; loss_pressure = max(0.0, loss_pressure - loss_relief_win)
            continue

        # sorties si en position
        if in_position:
            last_trade = trades[-1]
            if last_trade.action == "buy":
                ex, rs = _exit_price_long(row, last_trade.sl, last_trade.tp)
                if ex is not None:
                    adj = ex - (spread_price / 2.0) if use_spread else ex
                    last_trade.exit = adj
                    last_trade.pnl = (adj - last_trade.entry) - commission_per_trade
                    last_trade.exit_reason = rs
                    in_position = False; last_exit_bar_idx = idx_i; last_event_bar_idx = idx_i
                    last_exit_was_sl = rs.startswith("SL")
                    if last_trade.pnl < 0: loss_streak += 1; win_streak = 0; loss_pressure = min(loss_pressure + 1.0, float(loss_max))
                    else: win_streak += 1; loss_streak = 0; loss_pressure = max(0.0, loss_pressure - loss_relief_win)
                    logger.info(f"EXIT BUY {rs} @ {adj:.5f} pnl={last_trade.pnl:.5f} (loss_pressure={loss_pressure:.2f})")
            else:
                ex, rs = _exit_price_short(row, last_trade.sl, last_trade.tp)
                if ex is not None:
                    adj = ex + (spread_price / 2.0) if use_spread else ex
                    last_trade.exit = adj
                    last_trade.pnl = (last_trade.entry - last_trade.exit) - commission_per_trade
                    last_trade.exit_reason = rs
                    in_position = False; last_exit_bar_idx = idx_i; last_event_bar_idx = idx_i
                    last_exit_was_sl = rs.startswith("SL")
                    if last_trade.pnl < 0: loss_streak += 1; win_streak = 0; loss_pressure = min(loss_pressure + 1.0, float(loss_max))
                    else: win_streak += 1; loss_streak = 0; loss_pressure = max(0.0, loss_pressure - loss_relief_win)
                    logger.info(f"EXIT SELL {rs} @ {adj:.5f} pnl={last_trade.pnl:.5f} (loss_pressure={loss_pressure:.2f})")

            if in_position:
                continue  # pas de nouveau trade si encore en position

        # cooldown
        if last_exit_bar_idx is not None:
            bars_since_exit = idx_i - last_exit_bar_idx
            needed = cooldown_bars_after_sl if last_exit_was_sl else cooldown_bars_after_any_exit
            if bars_since_exit < needed:
                continue

        # -------- Gating de décision par cadence (ENV) ----------
        if decision_stride is not None:
            # On décide uniquement à la FIN d’un bloc (ex: chaque 3 barres si M15 sur M5)
            bar_index = (idx_i - loop_start + 1)
            if bar_index % decision_stride != 0:
                continue
        # --------------------------------------------------------

        # Contexte / features
        start_idx = max(0, idx_i - window_bars)
        ctx = df.iloc[start_idx:idx_i]
        if len(ctx) < 50:
            continue

        sma20 = float(row["sma20"]); sma100 = float(row["sma100"]); sma200 = float(row["sma200"])
        atr_now = float(row["atr"])

        trend_buy_ok = sma20 > sma100
        trend_sell_ok = sma20 < sma100
        if use_sma200_filter:
            trend_buy_ok = trend_buy_ok and (sma100 > sma200) and (sma20 > sma200)
            trend_sell_ok = trend_sell_ok and (sma100 < sma200) and (sma20 < sma200)

        mtf_features = {}
        try:
            if use_mtf and mtf:
                for tf_name, r in mtf.items():
                    r_slice = r.loc[:row.name]
                    if not r_slice.empty:
                        last_row = r_slice.iloc[-1]
                        mtf_features[f"sma20_{tf_name.lower()}"] = float(last_row.get(f"sma20_{tf_name.lower()}", float("nan")))
                        mtf_features[f"close_{tf_name.lower()}"] = float(last_row.get("close", float("nan")))
        except Exception:
            pass

        trades_last_n = trades[-10:]
        last_n_pnl = sum(t.pnl for t in trades_last_n if t.exit is not None)
        trades_today = day_trade_count.get(day_key, 0)

        # ---------- Filtre VOLATILITÉ (gating avant décision) ----------
        # try:
        #     vol_rel_now = float(df.iloc[idx_i]["vol_rel"])
        #     atr_ratio_now = float(df.iloc[idx_i]["atr_ratio"])
        # except Exception:
        #     vol_rel_now, atr_ratio_now = float("nan"), float("nan")
        # # si volatilité insuffisante → on saute cette barre
        # if not (_is_finite(vol_rel_now) and _is_finite(atr_ratio_now)):
        #     continue
        # if vol_rel_now < vol_rel_min or atr_ratio_now < atr_ratio_min:
        #     continue
        # ---------------------------------------------------------------

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

            # comportement
            "loss_streak": int(loss_streak),
            "win_streak": int(win_streak),
            "recent_pnl_sum": float(last_n_pnl),
            "trades_in_day": int(trades_today),
            "cooldown_active": False,
            "loss_pressure": float(round(loss_pressure, 3)),

            # filtres impératifs
            "trend_buy_ok": bool(trend_buy_ok),
            "trend_sell_ok": bool(trend_sell_ok),

            # hints
            "atr_min_hint": float(max(0.00025, atr_min_threshold)),

            # info cadence (optionnel mais utile pour le prompt/debug)
            "decision_stride": int(decision_stride) if decision_stride else 1,
            "decision_tf_env": decision_tf_env or "",
        }
        features.update(mtf_features)

        decision: StrategyOutput = gpt_decide(features, api_key=api_key, model=model)
        if decision.decision not in ("buy", "sell"):
            continue
        if decision.decision == "buy" and not trend_buy_ok:
            continue
        if decision.decision == "sell" and not trend_sell_ok:
            continue

        # Entrée au prochain open (du premier M5 du bloc suivant si stride > 1)
        raw_entry = float(next_row["open"])
        sl_pts = max(0.0, float(decision.sl_points))
        tp_pts = max(0.0, float(decision.tp_points))
        sl_min = max(sl_pts, min_sl_atr_mult * atr_now)
        tp_min = max(tp_pts, min_tp_atr_mult * atr_now)

        if decision.decision == "buy":
            entry = raw_entry + (spread_price / 2.0) if use_spread else raw_entry
            sl = entry - sl_min; tp = entry + tp_min
        else:
            entry = raw_entry - (spread_price / 2.0) if use_spread else raw_entry
            sl = entry + sl_min; tp = entry - tp_min

        if not (_is_finite(sl) and _is_finite(tp) and _is_finite(entry)):
            continue
        if decision.decision == "buy" and (sl >= entry or tp <= entry):
            continue
        if decision.decision == "sell" and (sl <= entry or tp >= entry):
            continue

        trades.append(Trade(
            time=df.index[idx_i + 1].to_pydatetime(),
            symbol=symbol,
            action=decision.decision,
            entry=entry, sl=sl, tp=tp, exit=None, reason=decision.reason,
        ))
        in_position = True
        day_trade_count[day_key] += 1

        logging.info(
            f"ENTER {decision.decision.upper()} @ {entry:.5f} sl={sl:.5f} tp={tp:.5f} "
            f"(raw_open={raw_entry:.5f}, ATR={atr_now:.6f}, loss_pressure={loss_pressure:.2f}, stride={decision_stride or 1}) "
            f"reason={decision.reason[:100]}"
        )

    if in_position and len(trades) > 0:
        last = df.iloc[-1]
        last_trade = trades[-1]
        eod_exit = float(last["close"])
        if last_trade.action == "buy":
            adj_exit = eod_exit - (spread_price / 2.0) if use_spread else eod_exit
            last_trade.exit = adj_exit
            last_trade.pnl = (last_trade.exit - last_trade.entry) - commission_per_trade
            last_trade.exit_reason = "EOD"
            logging.info(f"EXIT BUY EOD @ {adj_exit:.5f} pnl={last_trade.pnl:.5f}")
        else:
            adj_exit = eod_exit + (spread_price / 2.0) if use_spread else eod_exit
            last_trade.exit = adj_exit
            last_trade.pnl = (last_trade.entry - last_trade.exit) - commission_per_trade
            last_trade.exit_reason = "EOD"
            logging.info(f"EXIT SELL EOD @ {adj_exit:.5f} pnl={last_trade.pnl:.5f}")

    logging.info(f"Fin simulation {symbol}: trades={len(trades)}")
    return trades