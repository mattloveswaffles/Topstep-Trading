import argparse
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Dollars per point per contract for NQ futures (approximate)
POINT_VALUE = 20.0

# Default contracts per trade for the strategy (must be <= MAX_CONTRACTS)
DEFAULT_CONTRACTS = 1


def infer_bar_minutes(df: pd.DataFrame, timestamp_col: str = "datetime") -> int:
    """Infer bar size in minutes from the first two rows."""
    if len(df) < 2:
        raise ValueError("Not enough rows to infer bar size.")
    t0 = df[timestamp_col].iloc[0]
    t1 = df[timestamp_col].iloc[1]
    delta = (t1 - t0).total_seconds() / 60.0
    if delta <= 0:
        raise ValueError("Non-positive bar interval detected.")
    return int(round(delta))


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI with Wilder smoothing (avg gain / avg loss)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def _vwap_per_day(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    trading_day: pd.Series,
) -> pd.Series:
    """VWAP reset each trading day. Uses typical price (H+L+C)/3. Requires volume."""
    typical = (high + low + close) / 3.0
    pv = typical * volume.replace(0, np.nan)
    cum_pv = pv.groupby(trading_day).cumsum()
    cum_vol = volume.groupby(trading_day).cumsum().replace(0, np.nan)
    return cum_pv / cum_vol


def _session_mask(
    dt: pd.Series,
    start_minutes: int,
    end_minutes: int,
    session_tz: Optional[str] = None,
) -> pd.Series:
    """
    True where time (in minutes from midnight) is in [start_minutes, end_minutes].
    If session_tz is set (e.g. 'America/New_York') and dt is timezone-aware, times are
    converted to that zone first so start/end are in local session (e.g. RTH 9:30-15:45 ET).
    """
    if session_tz and hasattr(dt.dt, "tz") and dt.dt.tz is not None:
        try:
            dt = dt.dt.tz_convert(session_tz)
        except Exception:
            pass
    minutes = dt.dt.hour * 60 + dt.dt.minute
    return (minutes >= start_minutes) & (minutes <= end_minutes)


def _vwap_std_per_day(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    trading_day: pd.Series,
    vwap: pd.Series,
) -> pd.Series:
    """Intraday running std of typical price around VWAP, reset per day. For VWAP bands."""
    typical = (high + low + close) / 3.0
    diff_sq = (typical - vwap).astype(float) ** 2
    # Running weighted variance: sum(w * x^2) / sum(w) - (sum(w*x)/sum(w))^2 with w=volume
    cum_pv = (diff_sq * volume.replace(0, np.nan)).groupby(trading_day).cumsum()
    cum_vol = volume.groupby(trading_day).cumsum().replace(0, np.nan)
    var_approx = cum_pv / cum_vol  # simplified; full Welford would be more accurate
    return np.sqrt(var_approx.replace(0, np.nan))


def _micro_momentum_z(
    typical_price: pd.Series,
    fast: int = 3,
    slow: int = 7,
    z_period: int = 20,
) -> Tuple[pd.Series, pd.Series]:
    """
    Micro Momentum = SMA_fast(P) - SMA_slow(P). Z-score over last z_period bars.
    Returns (micro_momentum, z_score).
    """
    sma_fast = typical_price.rolling(fast, min_periods=fast).mean()
    sma_slow = typical_price.rolling(slow, min_periods=slow).mean()
    micro_mom = sma_fast - sma_slow
    roll_mean = micro_mom.rolling(z_period, min_periods=1).mean()
    roll_std = micro_mom.rolling(z_period, min_periods=1).std().replace(0, np.nan)
    z = (micro_mom - roll_mean) / roll_std
    return micro_mom, z


def _higher_timeframe_bias(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    close_col: str,
    higher_tf_minutes: int = 15,
    ema_span: int = 20,
) -> pd.DataFrame:
    """
    Build a simple higher-timeframe trend filter and align it back to base bars.
    Returns columns:
      - _htf_close
      - _htf_ema
      - _htf_ema_prev
    """
    htf = (
        df[[timestamp_col, close_col]]
        .set_index(timestamp_col)
        .resample(f"{higher_tf_minutes}min")
        .agg({close_col: "last"})
        .dropna()
        .reset_index()
    )
    htf["_htf_close"] = htf[close_col].astype(float)
    htf["_htf_ema"] = htf["_htf_close"].ewm(span=ema_span, adjust=False).mean()
    htf["_htf_ema_prev"] = htf["_htf_ema"].shift(1)
    merged = pd.merge_asof(
        df.sort_values(timestamp_col),
        htf[[timestamp_col, "_htf_close", "_htf_ema", "_htf_ema_prev"]].sort_values(timestamp_col),
        on=timestamp_col,
        direction="backward",
    )
    return merged


def _session_day_features(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    open_col: str,
    high_col: str,
    low_col: str,
    close_col: str,
    session_tz: Optional[str],
    session_start_min: int,
    session_end_min: int,
    first_window_minutes: int = 15,
) -> pd.DataFrame:
    """
    Build day-level features in session-local time:
      - _session_day
      - _session_open
      - _prev_day_close
      - _first_window_move
    """
    work = df.copy()
    local_dt = pd.to_datetime(work[timestamp_col])
    if session_tz and hasattr(local_dt.dt, "tz") and local_dt.dt.tz is not None:
        try:
            local_dt = local_dt.dt.tz_convert(session_tz)
        except Exception:
            pass
    work["_local_dt"] = local_dt
    work["_session_day"] = work["_local_dt"].dt.date.astype(str)
    work["_minutes"] = work["_local_dt"].dt.hour * 60 + work["_local_dt"].dt.minute

    day_close = (
        work.groupby("_session_day")[close_col]
        .last()
        .rename("_day_close")
        .to_frame()
    )
    day_close["_prev_day_close"] = day_close["_day_close"].shift(1)

    session_df = work[
        (work["_minutes"] >= session_start_min) & (work["_minutes"] <= session_end_min)
    ].copy()
    first_window_end = session_start_min + first_window_minutes
    first_window = session_df[session_df["_minutes"] <= first_window_end].copy()
    first_window_feats = first_window.groupby("_session_day").agg(
        _session_open=(open_col, "first"),
        _first_window_close=(close_col, "last"),
    )
    first_window_feats["_first_window_move"] = (
        first_window_feats["_first_window_close"] - first_window_feats["_session_open"]
    )
    day_feats = first_window_feats.join(day_close[["_prev_day_close"]], how="left")
    out = work.merge(day_feats, left_on="_session_day", right_index=True, how="left")
    return out


def run_vwap_micro_momentum_topstep(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "datetime",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: Optional[str] = None,
    contracts: int = 1,
    point_value: float = POINT_VALUE,
    session_start_min: int = 9 * 60 + 30,
    session_end_min: int = 15 * 60 + 45,
    session_tz: Optional[str] = None,
    max_trades_per_day: Optional[int] = 5,
    atr_period: int = 14,
    atr_target_k: float = 1.0,
    atr_stop_k: float = 0.35,
    micro_fast: int = 3,
    micro_slow: int = 7,
    micro_z_period: int = 20,
    z_cross_min: float = 0.1,
    z_exit_threshold: float = 0.25,
    min_vwap_distance: float = 0.5,
    breakeven_atr_k: float = 0.4,
    long_only: bool = True,
    use_vwap_bands: bool = False,
    high_prob_only: bool = False,
    z_prev_extreme: float = 0.7,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    VWAP + Micro Momentum strategy per Topstep-oriented spec.

    - VWAP as intraday trend filter; trade only in direction of VWAP slope.
    - Micro Momentum = SMA(fast, typical_price) - SMA(slow, typical_price); z-score normalized.
    - Long: VWAP_t > VWAP_{t-1}, price above VWAP, z crosses from negative to positive.
    - Short (if not long_only): VWAP_t < VWAP_{t-1}, price below VWAP, z crosses positive to negative.
    - Stop/target: ATR-based (k_stop * ATR14, k_target * ATR14).
    - Momentum exit: close when z crosses zero against position.
    - Optional VWAP ±1σ/2σ bands for confluence.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col)
    df["trading_day"] = df[timestamp_col].dt.date.astype(str)

    close = df[close_col].astype(float)
    high = df[high_col].astype(float)
    low = df[low_col].astype(float)
    vol = (
        df[volume_col].astype(float)
        if volume_col and volume_col in df.columns
        else pd.Series(1.0, index=df.index)
    )
    typical = (high + low + close) / 3.0

    df["_vwap"] = _vwap_per_day(high, low, close, vol, df["trading_day"])
    df["_vwap_prev"] = df["_vwap"].groupby(df["trading_day"]).shift(1)
    df["_session"] = _session_mask(
        df[timestamp_col], session_start_min, session_end_min, session_tz=session_tz
    )
    df["_atr"] = _atr(high, low, close, atr_period)
    micro_mom, z = _micro_momentum_z(typical, micro_fast, micro_slow, micro_z_period)
    df["_micro_mom"] = micro_mom
    df["_z"] = z
    df["_z_prev"] = z.shift(1)
    if use_vwap_bands:
        df["_vwap_std"] = _vwap_std_per_day(
            high, low, close, vol, df["trading_day"], df["_vwap"]
        )

    positions: List[int] = []
    cum_pnls: List[float] = []
    cum_pnl = 0.0
    trades: List[Dict[str, Any]] = []
    position = 0  # 1 long, -1 short, 0 flat
    entry_price = None
    stop_level = None
    target_level = None
    entry_atr = None
    trades_today = 0
    last_day: Optional[str] = None
    n = len(df)

    for k in range(n):
        row = df.iloc[k]
        ts = row[timestamp_col]
        c = float(row[close_col])
        h = float(row[high_col])
        l_ = float(row[low_col])
        day = row["trading_day"]
        in_session = bool(row["_session"])
        if day != last_day:
            trades_today = 0
            last_day = day

        if not in_session:
            if position != 0 and entry_price is not None:
                exit_price = c
                pnl = (exit_price - entry_price) * point_value * contracts * position
                cum_pnl += pnl
                trades.append({
                    "timestamp": ts, "instrument": "NQ", "entry_price": entry_price,
                    "exit_price": exit_price, "contracts": abs(position), "pnl": pnl,
                    "cumulative_pnl": cum_pnl, "trading_day": day,
                })
                position = 0
                entry_price = stop_level = target_level = entry_atr = None
            positions.append(position)
            cum_pnls.append(cum_pnl)
            continue

        vwap = row["_vwap"]
        vwap_prev = row["_vwap_prev"]
        atr = row["_atr"]
        z_val = row["_z"]
        z_prev = row["_z_prev"]
        effective_max_trades = min(max_trades_per_day or 99, 3) if high_prob_only else max_trades_per_day
        can_trade = effective_max_trades is None or trades_today < effective_max_trades
        vwap_slope_up = not pd.isna(vwap) and not pd.isna(vwap_prev) and vwap > vwap_prev
        vwap_slope_down = not pd.isna(vwap) and not pd.isna(vwap_prev) and vwap < vwap_prev

        # Entry: stricter z-cross (clear momentum flip) and min distance above VWAP
        z_cross_above = (
            not pd.isna(z_val) and not pd.isna(z_prev)
            and z_val > z_cross_min and z_prev < -z_cross_min
        )
        z_cross_below = (
            not pd.isna(z_val) and not pd.isna(z_prev)
            and z_val < -z_cross_min and z_prev > z_cross_min
        )
        # High-prob only: require reversal from extended move (|z_prev| >= z_prev_extreme)
        if high_prob_only:
            z_cross_above = z_cross_above and abs(float(z_prev)) >= z_prev_extreme
            z_cross_below = z_cross_below and abs(float(z_prev)) >= z_prev_extreme
        price_above_vwap = c >= float(vwap) + min_vwap_distance if not pd.isna(vwap) else False
        price_below_vwap = c <= float(vwap) - min_vwap_distance if not pd.isna(vwap) else False
        long_signal = (
            can_trade
            and position == 0
            and vwap_slope_up
            and price_above_vwap
            and z_cross_above
        )
        short_signal = (
            not long_only
            and can_trade
            and position == 0
            and vwap_slope_down
            and price_below_vwap
            and z_cross_below
        )

        if long_signal and position == 0:
            position = 1
            entry_price = c
            entry_atr = float(atr) if atr is not None and not pd.isna(atr) and atr > 0 else 10.0
            stop_level = entry_price - atr_stop_k * entry_atr
            target_level = entry_price + atr_target_k * entry_atr
        elif short_signal and position == 0:
            position = -1
            entry_price = c
            entry_atr = float(atr) if atr is not None and not pd.isna(atr) and atr > 0 else 10.0
            stop_level = entry_price + atr_stop_k * entry_atr
            target_level = entry_price - atr_target_k * entry_atr

        # Trail to breakeven: after profit >= breakeven_atr_k * ATR, move stop to entry
        if position != 0 and entry_price is not None and entry_atr is not None and entry_atr > 0:
            if position == 1 and h >= entry_price + breakeven_atr_k * entry_atr:
                stop_level = max(stop_level, entry_price)
            elif position == -1 and l_ <= entry_price - breakeven_atr_k * entry_atr:
                stop_level = min(stop_level, entry_price)

        # Exits: stop, target, momentum reversal (softer threshold), EOD
        exit_now = False
        exit_price = None
        if position != 0 and entry_price is not None:
            if position == 1:  # long
                if l_ <= stop_level:
                    exit_now = True
                    exit_price = stop_level
                elif h >= target_level:
                    exit_now = True
                    exit_price = target_level
                elif not pd.isna(z_val) and not pd.isna(z_prev) and z_val < -z_exit_threshold and z_prev >= -z_exit_threshold:
                    exit_now = True
                    exit_price = c
            else:  # short
                if h >= stop_level:
                    exit_now = True
                    exit_price = stop_level
                elif l_ <= target_level:
                    exit_now = True
                    exit_price = target_level
                elif not pd.isna(z_val) and not pd.isna(z_prev) and z_val > z_exit_threshold and z_prev <= z_exit_threshold:
                    exit_now = True
                    exit_price = c
            if not exit_now:
                last_bar = (k + 1 >= n) or (df.iloc[k + 1]["trading_day"] != day)
                if last_bar:
                    exit_now = True
                    exit_price = c

        if exit_now and exit_price is not None:
            pnl = (exit_price - entry_price) * point_value * contracts * position
            cum_pnl += pnl
            trades.append({
                "timestamp": ts, "instrument": "NQ", "entry_price": entry_price,
                "exit_price": exit_price, "contracts": abs(position), "pnl": pnl,
                "cumulative_pnl": cum_pnl, "trading_day": day,
            })
            trades_today += 1
            position = 0
            entry_price = stop_level = target_level = entry_atr = None

        positions.append(position)
        cum_pnls.append(cum_pnl)

    drop_cols = [
        "_vwap", "_vwap_prev", "_session", "_atr", "_micro_mom", "_z", "_z_prev"
    ]
    if use_vwap_bands and "_vwap_std" in df.columns:
        drop_cols.append("_vwap_std")
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")
    df["position"] = positions
    df["cum_pnl"] = cum_pnls
    trades_df = pd.DataFrame(trades)
    return df, trades_df, cum_pnl


def run_topstep_nq(
    df: pd.DataFrame,
    *,
    strategy: str = "vwap_micro",
    timestamp_col: str = "datetime",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: Optional[str] = None,
    stop_pts: float = 5.0,
    target_pts: float = 20.0,
    adaptive_target: bool = False,
    low_vol_target_pts: float = 16.0,
    high_vol_target_pts: float = 20.0,
    target_vol_window: int = 50,
    contracts: int = 1,
    point_value: float = POINT_VALUE,
    session_start_min: int = 9 * 60 + 30,
    session_end_min: int = 15 * 60 + 45,
    session_tz: Optional[str] = None,
    max_trades_per_day: Optional[int] = None,
    ema_fast: int = 20,
    ema_slow: int = 50,
    rsi_period: int = 14,
    min_vwap_distance_pts: float = 1.0,
    vwap_micro_mean_reversion: bool = False,
    high_prob_only: bool = False,
    min_dip_pts: float = 2.5,
    min_down_bar_pts: float = 1.0,
    rsi_oversold_max: Optional[float] = 38.0,
    a_plus_only: bool = False,
    a_plus_min_dip_pts: float = 4.0,
    a_plus_min_down_bar_pts: float = 1.5,
    a_plus_rsi_oversold_max: Optional[float] = 35.0,
    hybrid_sizing: bool = False,
    base_contracts: int = 1,
    partial_scale_out: bool = False,
    partial_exit_pts: Optional[float] = None,
    partial_exit_contracts: int = 1,
    breakeven_after_partial: bool = False,
    rolling_stop_distance_pts: Optional[float] = None,
    confirmation_entry: bool = False,
    confirmation_reclaim_prior_high: bool = False,
    higher_tf_bias: bool = False,
    higher_tf_minutes: int = 15,
    higher_tf_ema_span: int = 20,
    day_bias_filter: bool = False,
    day_bias_first_window_minutes: int = 15,
    day_bias_min_first_window_move_pts: float = 0.0,
    day_bias_min_gap_pts: float = 3.0,
    daily_stop_loss: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Topstep-style NQ strategies with fixed-point brackets (stop_pts / target_pts).
    high_prob_only: for mean reversion, require deeper dip, clear down bar, optional RSI oversold; cap 3 t/day.
    Long only. Session filter and optional max_trades_per_day for consistency rule.

    strategy:
      - "vwap_micro": close > VWAP and close > prev_close. High frequency, 3/12 pts typical. ~82% pass in 7-8 days.
      - "broad": (close > EMA20 and EMA20 > EMA50) OR RSI > 55. 5/20 pts, 8-15 t/d. Best for 4-5 day pass (50-72%).
      - "trend_momentum": (EMA stack trend) OR (RSI 55-75 and close > VWAP); filter close > VWAP. 5/20 pts. Balance.
    """
    if contracts <= 0 or base_contracts <= 0 or stop_pts <= 0 or target_pts <= 0:
        raise ValueError("contracts, base_contracts, stop_pts, target_pts must be positive.")

    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col)
    df["trading_day"] = df[timestamp_col].dt.date.astype(str)

    close = df[close_col].astype(float)
    high = df[high_col].astype(float)
    low = df[low_col].astype(float)
    vol = df[volume_col].astype(float) if volume_col and volume_col in df.columns else pd.Series(1.0, index=df.index)

    df["_ema20"] = close.ewm(span=ema_fast, adjust=False).mean()
    df["_ema50"] = close.ewm(span=ema_slow, adjust=False).mean()
    df["_rsi"] = _rsi(close, rsi_period)
    df["_atr"] = _atr(high, low, close, 14)
    df["_atr_ref"] = df["_atr"].rolling(target_vol_window, min_periods=10).median()
    df["_vwap"] = _vwap_per_day(high, low, close, vol, df["trading_day"])
    if day_bias_filter:
        df = _session_day_features(
            df,
            timestamp_col=timestamp_col,
            open_col=open_col,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col,
            session_tz=session_tz,
            session_start_min=session_start_min,
            session_end_min=session_end_min,
            first_window_minutes=day_bias_first_window_minutes,
        )
    else:
        df["_session_day"] = df["trading_day"]
        df["_session_open"] = np.nan
        df["_prev_day_close"] = np.nan
        df["_first_window_move"] = np.nan
    if higher_tf_bias:
        df = _higher_timeframe_bias(
            df,
            timestamp_col=timestamp_col,
            close_col=close_col,
            higher_tf_minutes=higher_tf_minutes,
            ema_span=higher_tf_ema_span,
        )
    else:
        df["_htf_close"] = np.nan
        df["_htf_ema"] = np.nan
        df["_htf_ema_prev"] = np.nan
    df["_session"] = _session_mask(
        df[timestamp_col], session_start_min, session_end_min, session_tz=session_tz
    )
    df["_prev_close"] = close.shift(1)
    df["_prev_high"] = high.shift(1)

    positions: List[int] = []
    cum_pnls: List[float] = []
    cum_pnl = 0.0
    trades: List[Dict[str, Any]] = []
    position = 0
    entry_price = None
    stop_level = None
    target_level = None
    partial_target_level = None
    partial_taken = False
    highest_price_since_entry = None
    pending_long_signal = False
    pending_signal_prev_high = None
    trades_today = 0
    day_realized_pnl = 0.0
    last_day: Optional[str] = None
    n = len(df)

    for k in range(n):
        row = df.iloc[k]
        ts = row[timestamp_col]
        o = float(row[open_col])
        c = float(row[close_col])
        h = float(row[high_col])
        l_ = float(row[low_col])
        day = row["trading_day"]
        in_session = bool(row["_session"])
        if day != last_day:
            trades_today = 0
            day_realized_pnl = 0.0
            last_day = day

        if not in_session:
            if position != 0 and entry_price is not None:
                exit_price = c
                pnl = (exit_price - entry_price) * point_value * abs(position)
                cum_pnl += pnl
                day_realized_pnl += pnl
                trades.append({
                    "timestamp": ts, "instrument": "NQ", "entry_price": entry_price,
                    "exit_price": exit_price, "contracts": abs(position), "pnl": pnl,
                    "cumulative_pnl": cum_pnl, "trading_day": day,
                })
                position = 0
                entry_price = stop_level = target_level = partial_target_level = None
                partial_taken = False
                highest_price_since_entry = None
            positions.append(position)
            cum_pnls.append(cum_pnl)
            continue

        ema20 = row["_ema20"]
        ema50 = row["_ema50"]
        rsi = row["_rsi"]
        atr = row["_atr"]
        atr_ref = row["_atr_ref"]
        vwap = row["_vwap"]
        prev_close = row["_prev_close"]
        prev_high = row["_prev_high"]
        htf_close = row["_htf_close"]
        htf_ema = row["_htf_ema"]
        htf_ema_prev = row["_htf_ema_prev"]
        session_open = row["_session_open"]
        prev_day_close = row["_prev_day_close"]
        first_window_move = row["_first_window_move"]
        effective_max_trades = (min(max_trades_per_day, 3) if max_trades_per_day is not None else 3) if high_prob_only else max_trades_per_day
        if a_plus_only:
            effective_max_trades = min(effective_max_trades, 2) if effective_max_trades is not None else 2
        trades_ok = effective_max_trades is None or trades_today < effective_max_trades
        daily_stop_ok = daily_stop_loss is None or day_realized_pnl > -daily_stop_loss
        can_trade = trades_ok and daily_stop_ok

        # Entry logic by strategy
        long_signal = False
        a_plus_signal = False
        if strategy == "vwap_micro":
            if not (pd.isna(vwap) or pd.isna(prev_close)):
                if vwap_micro_mean_reversion:
                    long_signal = c <= float(vwap) - min_vwap_distance_pts and c < prev_close
                    if long_signal and high_prob_only:
                        dip_ok = (float(vwap) - c) >= min_dip_pts
                        down_bar_ok = (o > c) and (o - c) >= min_down_bar_pts
                        rsi_ok = rsi_oversold_max is None or (not pd.isna(rsi) and rsi < rsi_oversold_max)
                        long_signal = dip_ok and down_bar_ok and rsi_ok
                    if long_signal:
                        a_plus_dip_ok = (float(vwap) - c) >= a_plus_min_dip_pts
                        a_plus_down_bar_ok = (o > c) and (o - c) >= a_plus_min_down_bar_pts
                        a_plus_rsi_ok = a_plus_rsi_oversold_max is None or (not pd.isna(rsi) and rsi < a_plus_rsi_oversold_max)
                        a_plus_signal = a_plus_dip_ok and a_plus_down_bar_ok and a_plus_rsi_ok
                        if a_plus_only:
                            long_signal = a_plus_signal
                else:
                    long_signal = c >= float(vwap) + min_vwap_distance_pts and c > prev_close
        elif strategy == "broad":
            if not (pd.isna(ema20) or pd.isna(ema50)):
                trend_ok = c > ema20 and ema20 > ema50
                rsi_ok = not pd.isna(rsi) and rsi > 55
                long_signal = trend_ok or rsi_ok
        elif strategy == "trend_momentum":
            if not (pd.isna(ema20) or pd.isna(ema50) or pd.isna(vwap) or pd.isna(rsi)):
                trend = c > ema20 and c > ema50 and ema20 > ema50
                mom = 55 < rsi < 75 and c > vwap
                filter_ok = c > vwap
                long_signal = filter_ok and (trend or mom)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if higher_tf_bias:
            htf_bias_ok = (
                not pd.isna(htf_close)
                and not pd.isna(htf_ema)
                and not pd.isna(htf_ema_prev)
                and float(htf_close) >= float(htf_ema)
                and float(htf_ema) >= float(htf_ema_prev)
            )
            long_signal = long_signal and htf_bias_ok

        if day_bias_filter:
            gap_ok = (
                not pd.isna(session_open)
                and not pd.isna(prev_day_close)
                and float(session_open) - float(prev_day_close) >= day_bias_min_gap_pts
            )
            first_window_ok = (
                not pd.isna(first_window_move)
                and float(first_window_move) >= day_bias_min_first_window_move_pts
            )
            long_signal = long_signal and gap_ok and first_window_ok

        if confirmation_entry:
            if pending_long_signal and position == 0 and can_trade and entry_price is None:
                confirm_ok = c > o
                if confirmation_reclaim_prior_high and pending_signal_prev_high is not None:
                    confirm_ok = confirm_ok and c >= float(pending_signal_prev_high)
                if confirm_ok:
                    long_signal = True
                else:
                    long_signal = False
                pending_long_signal = False
                pending_signal_prev_high = None
            elif long_signal:
                pending_long_signal = True
                pending_signal_prev_high = prev_high
                long_signal = False

        if position == 0 and can_trade and long_signal and entry_price is None:
            entry_contracts = contracts
            if hybrid_sizing:
                entry_contracts = contracts if a_plus_signal else base_contracts
            elif a_plus_only and a_plus_signal:
                entry_contracts = contracts
            position = entry_contracts
            entry_price = c
            target_pts_for_trade = target_pts
            if adaptive_target and not pd.isna(atr) and not pd.isna(atr_ref):
                target_pts_for_trade = (
                    low_vol_target_pts if float(atr) <= float(atr_ref) else high_vol_target_pts
                )
            stop_level = entry_price - stop_pts
            target_level = entry_price + target_pts_for_trade
            partial_target_level = entry_price + (
                partial_exit_pts if partial_exit_pts is not None else target_pts_for_trade / 2.0
            )
            partial_taken = False
            highest_price_since_entry = h

        exit_now = False
        exit_price = None
        if position != 0 and entry_price is not None:
            highest_price_since_entry = max(highest_price_since_entry or h, h)
            if l_ <= stop_level:
                exit_now = True
                exit_price = stop_level
            else:
                if (
                    partial_scale_out
                    and not partial_taken
                    and abs(position) >= 2
                    and partial_target_level is not None
                    and h >= partial_target_level
                ):
                    scale_contracts = min(partial_exit_contracts, abs(position) - 1)
                    partial_pnl = (
                        (partial_target_level - entry_price) * point_value * scale_contracts
                    )
                    cum_pnl += partial_pnl
                    day_realized_pnl += partial_pnl
                    trades.append({
                        "timestamp": ts, "instrument": "NQ", "entry_price": entry_price,
                        "exit_price": partial_target_level, "contracts": scale_contracts, "pnl": partial_pnl,
                        "cumulative_pnl": cum_pnl, "trading_day": day,
                    })
                    position -= scale_contracts
                    partial_taken = True
                    if breakeven_after_partial:
                        stop_level = max(stop_level, entry_price)

                if (
                    partial_taken
                    and rolling_stop_distance_pts is not None
                    and highest_price_since_entry is not None
                ):
                    stop_level = max(
                        stop_level,
                        float(highest_price_since_entry) - rolling_stop_distance_pts,
                    )

                if h >= target_level:
                    exit_now = True
                    exit_price = target_level

                last_bar = (k + 1 >= n) or (df.iloc[k + 1]["trading_day"] != day)
                if last_bar and not exit_now:
                    exit_now = True
                    exit_price = c

        if exit_now and exit_price is not None:
            pnl = (exit_price - entry_price) * point_value * abs(position)
            cum_pnl += pnl
            day_realized_pnl += pnl
            trades.append({
                "timestamp": ts, "instrument": "NQ", "entry_price": entry_price,
                "exit_price": exit_price, "contracts": abs(position), "pnl": pnl,
                "cumulative_pnl": cum_pnl, "trading_day": day,
            })
            trades_today += 1
            position = 0
            entry_price = stop_level = target_level = partial_target_level = None
            partial_taken = False
            highest_price_since_entry = None

        positions.append(position)
        cum_pnls.append(cum_pnl)

    drop_cols = ["_ema20", "_ema50", "_rsi", "_atr", "_atr_ref", "_vwap", "_session", "_prev_close", "_prev_high", "_htf_close", "_htf_ema", "_htf_ema_prev", "_session_day", "_session_open", "_prev_day_close", "_first_window_move"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")
    df["position"] = positions
    df["cum_pnl"] = cum_pnls
    trades_df = pd.DataFrame(trades)
    return df, trades_df, cum_pnl


def run_momentum_breakout_volume_validated(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "datetime",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: Optional[str] = None,
    ma_period: int = 20,
    rsi_period: int = 14,
    atr_period: int = 14,
    volume_ma_period: int = 20,
    volume_multiplier: float = 2.0,
    rsi_entry_min: float = 50.0,
    rsi_exit_below: float = 40.0,
    stop_r: float = 1.0,
    target_r: float = 2.0,
    contracts: int = DEFAULT_CONTRACTS,
    point_value: float = POINT_VALUE,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Momentum Breakout with Volume Validation (long only).

    Entry: price breaks above MA, RSI > rsi_entry_min, volume >= volume_multiplier * vol_ma.
    Exit: stop (entry - stop_r*ATR), target (entry + target_r*ATR), or close < MA or RSI < rsi_exit_below, or EOD.
    One trade per day; R = ATR at entry.
    """
    if contracts <= 0 or stop_r <= 0 or target_r <= 0:
        raise ValueError("contracts, stop_r, target_r must be positive.")

    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col)
    df["trading_day"] = df[timestamp_col].dt.date.astype(str)

    close = df[close_col].astype(float)
    high = df[high_col].astype(float)
    low = df[low_col].astype(float)

    df["_ma"] = close.rolling(ma_period, min_periods=ma_period).mean()
    df["_rsi"] = _rsi(close, rsi_period)
    df["_atr"] = _atr(high, low, close, atr_period)
    if volume_col and volume_col in df.columns:
        df["_vol_ma"] = df[volume_col].astype(float).rolling(volume_ma_period, min_periods=1).mean()
    else:
        df["_vol_ma"] = np.nan

    positions: List[int] = []
    cum_pnls: List[float] = []
    cum_pnl = 0.0
    trades: List[Dict[str, Any]] = []

    position = 0
    entry_price = None
    entry_time = None
    stop_level = None
    target_level = None
    entry_atr = None
    last_traded_day: Optional[str] = None

    n = len(df)
    for k in range(n):
        row = df.iloc[k]
        ts = row[timestamp_col]
        c = float(row[close_col])
        h = float(row[high_col])
        l_ = float(row[low_col])
        ma = row["_ma"]
        rsi = row["_rsi"]
        atr = row["_atr"]
        vol_ma = row["_vol_ma"]
        day = row["trading_day"]

        if pd.isna(ma) or pd.isna(rsi) or (atr is not None and (pd.isna(atr) or atr <= 0)):
            positions.append(position)
            cum_pnls.append(cum_pnl)
            continue

        prev_ma = df["_ma"].iloc[k - 1] if k > 0 else np.nan
        prev_close = df[close_col].iloc[k - 1] if k > 0 else np.nan

        vol_ok = True
        if volume_col and volume_col in df.columns and not pd.isna(vol_ma) and vol_ma > 0:
            vol_ok = float(row[volume_col]) >= volume_multiplier * vol_ma

        cross_above = k > 0 and not pd.isna(prev_ma) and prev_close <= prev_ma and c > ma
        can_enter_today = last_traded_day != day

        if position == 0 and can_enter_today and cross_above and rsi > rsi_entry_min and vol_ok and atr > 0:
            position = contracts
            entry_price = c
            entry_time = ts
            entry_atr = atr
            stop_level = entry_price - stop_r * atr
            target_level = entry_price + target_r * atr

        exit_now = False
        exit_price = None
        exit_time = None

        if position != 0 and entry_price is not None:
            if l_ <= stop_level:
                exit_now = True
                exit_price = stop_level
                exit_time = ts
            elif h >= target_level:
                exit_now = True
                exit_price = target_level
                exit_time = ts
            elif c < ma or (not pd.isna(rsi) and rsi < rsi_exit_below):
                exit_now = True
                exit_price = c
                exit_time = ts
            else:
                last_bar_of_day = (k + 1 >= n) or (df.iloc[k + 1]["trading_day"] != day)
                if last_bar_of_day:
                    exit_now = True
                    exit_price = c
                    exit_time = ts

        if exit_now and exit_price is not None and exit_time is not None:
            pnl = (exit_price - entry_price) * point_value * contracts
            cum_pnl += pnl
            trades.append({
                "timestamp": exit_time,
                "instrument": "NQ",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "contracts": abs(position),
                "pnl": pnl,
                "cumulative_pnl": cum_pnl,
                "trading_day": day,
            })
            last_traded_day = day
            position = 0
            entry_price = None
            entry_time = None
            stop_level = None
            target_level = None
            entry_atr = None

        positions.append(position)
        cum_pnls.append(cum_pnl)

    df["position"] = positions
    df["cum_pnl"] = cum_pnls
    df.drop(columns=["_ma", "_rsi", "_atr", "_vol_ma"], inplace=True, errors="ignore")

    trades_df = pd.DataFrame(trades)
    return df, trades_df, cum_pnl


START_EQUITY = 50_000.0
PROFIT_TARGET = 3_000.0
TRAILING_DRAWDOWN = 2_000.0
DAILY_LOSS_LIMIT = 1_000.0
MIN_TRADING_DAYS = 2  # Topstep 50K rule: min 2 trading days
MAX_CONTRACTS = 5
COMBINE_COST = 49.0

# Economic rule (keep consistent everywhere):
# - Losses from failed/blown accounts are UNREALISED (not counted as real loss).
# - Only the evaluation fee (COMBINE_COST) per attempt is the actual cost of a failed account.
# - Realised PnL = profits from passed accounts only. EV = pass_rate * avg_pass_profit - combine_cost.


def simulate_one_attempt(
    day_trades: List[pd.DataFrame],
    rng: np.random.Generator,
    max_days: int = 252,
) -> Dict[str, Any]:
    """
    Simulate a single Topstep $50k Combine attempt by bootstrapping trading days.
    """
    equity = START_EQUITY
    peak_equity = START_EQUITY
    trailing_level = peak_equity - TRAILING_DRAWDOWN
    max_drawdown = 0.0

    days_traded = 0
    rule_violations: List[str] = []
    per_day_profits: List[float] = []
    passed = False

    while days_traded < max_days and not passed:
        idx = rng.integers(0, len(day_trades))
        day_df = day_trades[idx]

        day_pnl = 0.0

        for _, trade in day_df.iterrows():
            if trade["contracts"] > MAX_CONTRACTS:
                rule_violations.append("contract_limit")
                result = "FAIL"
                final_profit = equity - START_EQUITY
                return {
                    "Profit": final_profit,
                    "Max Drawdown": max_drawdown,
                    "Days Traded": days_traded,
                    "Rule Violations": ",".join(sorted(set(rule_violations))) or "none",
                    "Result": result,
                }

            pnl = float(trade["pnl"])
            equity += pnl
            day_pnl += pnl

            if equity > peak_equity:
                peak_equity = equity
            trailing_level = peak_equity - TRAILING_DRAWDOWN

            current_dd = peak_equity - equity
            if current_dd > max_drawdown:
                max_drawdown = current_dd

            if equity < trailing_level - 1e-8:
                rule_violations.append("trailing_drawdown")
                result = "FAIL"
                final_profit = equity - START_EQUITY
                return {
                    "Profit": final_profit,
                    "Max Drawdown": max_drawdown,
                    "Days Traded": days_traded + 1,
                    "Rule Violations": ",".join(sorted(set(rule_violations))) or "none",
                    "Result": result,
                }

        if day_pnl <= -DAILY_LOSS_LIMIT - 1e-8:
            rule_violations.append("daily_loss_limit")
            equity_after = equity
            final_profit = equity_after - START_EQUITY
            days_traded += 1
            per_day_profits.append(day_pnl)
            result = "FAIL"
            return {
                "Profit": final_profit,
                "Max Drawdown": max_drawdown,
                "Days Traded": days_traded,
                "Rule Violations": ",".join(sorted(set(rule_violations))) or "none",
                "Result": result,
            }

        days_traded += 1
        per_day_profits.append(day_pnl)

        total_profit = equity - START_EQUITY
        if (total_profit >= PROFIT_TARGET) and (days_traded >= MIN_TRADING_DAYS):
            passed = True
            break

    total_profit = equity - START_EQUITY

    if passed:
        result = "PASS"
    else:
        if total_profit < PROFIT_TARGET:
            rule_violations.append("profit_target_not_reached")
        result = "FAIL"

    if total_profit > 0:
        max_day_profit = max(per_day_profits) if per_day_profits else 0.0
        if max_day_profit > 0.5 * total_profit + 1e-8:
            rule_violations.append("consistency_rule")
            result = "FAIL"

    return {
        "Profit": total_profit,
        "Max Drawdown": max_drawdown,
        "Days Traded": days_traded,
        "Rule Violations": ",".join(sorted(set(rule_violations))) or "none",
        "Result": result,
    }


def monte_carlo_topstep(
    trades: pd.DataFrame,
    n_simulations: int = 1000,
    combine_cost: float = COMBINE_COST,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Monte Carlo simulations of the Topstep Combine using historical trade days.

    Economic rule (unrealised losses): losses from failed/blown accounts are NOT
    counted as realised. Only the evaluation fee per attempt is the cost of a fail.
    EV = pass_rate * avg_profit_passed - combine_cost. Keep this rule consistent.
    """
    required_cols = [
        "timestamp",
        "instrument",
        "entry_price",
        "exit_price",
        "contracts",
        "pnl",
        "cumulative_pnl",
        "trading_day",
    ]
    missing = [c for c in required_cols if c not in trades.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    trades = trades.sort_values(["trading_day", "timestamp"])

    grouped = list(trades.groupby("trading_day"))
    day_trades = [g[1].reset_index(drop=True) for g in grouped]

    rng = np.random.default_rng(random_seed)

    attempt_results: List[Dict[str, Any]] = []
    for attempt in range(1, n_simulations + 1):
        res = simulate_one_attempt(day_trades, rng=rng)
        res["Attempt"] = attempt
        attempt_results.append(res)

    attempts_df = pd.DataFrame(attempt_results)

    total_simulations = len(attempts_df)
    passed_mask = attempts_df["Result"] == "PASS"
    failed_mask = ~passed_mask

    passed_accounts = passed_mask.sum()
    failed_accounts = failed_mask.sum()
    pass_rate = passed_accounts / total_simulations if total_simulations > 0 else 0.0
    fail_rate = 1.0 - pass_rate

    avg_profit_passed = (
        attempts_df.loc[passed_mask, "Profit"].mean()
        if passed_accounts > 0
        else 0.0
    )
    # Failed-account PnL is informational only; we do NOT count it as realised loss (see economic rule above).
    avg_loss_failed = (
        attempts_df.loc[failed_mask, "Profit"].mean()
        if failed_accounts > 0
        else 0.0
    )

    if len(attempts_df) > 0:
        dd_series = attempts_df["Max Drawdown"]
        dd_mean = dd_series.mean()
        dd_p05 = dd_series.quantile(0.05)
        dd_p50 = dd_series.quantile(0.50)
        dd_p95 = dd_series.quantile(0.95)
        max_dd_distribution_str = (
            f"mean={dd_mean:.0f}, p5={dd_p05:.0f}, median={dd_p50:.0f}, p95={dd_p95:.0f}"
        )
    else:
        max_dd_distribution_str = "n/a"

    avg_days_to_pass = (
        attempts_df.loc[passed_mask, "Days Traded"].mean()
        if passed_accounts > 0
        else 0.0
    )

    # Economic rule: failed-account losses UNREALISED; only fee + passed-account profits count.
    expected_value_per_combine = pass_rate * avg_profit_passed - combine_cost

    summary_rows = [
        ("Total Simulations", total_simulations),
        ("Passed Accounts", passed_accounts),
        ("Failed Accounts", failed_accounts),
        ("Pass Rate", pass_rate),
        ("Average Profit (Passed Accounts)", avg_profit_passed),
        ("Average PnL (Failed Accounts, unrealised)", avg_loss_failed),
        ("Expected Value Per Combine", expected_value_per_combine),
        ("Max Drawdown Distribution", max_dd_distribution_str),
        ("Average Days to Pass", avg_days_to_pass),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])

    attempts_df = attempts_df[
        ["Attempt", "Profit", "Max Drawdown", "Days Traded", "Rule Violations", "Result"]
    ].sort_values("Attempt")

    return summary_df, attempts_df


def _parse_session_minutes(s: str) -> int:
    """Parse 'HH:MM' or 'H:MM' to minutes from midnight."""
    parts = s.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected HH:MM, got {s}")
    h, m = int(parts[0]), int(parts[1])
    return h * 60 + m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Momentum Breakout or Topstep NQ strategies on futures data, "
            "with Topstep-style Monte Carlo and optional R:R grid search."
        )
    )

    parser.add_argument("parquet_path", help="Path to the parquet file.")
    parser.add_argument(
        "--strategy",
        choices=[
            "momentum",
            "topstep_vwap_micro",
            "topstep_broad",
            "topstep_trend_momentum",
            "topstep_vwap_micro_momentum",
        ],
        default="momentum",
        help=(
            "Strategy: momentum (ATR brackets), topstep_vwap_micro (~82%% pass, 7-8 days), "
            "topstep_broad (best for 4-5 days), topstep_trend_momentum (balance), "
            "topstep_vwap_micro_momentum (VWAP slope + Micro Momentum z-score, ATR stops)."
        ),
    )
    parser.add_argument(
        "--timestamp-col",
        default="datetime",
        help="Column name used to sort the data chronologically.",
    )
    parser.add_argument(
        "--stop-pts",
        type=float,
        default=5.0,
        help="Fixed stop in points (NQ). Used by Topstep strategies; default 5.",
    )
    parser.add_argument(
        "--target-pts",
        type=float,
        default=20.0,
        help="Fixed target in points (NQ). Used by Topstep strategies; default 20.",
    )
    parser.add_argument(
        "--session-start",
        default="09:30",
        help="Session start time HH:MM (same TZ as data). Default 09:30.",
    )
    parser.add_argument(
        "--session-end",
        default="15:45",
        help="Session end time HH:MM. Default 15:45 (RTH).",
    )
    parser.add_argument(
        "--session-tz",
        default=None,
        help="Timezone for session (e.g. America/New_York). If data is UTC, use this so 09:30-15:45 = RTH.",
    )
    parser.add_argument(
        "--max-trades-per-day",
        type=int,
        default=None,
        help="Cap trades per day (consistency rule). None = no cap.",
    )
    parser.add_argument(
        "--ma-period",
        type=int,
        default=20,
        help="Moving average period for breakout level (momentum strategy).",
    )
    parser.add_argument(
        "--rsi-period",
        type=int,
        default=14,
        help="RSI period.",
    )
    parser.add_argument(
        "--volume-multiplier",
        type=float,
        default=2.0,
        help="Require bar volume >= this multiple of volume MA (e.g. 2.0 = 2x).",
    )
    parser.add_argument(
        "--stop-r",
        type=float,
        default=1.0,
        help="Stop loss in ATR multiples (R) for single run.",
    )
    parser.add_argument(
        "--target-r",
        type=float,
        default=2.0,
        help="Take profit in ATR multiples (R) for single run.",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Run grid search over stop_r and target_r; print ranked RR combinations.",
    )
    parser.add_argument(
        "--stop-r-values",
        default="0.5,1.0,1.5,2.0",
        help="Comma-separated stop_r values for grid search.",
    )
    parser.add_argument(
        "--target-r-values",
        default="1.0,1.5,2.0,2.5,3.0",
        help="Comma-separated target_r values for grid search.",
    )
    parser.add_argument(
        "--contracts",
        type=int,
        default=DEFAULT_CONTRACTS,
        help="Number of contracts per trade (must be <= MAX_CONTRACTS).",
    )
    parser.add_argument(
        "--point-value",
        type=float,
        default=POINT_VALUE,
        help="Dollar value per point per contract (e.g. 20 for NQ).",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=100,
        help="Number of Monte Carlo Topstep account attempts per run.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to save bar-level backtest data (single run only).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        df = pd.read_parquet(args.parquet_path)
    except ImportError as e:
        raise SystemExit(
            "Reading parquet requires pandas with a parquet engine installed "
            "(e.g. pyarrow). Install with:\n"
            "  pip install pandas pyarrow\n"
        ) from e

    if isinstance(df.index, pd.DatetimeIndex):
        ts_name = df.index.name or "datetime"
        df = df.reset_index()
        if args.timestamp_col is None:
            args.timestamp_col = ts_name
        elif args.timestamp_col != ts_name:
            df = df.rename(columns={ts_name: args.timestamp_col})

    if args.timestamp_col and args.timestamp_col in df.columns:
        df = df.sort_values(args.timestamp_col)

    required_ohlc = ["open", "high", "low", "close"]
    missing = [c for c in required_ohlc if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing OHLC columns in parquet file: {missing}")

    if args.contracts > MAX_CONTRACTS:
        raise SystemExit(
            f"Strategy contracts ({args.contracts}) exceed Topstep limit ({MAX_CONTRACTS})."
        )

    volume_col = "volume" if "volume" in df.columns else None

    def run_one(stop_r: float, target_r: float) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        return run_momentum_breakout_volume_validated(
            df,
            timestamp_col=args.timestamp_col,
            open_col="open",
            high_col="high",
            low_col="low",
            close_col="close",
            volume_col=volume_col,
            ma_period=args.ma_period,
            rsi_period=args.rsi_period,
            volume_ma_period=20,
            volume_multiplier=args.volume_multiplier,
            stop_r=stop_r,
            target_r=target_r,
            contracts=args.contracts,
            point_value=args.point_value,
        )

    use_topstep = args.strategy.startswith("topstep_")
    if use_topstep:
        try:
            session_start_min = _parse_session_minutes(args.session_start)
            session_end_min = _parse_session_minutes(args.session_end)
        except ValueError as e:
            raise SystemExit(f"Invalid session time: {e}") from e
        session_tz = getattr(args, "session_tz", None)
        strategy_name = args.strategy.replace("topstep_", "", 1)

    if args.grid_search and use_topstep:
        raise SystemExit(
            "Grid search is only supported for strategy=momentum. "
            "For Topstep strategies, tune --stop-pts and --target-pts manually."
        )
    if args.grid_search:
        stop_vals = [float(x.strip()) for x in args.stop_r_values.split(",")]
        target_vals = [float(x.strip()) for x in args.target_r_values.split(",")]
        n_sim = args.simulations
        results: List[Dict[str, Any]] = []
        for stop_r in stop_vals:
            for target_r in target_vals:
                if target_r <= stop_r:
                    continue
                bar_df, trades_df, final_pnl = run_one(stop_r, target_r)
                if trades_df.empty:
                    results.append({
                        "stop_r": stop_r,
                        "target_r": target_r,
                        "rr_ratio": target_r / stop_r,
                        "backtest_pnl": final_pnl,
                        "pass_rate": 0.0,
                        "total_pnl_100": -n_sim * COMBINE_COST,
                        "ev": -COMBINE_COST,
                    })
                    continue
                summary, attempts = monte_carlo_topstep(
                    trades_df, n_simulations=n_sim, combine_cost=COMBINE_COST
                )
                pass_rate = float(summary.loc[summary["Metric"] == "Pass Rate", "Value"].iloc[0])
                ev = float(summary.loc[summary["Metric"] == "Expected Value Per Combine", "Value"].iloc[0])
                passed_profit = attempts.loc[attempts["Result"] == "PASS", "Profit"].sum()
                total_pnl_100 = passed_profit - n_sim * COMBINE_COST
                results.append({
                    "stop_r": stop_r,
                    "target_r": target_r,
                    "rr_ratio": target_r / stop_r,
                    "backtest_pnl": final_pnl,
                    "pass_rate": pass_rate,
                    "total_pnl_100": total_pnl_100,
                    "ev": ev,
                })
        res_df = pd.DataFrame(results)
        res_df = res_df.sort_values("ev", ascending=False).reset_index(drop=True)
        res_df["rank"] = res_df.index + 1
        print("Momentum Breakout + Volume Validation — R:R grid search results")
        print("(Economic rule: failed-account losses unrealised; total PnL = passed profit - n*fee.)")
        print(res_df[["rank", "stop_r", "target_r", "rr_ratio", "backtest_pnl", "pass_rate", "total_pnl_100", "ev"]].to_string(index=False))
        return

    if use_topstep:
        if args.strategy == "topstep_vwap_micro_momentum":
            bar_df, trades_df, final_pnl = run_vwap_micro_momentum_topstep(
                df,
                timestamp_col=args.timestamp_col,
                open_col="open",
                high_col="high",
                low_col="low",
                close_col="close",
                volume_col=volume_col,
                contracts=args.contracts,
                point_value=args.point_value,
                session_start_min=session_start_min,
                session_end_min=session_end_min,
                session_tz=session_tz,
                max_trades_per_day=args.max_trades_per_day or 10,
            )
        else:
            bar_df, trades_df, final_pnl = run_topstep_nq(
                df,
                strategy=strategy_name,
                timestamp_col=args.timestamp_col,
                open_col="open",
                high_col="high",
                low_col="low",
                close_col="close",
                volume_col=volume_col,
                stop_pts=args.stop_pts,
                target_pts=args.target_pts,
                contracts=args.contracts,
                point_value=args.point_value,
                session_start_min=session_start_min,
                session_end_min=session_end_min,
                session_tz=session_tz,
                max_trades_per_day=args.max_trades_per_day,
            )
        print(f"Topstep NQ ({args.strategy}) final PnL: {final_pnl}")
    else:
        bar_df, trades_df, final_pnl = run_one(args.stop_r, args.target_r)
        print("Momentum Breakout (Volume Validated) final PnL:", final_pnl)

    if args.output_csv:
        bar_df.to_csv(args.output_csv, index=False)
        print("Saved bar-level backtest data to:", args.output_csv)

    if trades_df.empty:
        print("No trades were generated by the strategy rules.")
        return

    summary, attempts = monte_carlo_topstep(
        trades_df, n_simulations=args.simulations, combine_cost=COMBINE_COST
    )
    print("\nTopstep-style Monte Carlo summary:")
    print(summary.to_string(index=False))
    print("\n(Economic rule: failed-account losses are unrealised; only evaluation fee and passed-account profits count.)")
    n = args.simulations
    passed_profit_total = attempts.loc[attempts["Result"] == "PASS", "Profit"].sum()
    total_pnl_after_n = passed_profit_total - n * COMBINE_COST
    print(f"\nTotal PnL after {n} accounts (unrealised losses): {total_pnl_after_n:.2f}")


if __name__ == "__main__":
    main()

