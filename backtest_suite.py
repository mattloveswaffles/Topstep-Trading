"""
Backtesting Suite — Topstep NQ Futures
======================================

Clean, modular backtesting framework with:
  - Strategy registry (add strategies as simple config dicts)
  - Grid search with parameter optimization
  - Monte Carlo eval simulation (pessimistic)
  - Funded account payout simulation
  - Side-by-side strategy comparison
  - Split testing (in-sample / out-of-sample)

Usage:
  python backtest_suite.py                    # run all registered strategies
  python backtest_suite.py --grid             # grid search for optimal params
  python backtest_suite.py --split            # split test (IS/OOS validation)
  python backtest_suite.py --strategy vwap_mr # run a single strategy
  python backtest_suite.py --combo             # test all eval/funded strategy combos
  python backtest_suite.py --combo --eval-strategy vwap_mr --funded-strategy vwap_mr_conservative
"""

import time
import warnings
import argparse
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from engine.backtest import (
    run_topstep_nq,
    monte_carlo_topstep,
    START_EQUITY,
    COMBINE_COST,
    PROFIT_TARGET,
    TRAILING_DRAWDOWN,
    DAILY_LOSS_LIMIT,
    POINT_VALUE,
)
from engine.alpha import load_daily, signal_to_trades, ALPHAS

warnings.filterwarnings("ignore")
console = Console()

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

DATA_PATH = "data/nq_1m.parquet"
DEFAULT_SIMS = 50_000
FUNDED_SIMS = 10_000
FUNDED_MAX_DAYS = 500

# Topstep funded account rules
FUNDED_TRAILING_DD = 2000.0
FUNDED_DAILY_LOSS = 1000.0
WINNING_DAY_THRESHOLD = 150.0
WINNING_DAYS_REQUIRED = 5
PAYOUT_MAX = 5000.0
PAYOUT_BAL_PCT = 0.50
ACTIVATION_FEE = 150.0


# ═══════════════════════════════════════════════════════════════════════════
# Strategy Registry
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Strategy:
    """A named strategy configuration for backtesting."""
    name: str
    description: str
    params: Dict

    # Default backtest params (can be overridden in params)
    strategy_type: str = "vwap_micro"
    stop_pts: float = 5.0
    target_pts: float = 20.0
    contracts: int = 2
    session_start_min: int = 570   # 9:30 AM ET
    session_end_min: int = 945     # 3:45 PM ET
    session_tz: str = "America/New_York"
    max_trades_per_day: int = 5
    trailing_stop_pts: Optional[float] = 2.0
    breakeven_stop_pts: Optional[float] = None
    long_only: bool = False
    alpha_fn: Optional[str] = None  # Alpha function name (e.g., "Alpha#44") for alpha strategies
    cooldown_after_loss_minutes: Optional[int] = None  # Minutes to wait after a losing trade
    max_daily_pnl: Optional[float] = None  # Stop trading for the day once daily PnL hits this
    sessions: Optional[List[Tuple[int, int]]] = None  # Multi-session: list of (start_min, end_min)
    session_day_start_min: Optional[int] = None  # Day boundary shift for overnight (e.g. 1080=6PM)


# ── Register your strategies here ────────────────────────────────────────

STRATEGIES: Dict[str, Strategy] = {}


def register_strategy(name: str, description: str, **kwargs) -> Strategy:
    """Register a strategy config for backtesting."""
    params = kwargs.pop("params", {})
    s = Strategy(name=name, description=description, params=params, **kwargs)
    STRATEGIES[name] = s
    return s


# Current best: VWAP Micro Mean-Reversion (pessimistic backtest)
register_strategy(
    name="vwap_mr",
    description="VWAP Micro MR — Full day, L+S, no bias, trail=2pt",
    strategy_type="vwap_micro",
    stop_pts=5.0,
    target_pts=20.0,
    contracts=2,
    session_start_min=9 * 60 + 30,
    session_end_min=15 * 60 + 45,
    trailing_stop_pts=2.0,
    long_only=False,
    params={
        "vwap_micro_mean_reversion": True,
        "high_prob_only": True,
        "day_bias_filter": False,
        "min_dip_pts": 1.5,
        "min_down_bar_pts": 0.5,
        "rsi_oversold_max": 50.0,
        "rsi_overbought_min": 50.0,
    },
)

# Alpha#44: -correlation(high, rank(volume), 5) — daily signal strategy
register_strategy(
    name="alpha44",
    description="Alpha#44 — daily signal, long when -corr(high, rank(vol), 5) > 0",
    strategy_type="alpha",
    alpha_fn="Alpha#44",
)

# ── 180-min cooldown + $1600 daily PnL cap variants ──────────────────────

register_strategy(
    name="vwap_mr_cd180_cap1600",
    description="VWAP MR + 180min cooldown + $1600 daily cap",
    strategy_type="vwap_micro",
    stop_pts=5.0,
    target_pts=20.0,
    contracts=2,
    session_start_min=9 * 60 + 30,
    session_end_min=15 * 60 + 45,
    trailing_stop_pts=2.0,
    long_only=False,
    cooldown_after_loss_minutes=180,
    max_daily_pnl=1600.0,
    params={
        "vwap_micro_mean_reversion": True,
        "high_prob_only": True,
        "day_bias_filter": False,
        "min_dip_pts": 1.5,
        "min_down_bar_pts": 0.5,
        "rsi_oversold_max": 50.0,
        "rsi_overbought_min": 50.0,
    },
)

# ── Session definitions (ET) ─────────────────────────────────────────────
# Asia:   6:00 PM - 2:00 AM ET (wraps midnight)
# London: 3:00 AM - 9:30 AM ET
# NY:     9:30 AM - 3:45 PM ET
SESSION_ASIA   = (18 * 60, 2 * 60)          # 1080, 120
SESSION_LONDON = (3 * 60, 9 * 60 + 30)      # 180, 570
SESSION_NY     = (9 * 60 + 30, 15 * 60 + 45) # 570, 945

# Topstep day starts at Asia open (6PM ET) — trade caps shared across sessions
TOPSTEP_DAY_START = 18 * 60  # 1080 minutes = 6:00 PM ET

_MR_PARAMS = {
    "vwap_micro_mean_reversion": True,
    "high_prob_only": True,
    "day_bias_filter": False,
    "min_dip_pts": 1.5,
    "min_down_bar_pts": 0.5,
    "rsi_oversold_max": 50.0,
    "rsi_overbought_min": 50.0,
}

# Asia only
register_strategy(
    name="vwap_mr_asia",
    description="VWAP MR — Asia (6PM-2AM ET), shared day cap",
    strategy_type="vwap_micro",
    stop_pts=5.0, target_pts=20.0, contracts=2,
    sessions=[SESSION_ASIA],
    session_day_start_min=TOPSTEP_DAY_START,
    trailing_stop_pts=2.0, long_only=False,
    cooldown_after_loss_minutes=180, max_daily_pnl=1600.0,
    params=_MR_PARAMS,
)

# London only
register_strategy(
    name="vwap_mr_london",
    description="VWAP MR — London (3AM-9:30AM ET), shared day cap",
    strategy_type="vwap_micro",
    stop_pts=5.0, target_pts=20.0, contracts=2,
    sessions=[SESSION_LONDON],
    session_day_start_min=TOPSTEP_DAY_START,
    trailing_stop_pts=2.0, long_only=False,
    cooldown_after_loss_minutes=180, max_daily_pnl=1600.0,
    params=_MR_PARAMS,
)

# Asia + London
register_strategy(
    name="vwap_mr_asia_london",
    description="VWAP MR — Asia+London (6PM-9:30AM ET), shared day cap",
    strategy_type="vwap_micro",
    stop_pts=5.0, target_pts=20.0, contracts=2,
    sessions=[SESSION_ASIA, SESSION_LONDON],
    session_day_start_min=TOPSTEP_DAY_START,
    trailing_stop_pts=2.0, long_only=False,
    cooldown_after_loss_minutes=180, max_daily_pnl=1600.0,
    params=_MR_PARAMS,
)

# Asia + NY
register_strategy(
    name="vwap_mr_asia_ny",
    description="VWAP MR — Asia+NY, shared day cap",
    strategy_type="vwap_micro",
    stop_pts=5.0, target_pts=20.0, contracts=2,
    sessions=[SESSION_ASIA, SESSION_NY],
    session_day_start_min=TOPSTEP_DAY_START,
    trailing_stop_pts=2.0, long_only=False,
    cooldown_after_loss_minutes=180, max_daily_pnl=1600.0,
    params=_MR_PARAMS,
)

# London + NY
register_strategy(
    name="vwap_mr_london_ny",
    description="VWAP MR — London+NY (3AM-3:45PM ET), shared day cap",
    strategy_type="vwap_micro",
    stop_pts=5.0, target_pts=20.0, contracts=2,
    sessions=[SESSION_LONDON, SESSION_NY],
    session_day_start_min=TOPSTEP_DAY_START,
    trailing_stop_pts=2.0, long_only=False,
    cooldown_after_loss_minutes=180, max_daily_pnl=1600.0,
    params=_MR_PARAMS,
)

# All sessions: Asia + London + NY (full Topstep day)
register_strategy(
    name="vwap_mr_all_sessions",
    description="VWAP MR — Asia+London+NY, shared 5-trade day cap",
    strategy_type="vwap_micro",
    stop_pts=5.0, target_pts=20.0, contracts=2,
    sessions=[SESSION_ASIA, SESSION_LONDON, SESSION_NY],
    session_day_start_min=TOPSTEP_DAY_START,
    trailing_stop_pts=2.0, long_only=False,
    cooldown_after_loss_minutes=180, max_daily_pnl=1600.0,
    params=_MR_PARAMS,
)


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_data(path: str = DATA_PATH) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load NQ OHLCV data. Returns (df_5m, df_1m).

    If source is 1m data, returns both 5m (for signals) and 1m (for stop management).
    If source is already 5m, returns (df_5m, None).
    """
    df = pd.read_parquet(path)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index(names=["datetime"])
    if "datetime" in df.columns and df["datetime"].dt.tz is None:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime")

    df_1m = None
    # Detect 1-min data and resample to 5-min
    if len(df) > 1000:
        sample_diff = (df["datetime"].iloc[10] - df["datetime"].iloc[9]).total_seconds()
        if sample_diff < 120:  # 1-min data
            df_1m = df.copy()
            print("  Resampling 1m -> 5m...", end=" ", flush=True)
            df = df.set_index("datetime")
            df = df.resample("5min").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna().reset_index()
            print(f"{len(df):,} bars")

    return df, df_1m


# ═══════════════════════════════════════════════════════════════════════════
# Core Backtest Runner
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    """Results from a single strategy backtest."""
    name: str
    trades_df: pd.DataFrame
    total_trades: int = 0
    long_trades: int = 0
    short_trades: int = 0
    trading_days: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_pnl: float = 0.0
    total_pnl: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    avg_duration_min: float = 0.0
    # MC results
    pass_rate: float = 0.0
    avg_cal_days: float = 0.0
    avg_trade_days: float = 0.0
    avg_profit_on_pass: float = 0.0
    eval_ev: float = 0.0
    # Funded results
    payout_pct: float = 0.0
    avg_days_to_payout: float = 0.0
    avg_per_payout: float = 0.0
    avg_total_extracted: float = 0.0
    avg_payouts: float = 0.0
    funded_bust_rate: float = 0.0
    # Full cycle
    full_cycle_ev: float = 0.0


def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    volume_col: Optional[str] = None,
    daily_df: Optional[pd.DataFrame] = None,
    df_1m: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Run a strategy backtest, returning (bar_df, trades_df, final_pnl)."""

    # Alpha-based strategies use daily signal pipeline
    if strategy.strategy_type == "alpha" and strategy.alpha_fn:
        if daily_df is None:
            daily_df = load_daily()
        alpha_func = ALPHAS.get(strategy.alpha_fn)
        if alpha_func is None:
            raise ValueError(f"Unknown alpha: {strategy.alpha_fn}")
        signal = alpha_func(daily_df)
        trades_df = signal_to_trades(daily_df, signal)
        final_pnl = trades_df["pnl"].sum() if not trades_df.empty else 0.0
        return pd.DataFrame(), trades_df, final_pnl

    # VWAP micro and other bar-level strategies
    if volume_col is None:
        volume_col = "volume" if "volume" in df.columns else None

    return run_topstep_nq(
        df,
        strategy=strategy.strategy_type,
        stop_pts=strategy.stop_pts,
        target_pts=strategy.target_pts,
        contracts=strategy.contracts,
        session_start_min=strategy.session_start_min,
        session_end_min=strategy.session_end_min,
        session_tz=strategy.session_tz,
        max_trades_per_day=strategy.max_trades_per_day,
        trailing_stop_pts=strategy.trailing_stop_pts,
        breakeven_stop_pts=strategy.breakeven_stop_pts,
        long_only=strategy.long_only,
        volume_col=volume_col,
        cooldown_after_loss_minutes=strategy.cooldown_after_loss_minutes,
        max_daily_pnl=strategy.max_daily_pnl,
        df_1m=df_1m,
        sessions=strategy.sessions,
        session_day_start_min=strategy.session_day_start_min,
        **strategy.params,
    )


def compute_trade_stats(trades_df: pd.DataFrame) -> Dict:
    """Compute trade-level statistics from a trades DataFrame."""
    if trades_df.empty:
        return {
            "total_trades": 0, "long_trades": 0, "short_trades": 0,
            "trading_days": 0, "win_rate": 0, "avg_win": 0, "avg_loss": 0,
            "avg_pnl": 0, "total_pnl": 0, "profit_factor": 0,
            "max_drawdown": 0, "sharpe": 0,
        }

    pnl = trades_df["pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    daily_pnl = trades_df.groupby("trading_day")["pnl"].sum()

    return {
        "total_trades": len(trades_df),
        "long_trades": len(trades_df[trades_df.get("side", pd.Series()) == "long"]) if "side" in trades_df.columns else 0,
        "short_trades": len(trades_df[trades_df.get("side", pd.Series()) == "short"]) if "side" in trades_df.columns else 0,
        "trading_days": trades_df["trading_day"].nunique(),
        "win_rate": len(wins) / len(pnl) * 100 if len(pnl) > 0 else 0,
        "avg_win": wins.mean() if len(wins) > 0 else 0,
        "avg_loss": losses.mean() if len(losses) > 0 else 0,
        "avg_pnl": pnl.mean(),
        "total_pnl": pnl.sum(),
        "profit_factor": wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 999,
        "max_drawdown": (pnl.cumsum() - pnl.cumsum().cummax()).min(),
        "sharpe": daily_pnl.mean() / daily_pnl.std() * (252 ** 0.5) if daily_pnl.std() > 0 else 0,
        "avg_duration_min": trades_df["duration_min"].mean() if "duration_min" in trades_df.columns else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Monte Carlo — Eval Simulation
# ═══════════════════════════════════════════════════════════════════════════

def run_eval_mc(
    trades_df: pd.DataFrame,
    total_calendar_days: int = 0,
    total_data_days: int = 0,
    n_sims: int = DEFAULT_SIMS,
) -> Dict:
    """Run Monte Carlo eval simulation. Returns key metrics as dict."""
    if trades_df.empty:
        return {
            "pass_rate": 0, "avg_cal_days": 0, "avg_trade_days": 0,
            "avg_profit_on_pass": 0, "eval_ev": 0,
        }

    summary_df, attempts_df = monte_carlo_topstep(
        trades_df,
        n_simulations=n_sims,
        total_calendar_days=total_calendar_days,
        total_data_days=total_data_days,
    )

    passed = attempts_df[attempts_df["Result"] == "PASS"]
    failed = attempts_df[attempts_df["Result"] == "FAIL"]
    pass_rate = len(passed) / len(attempts_df) * 100

    avg_profit = passed["Profit"].mean() if len(passed) > 0 else 0
    ev = (pass_rate / 100) * avg_profit - (1 - pass_rate / 100) * COMBINE_COST

    return {
        "pass_rate": pass_rate,
        "avg_cal_days": passed["Calendar Days"].mean() if len(passed) > 0 else 0,
        "avg_trade_days": passed["Trading Days"].mean() if len(passed) > 0 else 0,
        "avg_profit_on_pass": avg_profit,
        "eval_ev": ev,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Monte Carlo — Funded Payout Simulation
# ═══════════════════════════════════════════════════════════════════════════

def run_funded_mc(
    trades_df: pd.DataFrame,
    n_sims: int = FUNDED_SIMS,
    max_days: int = FUNDED_MAX_DAYS,
    seed: int = 42,
) -> Dict:
    """Simulate funded account payouts with consistency rule."""
    if trades_df.empty:
        return {
            "payout_pct": 0, "avg_days_to_payout": 0, "avg_per_payout": 0,
            "avg_total_extracted": 0, "avg_payouts": 0, "funded_bust_rate": 0,
        }

    daily_pnl = trades_df.groupby("trading_day")["pnl"].sum().values
    rng = np.random.default_rng(seed)

    N = n_sims
    equity = np.zeros(N)
    peak = np.zeros(N)
    winning_days = np.zeros(N, dtype=int)
    profit_since_payout = np.zeros(N)
    total_extracted = np.zeros(N)
    payout_count = np.zeros(N, dtype=int)
    first_payout_day = np.full(N, -1, dtype=int)
    alive = np.ones(N, dtype=bool)

    for day in range(max_days):
        if not alive.any():
            break

        idx = rng.integers(0, len(daily_pnl), size=N)
        dpnl = daily_pnl[idx].copy()
        dpnl = np.maximum(dpnl, -FUNDED_DAILY_LOSS)

        equity[alive] += dpnl[alive]
        peak[alive] = np.maximum(peak[alive], equity[alive])
        profit_since_payout[alive] += dpnl[alive]

        winning_days[alive & (dpnl >= WINNING_DAY_THRESHOLD)] += 1

        # Trailing DD bust
        dd = peak - equity
        busted = alive & (dd >= FUNDED_TRAILING_DD)
        alive[busted] = False

        # Payout check
        can_pay = (
            alive
            & (winning_days >= WINNING_DAYS_REQUIRED)
            & (profit_since_payout > 0)
            & (equity > 0)
        )
        if can_pay.any():
            payout_amt = np.minimum(PAYOUT_MAX, equity[can_pay] * PAYOUT_BAL_PCT)
            payout_amt = np.maximum(payout_amt, 0)
            total_extracted[can_pay] += payout_amt
            equity[can_pay] -= payout_amt
            payout_count[can_pay] += 1
            winning_days[can_pay] = 0
            profit_since_payout[can_pay] = 0.0
            first_time = can_pay & (first_payout_day == -1)
            first_payout_day[first_time] = day + 1

    got_payout = payout_count > 0
    pct = got_payout.sum() / N * 100

    return {
        "payout_pct": pct,
        "avg_days_to_payout": float(first_payout_day[got_payout].mean()) if got_payout.any() else 0,
        "avg_per_payout": float(total_extracted[got_payout].sum() / payout_count[got_payout].sum()) if got_payout.any() else 0,
        "avg_total_extracted": float(total_extracted[got_payout].mean()) if got_payout.any() else 0,
        "avg_payouts": float(payout_count[got_payout].mean()) if got_payout.any() else 0,
        "funded_bust_rate": float((~alive).sum() / N * 100),
    }


def run_funded_mc_coast(
    trades_df: pd.DataFrame,
    coast_threshold: float = 2000.0,
    coast_scale: float = 0.5,
    coast_daily_cap: Optional[float] = None,
    n_sims: int = FUNDED_SIMS,
    max_days: int = FUNDED_MAX_DAYS,
    seed: int = 42,
) -> Dict:
    """
    Funded MC with coast mode: once profit_since_payout exceeds coast_threshold,
    switch to safer trades (scaled PnL and/or capped daily PnL).

    coast_scale: multiply PnL by this when coasting (0.5 = 1 contract instead of 2)
    coast_daily_cap: if set, cap daily PnL at this value when coasting
    """
    if trades_df.empty:
        return {
            "payout_pct": 0, "avg_days_to_payout": 0, "avg_per_payout": 0,
            "avg_total_extracted": 0, "avg_payouts": 0, "funded_bust_rate": 0,
        }

    daily_pnl = trades_df.groupby("trading_day")["pnl"].sum().values
    rng = np.random.default_rng(seed)

    N = n_sims
    equity = np.zeros(N)
    peak = np.zeros(N)
    winning_days = np.zeros(N, dtype=int)
    profit_since_payout = np.zeros(N)
    total_extracted = np.zeros(N)
    payout_count = np.zeros(N, dtype=int)
    first_payout_day = np.full(N, -1, dtype=int)
    alive = np.ones(N, dtype=bool)

    for day in range(max_days):
        if not alive.any():
            break

        idx = rng.integers(0, len(daily_pnl), size=N)
        dpnl = daily_pnl[idx].copy().astype(float)
        dpnl = np.maximum(dpnl, -FUNDED_DAILY_LOSS)

        # Coast mode: scale down PnL for sims that exceeded threshold
        coasting = alive & (profit_since_payout >= coast_threshold)
        if coasting.any():
            dpnl[coasting] *= coast_scale
            if coast_daily_cap is not None:
                dpnl[coasting] = np.clip(dpnl[coasting], -FUNDED_DAILY_LOSS * coast_scale, coast_daily_cap)

        equity[alive] += dpnl[alive]
        peak[alive] = np.maximum(peak[alive], equity[alive])
        profit_since_payout[alive] += dpnl[alive]

        winning_days[alive & (dpnl >= WINNING_DAY_THRESHOLD)] += 1

        # Trailing DD bust
        dd = peak - equity
        busted = alive & (dd >= FUNDED_TRAILING_DD)
        alive[busted] = False

        # Payout check
        can_pay = (
            alive
            & (winning_days >= WINNING_DAYS_REQUIRED)
            & (profit_since_payout > 0)
            & (equity > 0)
        )
        if can_pay.any():
            payout_amt = np.minimum(PAYOUT_MAX, equity[can_pay] * PAYOUT_BAL_PCT)
            payout_amt = np.maximum(payout_amt, 0)
            total_extracted[can_pay] += payout_amt
            equity[can_pay] -= payout_amt
            payout_count[can_pay] += 1
            winning_days[can_pay] = 0
            profit_since_payout[can_pay] = 0.0
            first_time = can_pay & (first_payout_day == -1)
            first_payout_day[first_time] = day + 1

    got_payout = payout_count > 0
    pct = got_payout.sum() / N * 100

    return {
        "payout_pct": pct,
        "avg_days_to_payout": float(first_payout_day[got_payout].mean()) if got_payout.any() else 0,
        "avg_per_payout": float(total_extracted[got_payout].sum() / payout_count[got_payout].sum()) if got_payout.any() else 0,
        "avg_total_extracted": float(total_extracted[got_payout].mean()) if got_payout.any() else 0,
        "avg_payouts": float(payout_count[got_payout].mean()) if got_payout.any() else 0,
        "funded_bust_rate": float((~alive).sum() / N * 100),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Full Strategy Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_strategy(
    df: pd.DataFrame,
    strategy: Strategy,
    n_eval_sims: int = DEFAULT_SIMS,
    n_funded_sims: int = FUNDED_SIMS,
    df_1m: Optional[pd.DataFrame] = None,
) -> BacktestResult:
    """Full evaluation: backtest to trade stats to eval MC to funded MC to EV."""
    t0 = time.time()

    # Run backtest
    _, trades_df, _ = run_backtest(df, strategy, df_1m=df_1m)

    if trades_df.empty:
        console.print(f"  [red]{strategy.name}: no trades generated[/]")
        return BacktestResult(name=strategy.name, trades_df=trades_df)

    # Trade stats
    stats = compute_trade_stats(trades_df)

    # Calendar days: full span including weekends/holidays
    min_date = df["datetime"].dt.date.min()
    max_date = df["datetime"].dt.date.max()
    total_cal_days = (max_date - min_date).days + 1
    # Data days: days with actual market data (excludes weekends/holidays)
    total_data_days = df["datetime"].dt.date.nunique()

    # Eval MC
    eval_mc = run_eval_mc(
        trades_df, total_calendar_days=total_cal_days,
        total_data_days=total_data_days, n_sims=n_eval_sims,
    )

    # Funded MC
    funded_mc = run_funded_mc(trades_df, n_sims=n_funded_sims)

    # Full cycle EV: pass_rate * (funded_extracted - activation_fee) - (1 - pass_rate) * combine_cost
    pr = eval_mc["pass_rate"] / 100
    full_ev = pr * (funded_mc["avg_total_extracted"] - ACTIVATION_FEE) - (1 - pr) * COMBINE_COST

    elapsed = time.time() - t0

    result = BacktestResult(
        name=strategy.name,
        trades_df=trades_df,
        # Trade stats
        total_trades=stats["total_trades"],
        long_trades=stats["long_trades"],
        short_trades=stats["short_trades"],
        trading_days=stats["trading_days"],
        win_rate=stats["win_rate"],
        avg_win=stats["avg_win"],
        avg_loss=stats["avg_loss"],
        avg_pnl=stats["avg_pnl"],
        total_pnl=stats["total_pnl"],
        profit_factor=stats["profit_factor"],
        max_drawdown=stats["max_drawdown"],
        sharpe=stats["sharpe"],
        avg_duration_min=stats["avg_duration_min"],
        # Eval MC
        pass_rate=eval_mc["pass_rate"],
        avg_cal_days=eval_mc["avg_cal_days"],
        avg_trade_days=eval_mc["avg_trade_days"],
        avg_profit_on_pass=eval_mc["avg_profit_on_pass"],
        eval_ev=eval_mc["eval_ev"],
        # Funded MC
        payout_pct=funded_mc["payout_pct"],
        avg_days_to_payout=funded_mc["avg_days_to_payout"],
        avg_per_payout=funded_mc["avg_per_payout"],
        avg_total_extracted=funded_mc["avg_total_extracted"],
        avg_payouts=funded_mc["avg_payouts"],
        funded_bust_rate=funded_mc["funded_bust_rate"],
        # Full cycle
        full_cycle_ev=full_ev,
    )

    console.print(f"  [green]{strategy.name}[/] done in {elapsed:.1f}s "
                  f"({stats['total_trades']} trades, {eval_mc['pass_rate']:.1f}% pass)")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Combo Evaluation — Different Strategies for Eval vs Funded
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ComboResult:
    """Results from a combo evaluation (eval strategy + funded strategy)."""
    eval_strategy: str
    funded_strategy: str
    # Eval phase
    eval_trades: int = 0
    eval_win_rate: float = 0.0
    eval_avg_pnl: float = 0.0
    eval_profit_factor: float = 0.0
    pass_rate: float = 0.0
    avg_cal_days: float = 0.0
    eval_ev: float = 0.0
    # Funded phase
    funded_trades: int = 0
    funded_win_rate: float = 0.0
    funded_avg_pnl: float = 0.0
    funded_profit_factor: float = 0.0
    payout_pct: float = 0.0
    avg_days_to_payout: float = 0.0
    avg_per_payout: float = 0.0
    avg_total_extracted: float = 0.0
    avg_payouts: float = 0.0
    # Full cycle
    full_cycle_ev: float = 0.0


def evaluate_combo(
    df: pd.DataFrame,
    eval_strategy: Strategy,
    funded_strategy: Strategy,
    n_eval_sims: int = DEFAULT_SIMS,
    n_funded_sims: int = FUNDED_SIMS,
    df_1m: Optional[pd.DataFrame] = None,
) -> ComboResult:
    """
    Evaluate a combo: one strategy for passing eval, another for funded payouts.

    This models the real workflow: use a high-pass-rate strategy to get funded,
    then switch to a different strategy to extract payouts.
    """
    t0 = time.time()

    console.print(f"  Eval strategy:   [cyan]{eval_strategy.name}[/]")
    console.print(f"  Funded strategy: [cyan]{funded_strategy.name}[/]")

    # Run eval backtest
    _, eval_trades_df, _ = run_backtest(df, eval_strategy, df_1m=df_1m)
    if eval_trades_df.empty:
        console.print(f"  [red]Eval strategy produced no trades[/]")
        return ComboResult(eval_strategy=eval_strategy.name, funded_strategy=funded_strategy.name)

    eval_stats = compute_trade_stats(eval_trades_df)
    min_date = df["datetime"].dt.date.min()
    max_date = df["datetime"].dt.date.max()
    total_cal_days = (max_date - min_date).days + 1
    total_data_days = df["datetime"].dt.date.nunique()
    eval_mc = run_eval_mc(
        eval_trades_df, total_calendar_days=total_cal_days,
        total_data_days=total_data_days, n_sims=n_eval_sims,
    )

    # Run funded backtest (separate strategy)
    _, funded_trades_df, _ = run_backtest(df, funded_strategy, df_1m=df_1m)
    if funded_trades_df.empty:
        console.print(f"  [red]Funded strategy produced no trades[/]")
        return ComboResult(eval_strategy=eval_strategy.name, funded_strategy=funded_strategy.name)

    funded_stats = compute_trade_stats(funded_trades_df)
    funded_mc = run_funded_mc(funded_trades_df, n_sims=n_funded_sims)

    # Full cycle EV
    pr = eval_mc["pass_rate"] / 100
    full_ev = pr * (funded_mc["avg_total_extracted"] - ACTIVATION_FEE) - (1 - pr) * COMBINE_COST

    elapsed = time.time() - t0
    console.print(f"  Done in {elapsed:.1f}s | "
                  f"Pass={eval_mc['pass_rate']:.1f}% | "
                  f"Funded extracted=${funded_mc['avg_total_extracted']:,.0f} | "
                  f"Cycle EV=${full_ev:,.0f}")

    return ComboResult(
        eval_strategy=eval_strategy.name,
        funded_strategy=funded_strategy.name,
        eval_trades=eval_stats["total_trades"],
        eval_win_rate=eval_stats["win_rate"],
        eval_avg_pnl=eval_stats["avg_pnl"],
        eval_profit_factor=eval_stats["profit_factor"],
        pass_rate=eval_mc["pass_rate"],
        avg_cal_days=eval_mc["avg_cal_days"],
        eval_ev=eval_mc["eval_ev"],
        funded_trades=funded_stats["total_trades"],
        funded_win_rate=funded_stats["win_rate"],
        funded_avg_pnl=funded_stats["avg_pnl"],
        funded_profit_factor=funded_stats["profit_factor"],
        payout_pct=funded_mc["payout_pct"],
        avg_days_to_payout=funded_mc["avg_days_to_payout"],
        avg_per_payout=funded_mc["avg_per_payout"],
        avg_total_extracted=funded_mc["avg_total_extracted"],
        avg_payouts=funded_mc["avg_payouts"],
        full_cycle_ev=full_ev,
    )


def evaluate_all_combos(
    df: pd.DataFrame,
    n_eval_sims: int = DEFAULT_SIMS,
    n_funded_sims: int = FUNDED_SIMS,
    df_1m: Optional[pd.DataFrame] = None,
) -> List[ComboResult]:
    """Test every eval/funded strategy combination and rank by full cycle EV."""
    strategy_names = list(STRATEGIES.keys())
    combos = [(e, f) for e in strategy_names for f in strategy_names]

    console.print(f"\n[bold]Combo Testing:[/] {len(combos)} combinations "
                  f"({len(strategy_names)} eval x {len(strategy_names)} funded)\n")

    results = []
    for eval_name, funded_name in combos:
        console.print(f"\n[bold]Combo: {eval_name} (eval) + {funded_name} (funded)[/]")
        result = evaluate_combo(
            df,
            STRATEGIES[eval_name],
            STRATEGIES[funded_name],
            n_eval_sims=n_eval_sims,
            n_funded_sims=n_funded_sims,
            df_1m=df_1m,
        )
        results.append(result)

    results.sort(key=lambda r: r.full_cycle_ev, reverse=True)
    return results


def display_combo_results(results: List[ComboResult]):
    """Display combo results as a ranked table."""
    t = Table(
        title="Combo Results (ranked by Full Cycle EV)",
        show_lines=True, expand=True,
    )
    t.add_column("#", justify="center", width=4)
    t.add_column("Eval Strategy", width=22)
    t.add_column("Funded Strategy", width=22)
    t.add_column("Pass%", justify="right", width=7)
    t.add_column("Cal Days", justify="right", width=9)
    t.add_column("Payout%", justify="right", width=8)
    t.add_column("Extracted", justify="right", width=10)
    t.add_column("Payouts", justify="right", width=8)
    t.add_column("Cycle EV", justify="right", width=10)

    for i, r in enumerate(results):
        ev_style = "green" if r.full_cycle_ev > 0 else "red"
        same = "[dim](same)[/]" if r.eval_strategy == r.funded_strategy else ""
        t.add_row(
            str(i + 1),
            r.eval_strategy,
            f"{r.funded_strategy} {same}",
            f"{r.pass_rate:.1f}",
            f"{r.avg_cal_days:.1f}",
            f"{r.payout_pct:.1f}",
            f"${r.avg_total_extracted:,.0f}",
            f"{r.avg_payouts:.1f}",
            f"[{ev_style}]${r.full_cycle_ev:,.0f}[/]",
        )

    console.print(t)

    best = results[0]
    console.print(f"\n[bold green]Best combo: {best.eval_strategy} (eval) + "
                  f"{best.funded_strategy} (funded)[/]"
                  f" | Cycle EV: ${best.full_cycle_ev:,.0f}")


def display_combo_detail(result: ComboResult):
    """Display detailed combo result."""
    t = Table(show_header=False, expand=True, padding=(0, 2))
    t.add_column("Metric", style="bold", width=28)
    t.add_column("Value", justify="right", width=20)

    t.add_row("[cyan]-- EVAL PHASE --[/]", f"[cyan]{result.eval_strategy}[/]")
    t.add_row("Trades", str(result.eval_trades))
    t.add_row("Win Rate", f"{result.eval_win_rate:.1f}%")
    t.add_row("Avg PnL/Trade", f"${result.eval_avg_pnl:,.2f}")
    t.add_row("Profit Factor", f"{result.eval_profit_factor:.2f}")
    t.add_row("Pass Rate", f"[bold]{result.pass_rate:.1f}%[/]")
    t.add_row("Avg Calendar Days", f"{result.avg_cal_days:.1f}")
    t.add_row("Eval EV", f"${result.eval_ev:,.0f}")
    t.add_row("", "")
    t.add_row("[cyan]-- FUNDED PHASE --[/]", f"[cyan]{result.funded_strategy}[/]")
    t.add_row("Trades", str(result.funded_trades))
    t.add_row("Win Rate", f"{result.funded_win_rate:.1f}%")
    t.add_row("Avg PnL/Trade", f"${result.funded_avg_pnl:,.2f}")
    t.add_row("Profit Factor", f"{result.funded_profit_factor:.2f}")
    t.add_row("Funded Payout Rate", f"{result.payout_pct:.1f}%")
    t.add_row("Avg Days to First Payout", f"{result.avg_days_to_payout:.0f}")
    t.add_row("Avg Per Payout", f"${result.avg_per_payout:,.0f}")
    t.add_row("Avg Payouts", f"{result.avg_payouts:.1f}")
    t.add_row("Avg Total Extracted", f"${result.avg_total_extracted:,.0f}")
    t.add_row("", "")
    ev_style = "green" if result.full_cycle_ev > 0 else "red"
    t.add_row(
        f"[bold {ev_style}]Full Cycle EV[/]",
        f"[bold {ev_style}]${result.full_cycle_ev:,.0f}[/]",
    )

    title = f"{result.eval_strategy} (eval) + {result.funded_strategy} (funded)"
    console.print(Panel(t, title=f"[bold]{title}[/]", border_style="cyan"))


# ═══════════════════════════════════════════════════════════════════════════
# Grid Search
# ═══════════════════════════════════════════════════════════════════════════

def grid_search(
    df: pd.DataFrame,
    base_strategy: Strategy,
    param_grid: Dict[str, List],
    n_sims: int = 10_000,
    top_n: int = 10,
    df_1m: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Grid search over parameter combinations.

    param_grid example:
        {
            "stop_pts": [3, 5, 7],
            "trailing_stop_pts": [1, 2, 3, None],
            "min_dip_pts": [1.0, 1.5, 2.0],
        }

    Returns DataFrame ranked by full_cycle_ev.
    """
    # Separate strategy-level params from backtest params
    strategy_attrs = {f.name for f in Strategy.__dataclass_fields__.values()}

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))

    console.print(f"\n[bold]Grid Search:[/] {len(combos)} combinations")
    console.print(f"  Params: {keys}")

    min_date = df["datetime"].dt.date.min()
    max_date = df["datetime"].dt.date.max()
    total_cal_days = (max_date - min_date).days + 1
    total_data_days = df["datetime"].dt.date.nunique()

    volume_col = "volume" if "volume" in df.columns else None
    results = []

    for i, combo in enumerate(combos):
        combo_dict = dict(zip(keys, combo))
        label = " | ".join(f"{k}={v}" for k, v in combo_dict.items())

        # Build modified strategy
        s = Strategy(
            name=f"grid_{i}",
            description=label,
            strategy_type=base_strategy.strategy_type,
            stop_pts=base_strategy.stop_pts,
            target_pts=base_strategy.target_pts,
            contracts=base_strategy.contracts,
            session_start_min=base_strategy.session_start_min,
            session_end_min=base_strategy.session_end_min,
            session_tz=base_strategy.session_tz,
            max_trades_per_day=base_strategy.max_trades_per_day,
            trailing_stop_pts=base_strategy.trailing_stop_pts,
            long_only=base_strategy.long_only,
            params=dict(base_strategy.params),
        )

        # Apply overrides
        for k, v in combo_dict.items():
            if k in strategy_attrs and k not in ("name", "description", "params"):
                setattr(s, k, v)
            else:
                s.params[k] = v

        # Run backtest
        try:
            _, trades_df, _ = run_backtest(df, s, volume_col=volume_col, df_1m=df_1m)
        except Exception:
            continue

        if trades_df.empty or len(trades_df) < 10:
            continue

        stats = compute_trade_stats(trades_df)
        eval_mc = run_eval_mc(
            trades_df, total_calendar_days=total_cal_days,
            total_data_days=total_data_days, n_sims=n_sims,
        )
        funded_mc = run_funded_mc(trades_df, n_sims=min(n_sims, 5000))

        pr = eval_mc["pass_rate"] / 100
        full_ev = pr * (funded_mc["avg_total_extracted"] - ACTIVATION_FEE) - (1 - pr) * COMBINE_COST

        row = {
            "config": label,
            **combo_dict,
            "trades": stats["total_trades"],
            "trading_days": stats["trading_days"],
            "win_rate": stats["win_rate"],
            "avg_pnl": stats["avg_pnl"],
            "profit_factor": stats["profit_factor"],
            "pass_rate": eval_mc["pass_rate"],
            "cal_days": eval_mc["avg_cal_days"],
            "eval_ev": eval_mc["eval_ev"],
            "funded_extracted": funded_mc["avg_total_extracted"],
            "full_cycle_ev": full_ev,
        }
        results.append(row)

        if (i + 1) % 10 == 0 or i + 1 == len(combos):
            console.print(f"  [{i+1}/{len(combos)}] {label} to "
                          f"EV=${full_ev:,.0f} pass={eval_mc['pass_rate']:.1f}%")

    if not results:
        console.print("[red]No valid results from grid search[/]")
        return pd.DataFrame()

    res_df = pd.DataFrame(results).sort_values("full_cycle_ev", ascending=False)
    return res_df.head(top_n).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# Split Testing (In-Sample / Out-of-Sample)
# ═══════════════════════════════════════════════════════════════════════════

def split_test(
    df: pd.DataFrame,
    strategy: Strategy,
    split_ratio: float = 0.7,
    n_sims: int = DEFAULT_SIMS,
) -> Tuple[BacktestResult, BacktestResult]:
    """
    Split data into in-sample (training) and out-of-sample (validation).
    Returns (is_result, oos_result).
    """
    n = len(df)
    split_idx = int(n * split_ratio)
    df_is = df.iloc[:split_idx].copy()
    df_oos = df.iloc[split_idx:].copy()

    is_start = df_is["datetime"].min()
    is_end = df_is["datetime"].max()
    oos_start = df_oos["datetime"].min()
    oos_end = df_oos["datetime"].max()

    console.print(f"\n[bold]Split Test:[/] {strategy.name}")
    console.print(f"  In-sample:  {str(is_start)[:10]} to {str(is_end)[:10]} ({len(df_is):,} bars)")
    console.print(f"  Out-of-sample: {str(oos_start)[:10]} to {str(oos_end)[:10]} ({len(df_oos):,} bars)")

    console.print("\n  Running in-sample...")
    is_result = evaluate_strategy(df_is, strategy, n_eval_sims=n_sims)

    console.print("  Running out-of-sample...")
    oos_result = evaluate_strategy(df_oos, strategy, n_eval_sims=n_sims)

    return is_result, oos_result


# ═══════════════════════════════════════════════════════════════════════════
# Display / Reporting
# ═══════════════════════════════════════════════════════════════════════════

def display_result(result: BacktestResult):
    """Print a single strategy result as a rich panel."""
    t = Table(show_header=False, expand=True, padding=(0, 2))
    t.add_column("Metric", style="bold", width=28)
    t.add_column("Value", justify="right", width=20)

    t.add_row("Trades", f"{result.total_trades} ({result.long_trades}L / {result.short_trades}S)")
    t.add_row("Trading Days", str(result.trading_days))
    t.add_row("Win Rate", f"{result.win_rate:.1f}%")
    t.add_row("Avg Win", f"${result.avg_win:,.2f}")
    t.add_row("Avg Loss", f"${result.avg_loss:,.2f}")
    t.add_row("Avg PnL/Trade", f"${result.avg_pnl:,.2f}")
    t.add_row("Profit Factor", f"{result.profit_factor:.2f}")
    t.add_row("Sharpe (annualized)", f"{result.sharpe:.2f}")
    t.add_row("Max Drawdown", f"${result.max_drawdown:,.2f}")
    t.add_row("Avg Trade Duration", f"{result.avg_duration_min:.0f} min")
    t.add_row("", "")
    t.add_row("[cyan]Eval Pass Rate[/]", f"[bold]{result.pass_rate:.1f}%[/]")
    t.add_row("Avg Calendar Days to Pass", f"{result.avg_cal_days:.1f}")
    t.add_row("Avg Trading Days to Pass", f"{result.avg_trade_days:.1f}")
    t.add_row("Avg Profit on Pass", f"${result.avg_profit_on_pass:,.0f}")
    t.add_row("Eval EV", f"${result.eval_ev:,.0f}")
    t.add_row("", "")
    t.add_row("Funded Payout Rate", f"{result.payout_pct:.1f}%")
    t.add_row("Avg Days to First Payout", f"{result.avg_days_to_payout:.0f}")
    t.add_row("Avg Per Payout", f"${result.avg_per_payout:,.0f}")
    t.add_row("Avg Payouts Before Bust", f"{result.avg_payouts:.1f}")
    t.add_row("Avg Total Extracted", f"${result.avg_total_extracted:,.0f}")
    t.add_row("", "")
    ev_style = "green" if result.full_cycle_ev > 0 else "red"
    t.add_row(
        f"[bold {ev_style}]Full Cycle EV[/]",
        f"[bold {ev_style}]${result.full_cycle_ev:,.0f}[/]",
    )

    console.print(Panel(t, title=f"[bold]{result.name}[/]: {STRATEGIES.get(result.name, Strategy(name=result.name, description='', params={})).description}", border_style="cyan"))


def display_comparison(results: List[BacktestResult]):
    """Side-by-side comparison table of multiple strategies."""
    t = Table(title="Strategy Comparison", show_lines=True, expand=True)
    t.add_column("Metric", style="bold", width=24)
    for r in results:
        t.add_column(r.name, justify="right", width=18)

    rows = [
        ("Trades", lambda r: str(r.total_trades)),
        ("Win Rate", lambda r: f"{r.win_rate:.1f}%"),
        ("Avg PnL/Trade", lambda r: f"${r.avg_pnl:,.0f}"),
        ("Profit Factor", lambda r: f"{r.profit_factor:.2f}"),
        ("Sharpe", lambda r: f"{r.sharpe:.2f}"),
        ("Max Drawdown", lambda r: f"${r.max_drawdown:,.0f}"),
        ("", lambda r: ""),
        ("Eval Pass Rate", lambda r: f"{r.pass_rate:.1f}%"),
        ("Cal Days to Pass", lambda r: f"{r.avg_cal_days:.1f}"),
        ("Eval EV", lambda r: f"${r.eval_ev:,.0f}"),
        ("", lambda r: ""),
        ("Funded Payout %", lambda r: f"{r.payout_pct:.1f}%"),
        ("Avg Extracted", lambda r: f"${r.avg_total_extracted:,.0f}"),
        ("Avg Payouts", lambda r: f"{r.avg_payouts:.1f}"),
        ("", lambda r: ""),
        ("Full Cycle EV", lambda r: f"${r.full_cycle_ev:,.0f}"),
    ]

    for label, fn in rows:
        t.add_row(label, *[fn(r) for r in results])

    console.print(t)

    # Highlight best
    best = max(results, key=lambda r: r.full_cycle_ev)
    console.print(f"\n[bold green]Best strategy: {best.name}[/] "
                  f"(Full Cycle EV: ${best.full_cycle_ev:,.0f})")


def display_split_comparison(is_result: BacktestResult, oos_result: BacktestResult):
    """Show in-sample vs out-of-sample comparison."""
    t = Table(title="Split Test: In-Sample vs Out-of-Sample", show_lines=True, expand=True)
    t.add_column("Metric", style="bold", width=24)
    t.add_column("In-Sample", justify="right", width=18)
    t.add_column("Out-of-Sample", justify="right", width=18)
    t.add_column("Degradation", justify="right", width=14)

    def deg(is_val, oos_val):
        if is_val == 0:
            return "-"
        pct = (oos_val - is_val) / abs(is_val) * 100
        color = "green" if pct >= 0 else "red"
        return f"[{color}]{pct:+.0f}%[/]"

    rows = [
        ("Trades", is_result.total_trades, oos_result.total_trades),
        ("Win Rate", is_result.win_rate, oos_result.win_rate),
        ("Avg PnL/Trade", is_result.avg_pnl, oos_result.avg_pnl),
        ("Profit Factor", is_result.profit_factor, oos_result.profit_factor),
        ("Sharpe", is_result.sharpe, oos_result.sharpe),
        ("Eval Pass Rate", is_result.pass_rate, oos_result.pass_rate),
        ("Eval EV", is_result.eval_ev, oos_result.eval_ev),
        ("Full Cycle EV", is_result.full_cycle_ev, oos_result.full_cycle_ev),
    ]

    for label, is_val, oos_val in rows:
        if isinstance(is_val, float):
            t.add_row(label, f"{is_val:,.2f}", f"{oos_val:,.2f}", deg(is_val, oos_val))
        else:
            t.add_row(label, str(is_val), str(oos_val), deg(is_val, oos_val))

    console.print(t)

    # Overfitting warning
    if oos_result.full_cycle_ev < is_result.full_cycle_ev * 0.5:
        console.print("\n[bold red]WARNING: >50% EV degradation OOS — likely overfit[/]")
    elif oos_result.full_cycle_ev < is_result.full_cycle_ev * 0.8:
        console.print("\n[yellow]CAUTION: >20% EV degradation OOS — monitor closely[/]")
    else:
        console.print("\n[green]OOS performance holds — strategy looks robust[/]")


def display_grid_results(grid_df: pd.DataFrame):
    """Display grid search results as a ranked table."""
    t = Table(title="Grid Search Results (ranked by Full Cycle EV)", show_lines=True, expand=True)
    t.add_column("#", justify="center", width=4)
    t.add_column("Config", width=40)
    t.add_column("Trades", justify="right", width=8)
    t.add_column("WR%", justify="right", width=7)
    t.add_column("PF", justify="right", width=6)
    t.add_column("Pass%", justify="right", width=7)
    t.add_column("Cal Days", justify="right", width=9)
    t.add_column("Funded $", justify="right", width=10)
    t.add_column("Cycle EV", justify="right", width=10)

    for i, row in grid_df.iterrows():
        ev_style = "green" if row["full_cycle_ev"] > 0 else "red"
        t.add_row(
            str(i + 1),
            row["config"],
            str(int(row["trades"])),
            f"{row['win_rate']:.1f}",
            f"{row['profit_factor']:.2f}",
            f"{row['pass_rate']:.1f}",
            f"{row['cal_days']:.1f}",
            f"${row['funded_extracted']:,.0f}",
            f"[{ev_style}]${row['full_cycle_ev']:,.0f}[/]",
        )

    console.print(t)


# ═══════════════════════════════════════════════════════════════════════════
# Main CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Topstep NQ Backtesting Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--strategy", "-s", help="Run a single strategy by name")
    parser.add_argument("--grid", action="store_true", help="Run grid search")
    parser.add_argument("--split", action="store_true", help="Run split test (IS/OOS)")
    parser.add_argument("--combo", action="store_true", help="Test all eval/funded strategy combinations")
    parser.add_argument("--coast", action="store_true", help="Test funded coast mode variants")
    parser.add_argument("--eval-strategy", help="Eval strategy for combo mode")
    parser.add_argument("--funded-strategy", help="Funded strategy for combo mode")
    parser.add_argument("--sims", type=int, default=DEFAULT_SIMS, help="Number of MC sims")
    parser.add_argument("--data", default=DATA_PATH, help="Path to data file")
    args = parser.parse_args()

    console.print(Panel(
        "[bold]Topstep NQ Backtesting Suite[/]\n"
        "Pessimistic trailing stop | Monte Carlo eval + funded | Full cycle EV",
        border_style="cyan",
    ))

    console.print("Loading data...")
    df, df_1m = load_data(args.data)
    date_range = f"{str(df['datetime'].min())[:10]} to {str(df['datetime'].max())[:10]}"
    bars_label = f"{len(df):,} 5m bars"
    if df_1m is not None:
        bars_label += f" ({len(df_1m):,} 1m bars for stops)"
    console.print(f"  {bars_label} | {date_range}\n")

    # ── Grid Search ──────────────────────────────────────────────────
    if args.grid:
        base = STRATEGIES.get("vwap_mr")
        if not base:
            console.print("[red]No base strategy 'vwap_mr' registered[/]")
            return

        param_grid = {
            "stop_pts": [3.0, 5.0, 7.0],
            "trailing_stop_pts": [1.0, 2.0, 3.0, None],
            "min_dip_pts": [1.0, 1.5, 2.0],
            "min_down_bar_pts": [0.25, 0.5, 1.0],
        }

        grid_df = grid_search(df, base, param_grid, n_sims=args.sims, top_n=15, df_1m=df_1m)
        if not grid_df.empty:
            display_grid_results(grid_df)
        return

    # ── Split Test ───────────────────────────────────────────────────
    if args.split:
        strategy_name = args.strategy or "vwap_mr"
        if strategy_name not in STRATEGIES:
            console.print(f"[red]Unknown strategy: {strategy_name}[/]")
            console.print(f"Available: {', '.join(STRATEGIES.keys())}")
            return

        is_result, oos_result = split_test(df, STRATEGIES[strategy_name], n_sims=args.sims)
        display_split_comparison(is_result, oos_result)
        return

    # ── Coast Mode Test ───────────────────────────────────────────────
    if args.coast:
        strategy_name = args.strategy or "vwap_mr_cd180_cap1600"
        if strategy_name not in STRATEGIES:
            console.print(f"[red]Unknown strategy: {strategy_name}[/]")
            return

        console.print(f"[bold]Coast Mode Test — base strategy: {strategy_name}[/]\n")
        strat = STRATEGIES[strategy_name]
        _, trades_df, _ = run_backtest(df, strat, df_1m=df_1m)

        if trades_df.empty:
            console.print("[red]No trades generated[/]")
            return

        # Eval MC for full cycle EV calc
        min_date = df["datetime"].dt.date.min()
        max_date = df["datetime"].dt.date.max()
        total_cal_days = (max_date - min_date).days + 1
        total_data_days = df["datetime"].dt.date.nunique()
        eval_mc = run_eval_mc(
            trades_df, total_calendar_days=total_cal_days,
            total_data_days=total_data_days, n_sims=args.sims,
        )
        pr = eval_mc["pass_rate"] / 100

        # Define coast variants to test
        coast_configs = [
            {"label": "No coast (baseline)",        "threshold": 999999, "scale": 1.0,  "cap": None},
            {"label": "Coast@$1500, 1 contract",    "threshold": 1500,   "scale": 0.5,  "cap": None},
            {"label": "Coast@$1500, 1ct + $400 cap","threshold": 1500,   "scale": 0.5,  "cap": 400},
            {"label": "Coast@$2000, 1 contract",    "threshold": 2000,   "scale": 0.5,  "cap": None},
            {"label": "Coast@$2000, 1ct + $400 cap","threshold": 2000,   "scale": 0.5,  "cap": 400},
            {"label": "Coast@$1000, 1 contract",    "threshold": 1000,   "scale": 0.5,  "cap": None},
            {"label": "Coast@$1000, 1ct + $400 cap","threshold": 1000,   "scale": 0.5,  "cap": 400},
            {"label": "Coast@$1500, 1ct + $200 cap","threshold": 1500,   "scale": 0.5,  "cap": 200},
            {"label": "Coast@$2000, 1ct + $200 cap","threshold": 2000,   "scale": 0.5,  "cap": 200},
        ]

        best_ev = -float("inf")
        best_label = ""
        rows = []

        for i, cfg in enumerate(coast_configs):
            mc = run_funded_mc_coast(
                trades_df,
                coast_threshold=cfg["threshold"],
                coast_scale=cfg["scale"],
                coast_daily_cap=cfg["cap"],
                n_sims=10000,
            )
            full_ev = pr * (mc["avg_total_extracted"] - ACTIVATION_FEE) - (1 - pr) * COMBINE_COST
            if full_ev > best_ev:
                best_ev = full_ev
                best_label = cfg["label"]
            rows.append((cfg["label"], mc, full_ev))

        # Print as a clean list
        console.print("[bold]Funded Coast Mode Comparison[/]\n")
        console.print(f"  {'Config':<34} {'Pay%':>6} {'Days':>5} {'$/Pay':>8} {'#Pay':>5} {'Extracted':>11} {'Bust%':>7} {'Cycle EV':>10}")
        console.print(f"  {'-'*34} {'-'*6} {'-'*5} {'-'*8} {'-'*5} {'-'*11} {'-'*7} {'-'*10}")
        for label, mc, full_ev in rows:
            ev_style = "[bold green]" if full_ev == best_ev else ""
            ev_end = "[/]" if ev_style else ""
            console.print(
                f"  {label:<34} {mc['payout_pct']:>5.1f}% {mc['avg_days_to_payout']:>5.0f} "
                f"${mc['avg_per_payout']:>6,.0f} {mc['avg_payouts']:>5.1f} "
                f"${mc['avg_total_extracted']:>9,.0f} {mc['funded_bust_rate']:>6.1f}% "
                f"{ev_style}${full_ev:>8,.0f}{ev_end}"
            )

        console.print(f"\n[bold green]Best: {best_label} (Full Cycle EV: ${best_ev:,.0f})[/]")
        return

    # ── Combo Test ────────────────────────────────────────────────────
    if args.combo:
        if args.eval_strategy and args.funded_strategy:
            # Single combo pair
            eval_strat = STRATEGIES.get(args.eval_strategy)
            funded_strat = STRATEGIES.get(args.funded_strategy)
            if not eval_strat:
                console.print(f"[red]Unknown eval strategy: {args.eval_strategy}[/]")
                console.print(f"Available: {', '.join(STRATEGIES.keys())}")
                return
            if not funded_strat:
                console.print(f"[red]Unknown funded strategy: {args.funded_strategy}[/]")
                console.print(f"Available: {', '.join(STRATEGIES.keys())}")
                return
            result = evaluate_combo(df, eval_strat, funded_strat, n_eval_sims=args.sims, df_1m=df_1m)
            display_combo_detail(result)
        else:
            # All combos
            combo_results = evaluate_all_combos(df, n_eval_sims=args.sims, df_1m=df_1m)
            display_combo_results(combo_results)
            console.print()
            if combo_results:
                console.print("[bold]Best combo detail:[/]\n")
                display_combo_detail(combo_results[0])
        return

    # ── Single Strategy ──────────────────────────────────────────────
    if args.strategy:
        if args.strategy not in STRATEGIES:
            console.print(f"[red]Unknown strategy: {args.strategy}[/]")
            console.print(f"Available: {', '.join(STRATEGIES.keys())}")
            return

        result = evaluate_strategy(df, STRATEGIES[args.strategy], n_eval_sims=args.sims, df_1m=df_1m)
        display_result(result)
        return

    # ── Compare All Strategies ───────────────────────────────────────
    console.print(f"[bold]Running {len(STRATEGIES)} strategies...[/]\n")
    results = []
    for name, strategy in STRATEGIES.items():
        result = evaluate_strategy(df, strategy, n_eval_sims=args.sims, df_1m=df_1m)
        results.append(result)

    console.print()
    for r in results:
        display_result(r)
        console.print()

    if len(results) > 1:
        display_comparison(results)


if __name__ == "__main__":
    main()
