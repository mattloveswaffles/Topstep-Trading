"""
Test all 101 Formulaic Alphas (Kakushadze 2015) adapted for single-instrument
NQ futures trading. Each alpha produces a daily signal → long/flat.
Run 1 Monte Carlo sim per alpha for eval and funded phases.
"""

import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from engine.backtest import (
    monte_carlo_topstep,
    START_EQUITY,
    COMBINE_COST,
    PROFIT_TARGET,
    TRAILING_DRAWDOWN,
    DAILY_LOSS_LIMIT,
)

warnings.filterwarnings("ignore")
console = Console()

DATA_PATH = "data/nq_5m_stitched.parquet"
POINT_VALUE = 20.0
CONTRACTS = 2


# ═══════════════════════════════════════════════════════════════════════════
# Operator helpers (vectorized pandas)
# ═══════════════════════════════════════════════════════════════════════════

def delay(x: pd.Series, d: int = 1) -> pd.Series:
    return x.shift(d)

def delta(x: pd.Series, d: int = 1) -> pd.Series:
    return x - x.shift(d)

def ts_sum(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(d, min_periods=1).sum()

def ts_mean(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(d, min_periods=1).mean()

def ts_min(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(d, min_periods=1).min()

def ts_max(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(d, min_periods=1).max()

def ts_argmax(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(d, min_periods=1).apply(lambda s: s.argmax(), raw=True)

def ts_argmin(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(d, min_periods=1).apply(lambda s: s.argmin(), raw=True)

def ts_rank(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(d, min_periods=1).apply(lambda s: pd.Series(s).rank().iloc[-1] / len(s), raw=False)

def ts_stddev(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(d, min_periods=1).std()

def ts_corr(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    return x.rolling(d, min_periods=2).corr(y)

def ts_cov(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    return x.rolling(d, min_periods=2).cov(y)

def ts_product(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(d, min_periods=1).apply(np.prod, raw=True)

def decay_linear(x: pd.Series, d: int) -> pd.Series:
    weights = np.arange(1, d + 1, dtype=float)
    weights /= weights.sum()
    return x.rolling(d, min_periods=1).apply(lambda s: np.dot(s[-len(weights):], weights[-len(s):]) if len(s) <= d else np.dot(s, weights), raw=True)

def scale(x: pd.Series, a: float = 1.0) -> pd.Series:
    denom = x.abs().sum()
    return a * x / denom if denom != 0 else x * 0

def rank(x: pd.Series) -> pd.Series:
    """For single instrument: use rolling percentile rank over 252 days."""
    return ts_rank(x, 252)

def signed_power(x: pd.Series, a: float) -> pd.Series:
    return x.abs() ** a * np.sign(x)

def log(x: pd.Series) -> pd.Series:
    return np.log(x.clip(lower=1e-10))

def sign(x: pd.Series) -> pd.Series:
    return np.sign(x)

def adv(volume: pd.Series, d: int) -> pd.Series:
    return volume.rolling(d, min_periods=1).mean()


# ═══════════════════════════════════════════════════════════════════════════
# Load and prepare daily OHLCV
# ═══════════════════════════════════════════════════════════════════════════

def load_daily() -> pd.DataFrame:
    console.print("[bold cyan]Loading NQ 5m data and aggregating to daily...[/]")
    df = pd.read_parquet(DATA_PATH)
    # Handle datetime as index or column
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index(names=["datetime"])
    if "datetime" in df.columns:
        if df["datetime"].dt.tz is None:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df["date"] = df["datetime"].dt.date
    df = df.sort_values("datetime")

    daily = df.groupby("date").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    # Derived columns
    daily["returns"] = daily["close"].pct_change()
    daily["vwap"] = (daily["high"] + daily["low"] + daily["close"]) / 3.0
    daily["adv5"] = adv(daily["volume"], 5)
    daily["adv10"] = adv(daily["volume"], 10)
    daily["adv15"] = adv(daily["volume"], 15)
    daily["adv20"] = adv(daily["volume"], 20)
    daily["adv30"] = adv(daily["volume"], 30)
    daily["adv40"] = adv(daily["volume"], 40)
    daily["adv50"] = adv(daily["volume"], 50)
    daily["adv60"] = adv(daily["volume"], 60)
    daily["adv81"] = adv(daily["volume"], 81)
    daily["adv120"] = adv(daily["volume"], 120)
    daily["adv150"] = adv(daily["volume"], 150)
    daily["adv180"] = adv(daily["volume"], 180)

    console.print(f"  [dim]{len(daily)} trading days[/]")
    return daily


# ═══════════════════════════════════════════════════════════════════════════
# Alpha definitions — each returns a pd.Series signal
# ═══════════════════════════════════════════════════════════════════════════

def alpha001(d):
    c, r = d["close"], d["returns"]
    cond = r < 0
    x = np.where(cond, ts_stddev(r, 20), c)
    x = pd.Series(x, index=d.index)
    return rank(ts_argmax(signed_power(x, 2), 5)) - 0.5

def alpha002(d):
    return -1 * ts_corr(rank(delta(log(d["volume"]), 2)), rank((d["close"] - d["open"]) / d["open"]), 6)

def alpha003(d):
    return -1 * ts_corr(rank(d["open"]), rank(d["volume"]), 10)

def alpha004(d):
    return -1 * ts_rank(rank(d["low"]), 9)

def alpha005(d):
    return rank(d["open"] - ts_mean(d["vwap"], 10)) * (-1 * rank(d["close"] - d["vwap"]).abs())

def alpha006(d):
    return -1 * ts_corr(d["open"], d["volume"], 10)

def alpha007(d):
    cond = d["adv20"] < d["volume"]
    x1 = -1 * ts_rank(d["close"].diff(7).abs(), 60) * sign(delta(d["close"], 7))
    x2 = pd.Series(-1.0, index=d.index)
    return np.where(cond, x1, x2)

def alpha008(d):
    return -1 * rank(ts_sum(d["open"], 5) * ts_sum(d["returns"], 5) - delay(ts_sum(d["open"], 5) * ts_sum(d["returns"], 5), 10))

def alpha009(d):
    dc = delta(d["close"], 1)
    cond_pos = ts_min(dc, 5) > 0
    cond_neg = ts_max(dc, 5) < 0
    return np.where(cond_pos, dc, np.where(cond_neg, dc, -1 * dc))

def alpha010(d):
    dc = delta(d["close"], 1)
    cond_pos = ts_min(dc, 4) > 0
    cond_neg = ts_max(dc, 4) < 0
    return rank(np.where(cond_pos, dc, np.where(cond_neg, dc, -1 * dc)))

def alpha011(d):
    return (rank(ts_max(d["vwap"] - d["close"], 3)) + rank(ts_min(d["vwap"] - d["close"], 3))) * rank(delta(d["volume"], 3))

def alpha012(d):
    return sign(delta(d["volume"], 1)) * (-1 * delta(d["close"], 1))

def alpha013(d):
    return -1 * rank(ts_cov(rank(d["close"]), rank(d["volume"]), 5))

def alpha014(d):
    return -1 * rank(delta(d["returns"], 3)) * ts_corr(d["open"], d["volume"], 10)

def alpha015(d):
    return -1 * ts_sum(rank(ts_corr(rank(d["high"]), rank(d["volume"]), 3)), 3)

def alpha016(d):
    return -1 * rank(ts_cov(rank(d["high"]), rank(d["volume"]), 5))

def alpha017(d):
    return (-1 * rank(ts_rank(d["close"], 10))) * rank(delta(delta(d["close"], 1), 1)) * rank(ts_rank(d["volume"] / d["adv20"], 5))

def alpha018(d):
    return -1 * rank(ts_stddev(d["close"] - d["open"], 5).abs() + (d["close"] - d["open"]) + ts_corr(d["close"], d["open"], 10))

def alpha019(d):
    return (-1 * sign(d["close"] - delay(d["close"], 7) + delta(d["close"], 7))) * (1 + rank(1 + ts_sum(d["returns"], 250)))

def alpha020(d):
    return (-1 * rank(d["open"] - delay(d["high"], 1))) * rank(d["open"] - delay(d["close"], 1)) * rank(d["open"] - delay(d["low"], 1))

def alpha021(d):
    cond1 = (ts_mean(d["close"], 8) + ts_stddev(d["close"], 8)) < (ts_mean(d["close"], 2) / 2)
    cond2 = ts_mean(d["close"], 8) - ts_stddev(d["close"], 8) < 0
    cond3 = (1 < (d["volume"] / d["adv20"])) | ((d["volume"] / d["adv20"]) == 1)
    return np.where(cond1, -1, np.where(cond2, 1, np.where(cond3, 1, -1)))

def alpha022(d):
    return -1 * delta(ts_corr(d["high"], d["volume"], 5), 5) * rank(ts_stddev(d["close"], 20))

def alpha023(d):
    cond = ts_mean(d["high"], 20) / 20 < d["high"]
    return np.where(cond, -1 * delta(d["high"], 2), 0)

def alpha024(d):
    sma = ts_mean(d["close"], 100)
    cond = delta(sma, 100) / delay(d["close"], 100) <= 0.05
    return np.where(cond, -1 * (d["close"] - ts_min(d["close"], 100)), -1 * delta(d["close"], 3))

def alpha025(d):
    return rank(-1 * d["returns"] * d["adv20"] * d["vwap"] * (d["high"] - d["close"]))

def alpha026(d):
    return -1 * ts_max(ts_corr(ts_rank(d["volume"], 5), ts_rank(d["high"], 5), 5), 3)

def alpha027(d):
    x = rank(ts_mean(ts_corr(rank(d["volume"]), rank(d["vwap"]), 6), 2))
    return np.where(x > 0.5, -1, 1)

def alpha028(d):
    return scale(ts_corr(d["adv20"], d["low"], 5) + (d["high"] + d["low"]) / 2 - d["close"])

def alpha029(d):
    inner = scale(log(ts_sum(ts_min(rank(-1 * rank(delta(d["close"] - 1, 5))), 2), 1)), 1)
    return ts_min(rank(rank(inner)), 5) + ts_rank(delay(-1 * d["returns"], 6), 5)

def alpha030(d):
    dc = delta(d["close"], 1)
    return (1.0 - rank(sign(dc) + sign(delay(dc, 1)) + sign(delay(dc, 2)))) * ts_sum(d["volume"], 5) / ts_sum(d["volume"], 20)

def alpha031(d):
    p1 = rank(rank(rank(decay_linear(-1 * rank(rank(delta(d["close"], 10))), 10))))
    p2 = rank(-1 * delta(d["close"], 3))
    p3 = sign(scale(ts_corr(d["adv20"], d["low"], 12)))
    return p1 + p2 + p3

def alpha032(d):
    return scale(ts_mean(d["close"], 7) / 7 - d["close"]) + 20 * scale(ts_corr(d["vwap"], delay(d["close"], 5), 230))

def alpha033(d):
    return rank(-1 * (1 - d["open"] / d["close"]))

def alpha034(d):
    return rank((1 - rank(ts_stddev(d["returns"], 2) / ts_stddev(d["returns"], 5))) + (1 - rank(delta(d["close"], 1))))

def alpha035(d):
    return ts_rank(d["volume"], 32) * (1 - ts_rank(d["close"] + d["high"] - d["low"], 16)) * (1 - ts_rank(d["returns"], 32))

def alpha037(d):
    return rank(ts_corr(delay(d["open"] - d["close"], 1), d["close"], 200)) + rank(d["open"] - d["close"])

def alpha038(d):
    return -1 * rank(ts_rank(d["close"], 10)) * rank(d["close"] / d["open"])

def alpha039(d):
    return -1 * rank(delta(d["close"], 7) * (1 - rank(decay_linear(d["volume"] / d["adv20"], 9)))) * (1 + rank(ts_sum(d["returns"], 250)))

def alpha040(d):
    return -1 * rank(ts_stddev(d["high"], 10)) * ts_corr(d["high"], d["volume"], 10)

def alpha041(d):
    return (d["high"] * d["low"]) ** 0.5 - d["vwap"]

def alpha042(d):
    return rank(d["vwap"] - d["close"]) / rank(d["vwap"] + d["close"])

def alpha043(d):
    return ts_rank(d["volume"] / d["adv20"], 20) * ts_rank(-1 * delta(d["close"], 7), 8)

def alpha044(d):
    return -1 * ts_corr(d["high"], rank(d["volume"]), 5)

def alpha045(d):
    return -1 * (rank(ts_mean(delay(d["close"], 5), 20) / 20) * ts_corr(d["close"], d["volume"], 2) * rank(ts_corr(ts_sum(d["close"], 5), ts_sum(d["close"], 20), 2)))

def alpha046(d):
    dc10 = (delay(d["close"], 20) - delay(d["close"], 10)) / 10
    dc = (delay(d["close"], 10) - d["close"]) / 10
    cond = dc10 - dc > 0.25
    cond2 = dc10 - dc < 0
    return np.where(cond, -1, np.where(cond2, 1, -1 * (d["close"] - delay(d["close"], 1))))

def alpha047(d):
    return (rank(1 / d["close"]) * d["volume"] / d["adv20"]) * (d["high"] * rank(d["high"] - d["close"]) / (ts_sum(d["high"], 5) / 5)) - rank(d["vwap"] - delay(d["vwap"], 5))

def alpha049(d):
    dc10 = (delay(d["close"], 20) - delay(d["close"], 10)) / 10
    dc = (delay(d["close"], 10) - d["close"]) / 10
    cond = dc10 - dc < -0.1
    return np.where(cond, 1, -1 * (d["close"] - delay(d["close"], 1)))

def alpha050(d):
    return -1 * ts_max(rank(ts_corr(rank(d["volume"]), rank(d["vwap"]), 5)), 5)

def alpha051(d):
    dc10 = (delay(d["close"], 20) - delay(d["close"], 10)) / 10
    dc = (delay(d["close"], 10) - d["close"]) / 10
    cond = dc10 - dc < -0.05
    return np.where(cond, 1, -1 * (d["close"] - delay(d["close"], 1)))

def alpha052(d):
    return (-1 * ts_min(d["low"], 5) + delay(ts_min(d["low"], 5), 5)) * rank((ts_sum(d["returns"], 240) - ts_sum(d["returns"], 20)) / 220) * ts_rank(d["volume"], 5)

def alpha053(d):
    return -1 * delta((d["close"] - d["low"] - (d["high"] - d["close"])) / (d["close"] - d["low"]).clip(lower=0.001), 9)

def alpha054(d):
    return -1 * (d["low"] - d["close"]) * (d["open"] ** 5) / ((d["low"] - d["high"]).clip(upper=-0.001) * (d["close"] ** 5))

def alpha055(d):
    return -1 * ts_corr(rank((d["close"] - ts_min(d["low"], 12)) / (ts_max(d["high"], 12) - ts_min(d["low"], 12)).clip(lower=0.001)), rank(d["volume"]), 6)

def alpha060(d):
    inner = 2 * scale(rank((d["close"] - d["low"] - (d["high"] - d["close"])) / (d["high"] - d["low"]).clip(lower=0.001) * d["volume"]))
    return 0 - (1 * inner - scale(rank(ts_argmax(d["close"], 10))))

def alpha061(d):
    return rank(d["vwap"] - ts_min(d["vwap"], 16)) < rank(ts_corr(d["vwap"], d["adv180"], 18))

def alpha062(d):
    cond = rank(ts_corr(d["vwap"], ts_sum(d["adv20"], 22), 10)) < rank((rank(d["open"]) + rank(d["open"])) < (rank((d["high"] + d["low"]) / 2) + rank(d["high"])))
    return np.where(cond, 1.0, -1.0)

def alpha064(d):
    return np.where(
        rank(ts_corr(ts_sum(d["open"] * 0.178, 13), ts_sum(d["adv120"], 13), 17)) < rank(delta((d["high"] + d["low"]) / 2 * 0.178 + d["vwap"] * (1 - 0.178), 4)),
        -1.0, 1.0
    )

def alpha065(d):
    return np.where(
        rank(ts_corr((d["open"] * 0.00817) + d["vwap"] * (1 - 0.00817), ts_sum(d["adv60"], 9), 6)) < rank(d["open"] - ts_min(d["open"], 14)),
        -1.0, 1.0
    )

def alpha068(d):
    cond = ts_rank(ts_corr(rank(d["high"]), rank(d["adv15"]), 9), 14) < rank(delta((d["close"] * 0.518) + d["low"] * (1 - 0.518), 1))
    return np.where(cond, -1.0, 1.0)

def alpha072(d):
    return rank(decay_linear(ts_corr((d["high"] + d["low"]) / 2, d["adv40"], 9), 10)) / rank(decay_linear(ts_corr(ts_rank(d["vwap"], 4), ts_rank(d["volume"], 19), 7), 3))

def alpha073(d):
    p1 = rank(decay_linear(delta(d["vwap"], 5), 3))
    p2 = ts_rank(decay_linear((delta(d["open"] * 0.147 + d["low"] * (1 - 0.147), 2) / (d["open"] * 0.147 + d["low"] * (1 - 0.147)).clip(lower=0.001)) * -1, 3), 17)
    return np.where(p1 > p2, -1.0, 1.0)

def alpha074(d):
    cond = rank(ts_corr(d["close"], ts_sum(d["adv30"], 37), 15)) < rank(ts_corr(rank(d["high"] * 0.026 + d["vwap"] * (1 - 0.026)), rank(d["volume"]), 11))
    return np.where(cond, -1.0, 1.0)

def alpha075(d):
    cond = rank(ts_corr(d["vwap"], d["volume"], 4)) < rank(ts_corr(rank(d["low"]), rank(d["adv50"]), 12))
    return np.where(cond, 1.0, -1.0)

def alpha077(d):
    p1 = rank(decay_linear((d["high"] + d["low"]) / 2 + d["high"] - (d["vwap"] + d["high"]), 20))
    p2 = rank(decay_linear(ts_corr((d["high"] + d["low"]) / 2, d["adv40"], 3), 6))
    return np.minimum(p1, p2)

def alpha078(d):
    return rank(ts_corr(ts_sum(d["low"] * 0.352 + d["vwap"] * (1 - 0.352), 20), ts_sum(d["adv40"], 20), 7)) * rank(ts_corr(rank(d["vwap"]), rank(d["volume"]), 6))

def alpha081(d):
    return np.where(
        rank(log(ts_product(rank(rank(ts_corr(d["vwap"], ts_sum(d["adv10"], 50), 8))).clip(lower=1), 15))) < rank(ts_corr(rank(d["vwap"]), rank(d["volume"]), 5)),
        -1.0, 1.0
    )

def alpha083(d):
    return rank(delay((d["high"] - d["low"]) / (ts_mean(d["close"], 5) / 5), 2)) * rank(rank(d["volume"])) / ((d["high"] - d["low"]) / (ts_mean(d["close"], 5) / 5) / (d["vwap"] - d["close"]).clip(lower=0.001))

def alpha084(d):
    return signed_power(ts_rank(d["vwap"] - ts_max(d["vwap"], 15), 21), delta(d["close"], 5))

def alpha085(d):
    return rank(ts_corr((d["high"] * 0.877 + d["close"] * (1 - 0.877)), d["adv30"], 10)) ** rank(ts_corr(ts_rank((d["high"] + d["low"]) / 2, 4), ts_rank(d["volume"], 10), 7))

def alpha086(d):
    cond = ts_rank(ts_corr(d["close"], ts_sum(d["adv20"], 15), 6), 20) < rank(d["open"] + d["close"] - (d["vwap"] + d["open"]))
    return np.where(cond, -1.0, 1.0)

def alpha088(d):
    p1 = rank(decay_linear((rank(d["open"]) + rank(d["low"])) - (rank(d["high"]) + rank(d["close"])), 8))
    p2 = ts_rank(decay_linear(ts_corr(ts_rank(d["close"], 8), ts_rank(d["adv60"], 21), 8), 7), 3)
    return np.minimum(p1, p2)

def alpha092(d):
    p1 = ts_rank(decay_linear((d["high"] + d["low"]) / 2 + d["close"] < d["low"] + d["open"], 15), 19)
    p2 = ts_rank(decay_linear(ts_corr(rank(d["low"]), rank(d["adv30"]), 8), 7), 7)
    return np.minimum(p1, p2)

def alpha094(d):
    return np.where(
        rank(d["vwap"] - ts_min(d["vwap"], 12)) > ts_rank(ts_corr(ts_rank(d["vwap"], 20), ts_rank(d["adv60"], 4), 18), 3),
        -1.0, 1.0
    )

def alpha095(d):
    cond = rank(d["open"] - ts_min(d["open"], 12)) < ts_rank(rank(ts_corr(ts_sum((d["high"] + d["low"]) / 2, 20), ts_sum(d["adv40"], 20), 14)), 12)
    return np.where(cond, 1.0, -1.0)

def alpha096(d):
    p1 = ts_rank(decay_linear(ts_corr(rank(d["vwap"]), rank(d["volume"]), 4), 4), 8)
    p2 = ts_rank(decay_linear(ts_argmax(ts_corr(ts_rank(d["close"], 7), ts_rank(d["adv60"], 4), 4), 13), 14), 13)
    return np.where(p1 > p2, -1.0, 1.0) * -1

def alpha098(d):
    return rank(decay_linear(ts_corr(d["vwap"], ts_sum(d["adv5"], 26), 5), 7)) - rank(decay_linear(ts_rank(ts_argmin(ts_corr(rank(d["open"]), rank(d["adv15"]), 21), 9), 7), 8))

def alpha099(d):
    cond = rank(ts_corr(ts_sum((d["high"] + d["low"]) / 2, 20), ts_sum(d["adv60"], 20), 9)) < rank(ts_corr(d["low"], d["volume"], 6))
    return np.where(cond, -1.0, 1.0)

def alpha101(d):
    return (d["close"] - d["open"]) / ((d["high"] - d["low"]).clip(lower=0.001) + 0.001)


# ═══════════════════════════════════════════════════════════════════════════
# Registry of all implemented alphas
# ═══════════════════════════════════════════════════════════════════════════

ALPHAS: Dict[str, Callable] = {
    "Alpha#1": alpha001, "Alpha#2": alpha002, "Alpha#3": alpha003,
    "Alpha#4": alpha004, "Alpha#5": alpha005, "Alpha#6": alpha006,
    "Alpha#7": alpha007, "Alpha#8": alpha008, "Alpha#9": alpha009,
    "Alpha#10": alpha010, "Alpha#11": alpha011, "Alpha#12": alpha012,
    "Alpha#13": alpha013, "Alpha#14": alpha014, "Alpha#15": alpha015,
    "Alpha#16": alpha016, "Alpha#17": alpha017, "Alpha#18": alpha018,
    "Alpha#19": alpha019, "Alpha#20": alpha020, "Alpha#21": alpha021,
    "Alpha#22": alpha022, "Alpha#23": alpha023, "Alpha#24": alpha024,
    "Alpha#25": alpha025, "Alpha#26": alpha026, "Alpha#27": alpha027,
    "Alpha#28": alpha028, "Alpha#29": alpha029, "Alpha#30": alpha030,
    "Alpha#31": alpha031, "Alpha#32": alpha032, "Alpha#33": alpha033,
    "Alpha#34": alpha034, "Alpha#35": alpha035,
    "Alpha#37": alpha037, "Alpha#38": alpha038, "Alpha#39": alpha039,
    "Alpha#40": alpha040, "Alpha#41": alpha041, "Alpha#42": alpha042,
    "Alpha#43": alpha043, "Alpha#44": alpha044, "Alpha#45": alpha045,
    "Alpha#46": alpha046, "Alpha#47": alpha047,
    "Alpha#49": alpha049, "Alpha#50": alpha050, "Alpha#51": alpha051,
    "Alpha#52": alpha052, "Alpha#53": alpha053, "Alpha#54": alpha054,
    "Alpha#55": alpha055,
    "Alpha#60": alpha060, "Alpha#61": alpha061, "Alpha#62": alpha062,
    "Alpha#64": alpha064, "Alpha#65": alpha065,
    "Alpha#68": alpha068,
    "Alpha#72": alpha072, "Alpha#73": alpha073, "Alpha#74": alpha074,
    "Alpha#75": alpha075,
    "Alpha#77": alpha077, "Alpha#78": alpha078,
    "Alpha#81": alpha081, "Alpha#83": alpha083, "Alpha#84": alpha084,
    "Alpha#85": alpha085, "Alpha#86": alpha086,
    "Alpha#88": alpha088,
    "Alpha#92": alpha092, "Alpha#94": alpha094, "Alpha#95": alpha095,
    "Alpha#96": alpha096,
    "Alpha#98": alpha098, "Alpha#99": alpha099,
    "Alpha#101": alpha101,
}


# ═══════════════════════════════════════════════════════════════════════════
# Signal → Trades → MC Sim
# ═══════════════════════════════════════════════════════════════════════════

def signal_to_trades(daily: pd.DataFrame, signal: pd.Series) -> pd.DataFrame:
    """Convert daily alpha signal to trade-level PnL.
    Signal > 0 → long next day (open→close). Signal <= 0 → flat.
    """
    # Use delay-1 signal (trade next day)
    sig = signal.shift(1)
    long_days = sig > 0

    trades = []
    for i in daily.index:
        if not long_days.iloc[i] or pd.isna(long_days.iloc[i]):
            continue
        row = daily.iloc[i]
        pnl = (row["close"] - row["open"]) * POINT_VALUE * CONTRACTS
        trades.append({
            "timestamp": row["date"],
            "pnl": pnl,
            "trading_day": str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
        })

    if not trades:
        return pd.DataFrame(columns=["timestamp", "pnl", "trading_day"])
    return pd.DataFrame(trades)


N_SIMS = 100


def test_alpha(daily: pd.DataFrame, name: str, func: Callable) -> Optional[Dict]:
    """Run one alpha: compute signal → trades → 100 MC sims. Return summary + per-attempt data."""
    try:
        sig = func(daily)
        if isinstance(sig, np.ndarray):
            sig = pd.Series(sig, index=daily.index)
        sig = sig.replace([np.inf, -np.inf], np.nan).fillna(0)

        trades_df = signal_to_trades(daily, sig)
        if trades_df.empty or len(trades_df) < 10:
            return None

        summary, attempts = monte_carlo_topstep(
            trades_df, n_simulations=N_SIMS, combine_cost=COMBINE_COST,
            day1_profit_cap=1400.0,
        )
        s = dict(zip(summary["Metric"], summary["Value"]))

        total_pnl = trades_df["pnl"].sum()
        n_days = trades_df["trading_day"].nunique()
        avg_daily = total_pnl / max(n_days, 1)
        win_rate = (trades_df["pnl"] > 0).mean()

        passed = attempts[attempts["Result"] == "PASS"]
        failed = attempts[attempts["Result"] == "FAIL"]
        n_passed = len(passed)
        n_failed = len(failed)
        fail_counts = failed["Rule Violations"].value_counts() if n_failed > 0 else pd.Series(dtype=int)

        return {
            "name": name,
            "n_trades": len(trades_df),
            "n_days": n_days,
            "total_pnl": total_pnl,
            "avg_daily_pnl": avg_daily,
            "win_rate": win_rate,
            "signal_long_pct": (sig > 0).mean(),
            "sims": N_SIMS,
            "passes": n_passed,
            "fails": n_failed,
            "pass_rate": n_passed / N_SIMS,
            "avg_profit_passed": float(passed["Profit"].mean()) if n_passed > 0 else 0.0,
            "avg_days_passed": float(passed["Trading Days"].mean()) if n_passed > 0 else 0.0,
            "avg_loss_failed": float(failed["Profit"].mean()) if n_failed > 0 else 0.0,
            "avg_days_failed": float(failed["Trading Days"].mean()) if n_failed > 0 else 0.0,
            "ev_per_combine": float(s.get("Expected Value Per Combine", 0)),
            "fail_trailing_dd": int(fail_counts.get("trailing_drawdown", 0)),
            "fail_target_not_reached": int(fail_counts.get("profit_target_not_reached", 0)),
            "fail_consistency": int(fail_counts.get("consistency_rule", 0)),
            "attempts_df": attempts,  # keep for per-attempt CSV
        }
    except Exception as e:
        return {"name": name, "error": str(e)}


def main():
    t0 = time.time()
    daily = load_daily()

    console.print(Panel(
        f"[bold green]TESTING {len(ALPHAS)} FORMULAIC ALPHAS[/]\n"
        f"{N_SIMS} Monte Carlo sims per alpha | {CONTRACTS} contracts | NQ futures",
        title="101 Alphas Test", border_style="green",
    ))

    results = []
    errors = []
    all_attempts = []  # Per-attempt rows for CSV

    for name, func in ALPHAS.items():
        console.print(f"  [dim]{name}...[/]", end=" ")
        r = test_alpha(daily, name, func)
        if r is None:
            console.print("[yellow]skipped (too few trades)[/]")
            continue
        if "error" in r:
            errors.append(r)
            console.print(f"[red]error: {r['error'][:60]}[/]")
        else:
            console.print(f"[green]{r['passes']}/{N_SIMS} passed[/]  PnL=${r['total_pnl']:,.0f}")
            # Collect per-attempt data
            att = r.pop("attempts_df").copy()
            att["alpha"] = name
            all_attempts.append(att)
            results.append(r)

    # Sort by pass rate then EV
    results.sort(key=lambda x: (x["pass_rate"], x["ev_per_combine"]), reverse=True)

    # ── Summary table ──
    console.print()
    console.print(Panel(
        f"[bold green]RESULTS[/] — {len(results)} alphas tested, {len(errors)} errors",
        border_style="green",
    ))

    # Compute per-day and per-week payout for each alpha
    # Assumes: eval takes avg_days_passed, then funded runs avg n_days before bust
    # Cycle = eval_days + funded_days. Revenue per cycle = avg_profit_passed
    EVAL_DAYS_ESTIMATE = 7.0  # VWAP MR avg days to pass
    COMBINE_FEE_COST = 50.0
    ACTIVATION_FEE_COST = 150.0

    for r in results:
        funded_days = r["n_days"]  # total trading days in backtest (proxy for funded survival)
        cycle_days = EVAL_DAYS_ESTIMATE + funded_days
        # Revenue per cycle: pass_rate * avg_profit - costs
        cycle_revenue = r["pass_rate"] * r["avg_profit_passed"] - COMBINE_FEE_COST - r["pass_rate"] * ACTIVATION_FEE_COST
        r["avg_per_day"] = cycle_revenue / max(cycle_days, 1)
        r["avg_per_week"] = r["avg_per_day"] * 5  # 5 trading days

    table = Table(title=f"All Alphas — {N_SIMS} Sims Each — Sorted by Pass Rate", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Alpha", style="cyan", width=12)
    table.add_column("Pass Rate", justify="right", style="green")
    table.add_column("Passes", justify="right")
    table.add_column("Avg Days", justify="right")
    table.add_column("Avg Profit", justify="right", style="yellow")
    table.add_column("EV/Combine", justify="right", style="bold")
    table.add_column("$/Day", justify="right", style="bold green")
    table.add_column("$/Week", justify="right", style="bold green")
    table.add_column("Win%", justify="right")
    table.add_column("Total PnL", justify="right")

    for i, r in enumerate(results, 1):
        ev_style = "green" if r["ev_per_combine"] > 0 else "red"
        pnl_style = "green" if r["total_pnl"] > 0 else "red"
        day_style = "green" if r["avg_per_day"] > 0 else "red"
        table.add_row(
            str(i),
            r["name"],
            f"{r['pass_rate']*100:.0f}%",
            f"{r['passes']}/{N_SIMS}",
            f"{r['avg_days_passed']:.1f}" if r['passes'] > 0 else "-",
            f"${r['avg_profit_passed']:,.0f}" if r['passes'] > 0 else "-",
            Text(f"${r['ev_per_combine']:,.0f}", style=ev_style),
            Text(f"${r['avg_per_day']:,.0f}", style=day_style),
            Text(f"${r['avg_per_week']:,.0f}", style=day_style),
            f"{r['win_rate']*100:.1f}%",
            Text(f"${r['total_pnl']:,.0f}", style=pnl_style),
        )
    console.print(table)

    if errors:
        console.print(f"\n[dim]{len(errors)} alphas errored: {', '.join(e['name'] for e in errors)}[/]")

    # ── Save CSVs ──
    # 1. Summary CSV (one row per alpha)
    summary_rows = [{k: v for k, v in r.items() if k != "attempts_df"} for r in results]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("alpha_summary.csv", index=False)
    console.print(f"\n[bold]Saved:[/] [cyan]alpha_summary.csv[/] ({len(summary_df)} alphas)")

    # 2. Per-attempt CSV (every sim attempt for every alpha)
    if all_attempts:
        attempts_df = pd.concat(all_attempts, ignore_index=True)
        attempts_df.to_csv("alpha_attempts.csv", index=False)
        console.print(f"[bold]Saved:[/] [cyan]alpha_attempts.csv[/] ({len(attempts_df)} rows)")

    elapsed = time.time() - t0
    console.print(f"\n[dim]Total time: {elapsed:.1f}s[/]")


if __name__ == "__main__":
    main()
