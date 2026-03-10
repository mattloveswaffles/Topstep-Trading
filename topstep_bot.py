"""
TopstepX Automated Trading Bot
═══════════════════════════════
Runs 24/7 monitoring the market and trading NQ futures automatically.

Phase 1 (EVAL):   VWAP Micro Mean-Reversion — pass the 50K combine
Phase 2 (FUNDED): Alpha#44 — extract payouts before bust

Supports multiple accounts simultaneously, auto-detects eval vs funded.
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ═════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════

load_dotenv()

console = Console()
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"bot_{date.today()}.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("topstep_bot")


class Phase(str, Enum):
    EVAL = "eval"
    FUNDED = "funded"


# ── Topstep Rules ────────────────────────────────────────────────────────
POINT_VALUE = 20.0            # NQ $20 per point
TICK_SIZE = 0.25              # NQ min tick
TICK_VALUE = 5.0              # $5 per tick
START_BALANCE = 50000.0       # Topstep 50K combine starting balance
EVAL_PROFIT_TARGET = 3000.0
EVAL_TRAILING_DD = 2000.0
DAILY_LOSS_LIMIT = 1000.0
FUNDED_TRAILING_DD = 2000.0
PAYOUT_MAX = 5000.0
PAYOUT_BAL_PCT = 0.50
WINNING_DAY_THRESHOLD = 150.0
WINNING_DAYS_REQUIRED = 5

# ── Strategy Config ──────────────────────────────────────────────────────
EVAL_CONFIG = {
    "stop_pts": 5.0,
    "target_pts": 20.0,
    "contracts": 2,
    "max_trades_per_day": 5,
    "session_start_min": 9 * 60 + 30,   # 9:30 AM ET
    "session_end_min": 15 * 60 + 45,    # 3:45 PM ET (full day)
    "min_dip_pts": 1.5,
    "min_down_bar_pts": 0.5,
    "rsi_oversold_max": 50.0,           # effectively disabled
    "rsi_overbought_min": 50.0,         # effectively disabled
    "day_bias_filter": False,           # no day bias — trade any day
    "trailing_stop_pts": 2.0,           # trail stop 2 pts behind best price
    "long_only": False,                 # trade both long and short
}

FUNDED_CONFIG = {
    "contracts": 2,
}


# ═════════════════════════════════════════════════════════════════════════
# ProjectX / TopstepX API Client
# Docs: https://gateway.docs.projectx.com/
# ═════════════════════════════════════════════════════════════════════════

class OrderType:
    LIMIT = 1
    MARKET = 2
    STOP = 4
    TRAILING_STOP = 5
    JOIN_BID = 6
    JOIN_ASK = 7


class OrderSide:
    BID = 0   # Buy
    ASK = 1   # Sell


class BarUnit:
    SECOND = 1
    MINUTE = 2
    HOUR = 3
    DAY = 4
    WEEK = 5
    MONTH = 6


class ProjectXAPI:
    """REST client for TopstepX (ProjectX Gateway API)."""

    BASE_URL = "https://api.topstepx.com"

    def __init__(self, api_key: str, username: str = ""):
        self.api_key = api_key
        self.username = username
        self.token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    # ── Auth ──────────────────────────────────────────────────────────

    def authenticate(self) -> bool:
        """Login with API key, get JWT token."""
        try:
            resp = self.session.post(
                f"{self.BASE_URL}/api/Auth/loginKey",
                json={
                    "userName": self.username,
                    "apiKey": self.api_key,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            # Try to find token in various possible response shapes
            token = (
                data.get("token")
                or data.get("access_token")
                or data.get("Token")
                or data.get("accessToken")
            )

            # Check for error
            if "errorCode" in data and data["errorCode"] != 0:
                log.error(f"Auth error: {data}")
                return False

            if not token:
                log.error(f"Auth response (no token found): {data}")
                return False

            self.token = token
            self.session.headers["Authorization"] = f"Bearer {self.token}"
            self.token_expiry = datetime.now(timezone.utc) + timedelta(hours=12)
            log.info("Authenticated with TopstepX")
            return True
        except Exception as e:
            log.error(f"Auth exception: {e}")
            return False

    def validate_session(self) -> bool:
        """Check if current token is still valid."""
        if not self.token:
            return False
        try:
            resp = self.session.post(f"{self.BASE_URL}/api/Auth/validateSession", json={})
            resp.raise_for_status()
            data = resp.json()
            return data.get("success", False)
        except Exception:
            return False

    def ensure_auth(self):
        """Re-auth if token expired or invalid."""
        if not self.token or (self.token_expiry and datetime.now(timezone.utc) > self.token_expiry):
            if not self.authenticate():
                raise ConnectionError("Failed to re-authenticate")

    def _check_response(self, data: dict, context: str = "") -> bool:
        """Check API response for errors."""
        if not data.get("success", True):
            err = data.get("errorMessage", "Unknown error")
            log.error(f"API error ({context}): {err} (code={data.get('errorCode')})")
            return False
        return True

    # ── Accounts ──────────────────────────────────────────────────────

    def get_accounts(self) -> List[Dict]:
        """Get all active trading accounts."""
        self.ensure_auth()
        resp = self.session.post(
            f"{self.BASE_URL}/api/Account/search",
            json={"onlyActiveAccounts": True},
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        # Wrapped response
        if not self._check_response(data, "get_accounts"):
            return []
        return data.get("accounts", data.get("data", []))

    # ── Contracts ─────────────────────────────────────────────────────

    def search_contracts(self, text: str = "NQ", live: bool = False) -> List[Dict]:
        """Search for futures contracts. Returns up to 20 results."""
        self.ensure_auth()
        resp = self.session.post(
            f"{self.BASE_URL}/api/Contract/search",
            json={"searchText": text, "live": live},
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        if not self._check_response(data, "search_contracts"):
            return []
        return data.get("contracts", data.get("data", []))

    def get_nq_contract(self) -> Optional[Dict]:
        """Find the active front-month NQ E-mini contract."""
        contracts = self.search_contracts("NQ")
        if not contracts:
            return None

        # Prefer active E-mini NQ (not micro MNQ)
        for c in contracts:
            cid = c.get("id", "")
            name = c.get("name", "")
            active = c.get("activeContract", True)
            # ENQ = E-mini Nasdaq, skip MNQ/micro
            is_emini = "ENQ" in cid or ("NQ" in name and "MNQ" not in name)
            if is_emini and active:
                return c

        # Fallback: first active NQ
        for c in contracts:
            if c.get("activeContract", True):
                return c
        return contracts[0]

    # ── Orders ────────────────────────────────────────────────────────

    def place_market_order(
        self,
        account_id: int,
        contract_id: str,
        side: int,           # OrderSide.BID (0) or OrderSide.ASK (1)
        size: int,
        stop_loss_ticks: Optional[int] = None,
        take_profit_ticks: Optional[int] = None,
    ) -> Dict:
        """Place a market order with optional bracket orders."""
        self.ensure_auth()
        payload = {
            "accountId": account_id,
            "contractId": contract_id,
            "type": OrderType.MARKET,
            "side": side,
            "size": size,
        }

        # Attach bracket orders (stop loss / take profit in ticks)
        if stop_loss_ticks is not None:
            payload["stopLossBracket"] = {
                "ticks": stop_loss_ticks,
                "type": OrderType.STOP,
            }
        if take_profit_ticks is not None:
            payload["takeProfitBracket"] = {
                "ticks": take_profit_ticks,
                "type": OrderType.LIMIT,
            }

        side_str = "BUY" if side == OrderSide.BID else "SELL"
        log.info(f"Placing order: {side_str} {size}x {contract_id} on account {account_id}")
        resp = self.session.post(f"{self.BASE_URL}/api/Order/place", json=payload)
        resp.raise_for_status()
        result = resp.json()

        if not self._check_response(result, "place_order"):
            raise RuntimeError(f"Order rejected: {result.get('errorMessage')}")

        log.info(f"Order placed: {result}")
        return result

    def cancel_order(self, order_id: int) -> Dict:
        """Cancel an open order."""
        self.ensure_auth()
        resp = self.session.post(
            f"{self.BASE_URL}/api/Order/cancel",
            json={"orderId": order_id},
        )
        resp.raise_for_status()
        return resp.json()

    def get_open_orders(self, account_id: int) -> List[Dict]:
        """Get open orders for an account."""
        self.ensure_auth()
        resp = self.session.post(
            f"{self.BASE_URL}/api/Order/searchOpen",
            json={"accountId": account_id},
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        return data.get("orders", data.get("data", []))

    # ── Positions ─────────────────────────────────────────────────────

    def get_positions(self, account_id: int) -> List[Dict]:
        """Get open positions for an account."""
        self.ensure_auth()
        resp = self.session.post(
            f"{self.BASE_URL}/api/Position/searchOpen",
            json={"accountId": account_id},
        )
        resp.raise_for_status()
        if not resp.text or not resp.text.strip():
            return []  # no positions
        data = resp.json()
        if isinstance(data, list):
            return data
        return data.get("positions", data.get("data", []))

    def get_trades(self, account_id: int, start_timestamp: str) -> List[Dict]:
        """Search trades for an account from a given start time."""
        self.ensure_auth()
        resp = self.session.post(
            f"{self.BASE_URL}/api/Trade/search",
            json={
                "accountId": account_id,
                "startTimestamp": start_timestamp,
            },
        )
        resp.raise_for_status()
        if not resp.text or not resp.text.strip():
            return []
        data = resp.json()
        return data.get("trades", [])

    # ── Market Data ───────────────────────────────────────────────────

    def get_bars(
        self,
        contract_id: str,
        unit: int = BarUnit.MINUTE,
        unit_number: int = 5,
        limit: int = 100,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve historical bars. Max 20,000 per request.

        Response bars have fields: t (timestamp), o (open), h (high), l (low), c (close), v (volume)
        """
        self.ensure_auth()

        # Default time range: last N days
        if not end_time:
            end_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if not start_time:
            # Estimate how far back we need based on limit and bar size
            days_back = max(7, (limit * unit_number) // (60 * 24) + 5)
            start_dt = datetime.now(timezone.utc) - timedelta(days=days_back)
            start_time = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        inner: Dict[str, Any] = {
            "contractId": contract_id,
            "live": False,
            "unit": unit,
            "unitNumber": unit_number,
            "limit": limit,
            "includePartialBar": True,
            "startTime": start_time,
            "endTime": end_time,
        }

        log.debug(f"Bars request: {inner}")
        resp = self.session.post(
            f"{self.BASE_URL}/api/History/retrieveBars",
            json=inner,
        )
        if resp.status_code != 200:
            log.error(f"Bars response {resp.status_code}: {resp.text[:500]}")
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list):
            return data
        if not self._check_response(data, "get_bars"):
            return []
        return data.get("bars", data.get("data", []))


# ═════════════════════════════════════════════════════════════════════════
# Account Tracker — tracks state per account
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class AccountState:
    account_id: int
    name: str = ""
    phase: Phase = Phase.EVAL
    contract_id: str = ""
    can_trade: bool = True

    # Account tracking
    equity: float = 0.0
    peak: float = 0.0
    daily_pnl: float = 0.0
    today: str = ""
    trading_days: int = 0

    # Position tracking
    position: int = 0             # current contracts held (0 = flat)
    entry_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    highest_price_since_entry: float = 0.0
    trades_today: int = 0

    # VWAP MR state (eval)
    bars_5m: list = field(default_factory=list)
    vwap: float = 0.0

    # Alpha#44 state (funded)
    alpha_signal: float = 0.0
    alpha_direction: str = "FLAT"   # "LONG" or "FLAT"
    entered_today: bool = False

    # Funded payout tracking
    winning_days: int = 0
    profit_since_payout: float = 0.0
    lifetime_extracted: float = 0.0
    total_payouts: int = 0
    total_extracted: float = 0.0

    # Status
    active: bool = True
    last_error: str = ""


# ═════════════════════════════════════════════════════════════════════════
# Alpha#44 Signal Computation
# ═════════════════════════════════════════════════════════════════════════

def compute_alpha044(daily_df: pd.DataFrame) -> float:
    """
    Alpha#44: -1 * correlation(high, rank(volume), 5)
    Returns today's signal value. > 0 means go LONG tomorrow.
    """
    if len(daily_df) < 10:
        return 0.0

    high = daily_df["high"].astype(float)
    volume = daily_df["volume"].astype(float)

    # rank = rolling percentile rank over 252 days
    vol_rank = volume.rolling(min(252, len(daily_df)), min_periods=1).apply(
        lambda s: pd.Series(s).rank().iloc[-1] / len(s), raw=False
    )

    # correlation(high, rank(volume), 5)
    corr = high.rolling(5, min_periods=2).corr(vol_rank)

    signal = -1 * corr
    val = signal.iloc[-1]
    return float(val) if not np.isnan(val) else 0.0


# ═════════════════════════════════════════════════════════════════════════
# VWAP MR Signal Computation (real-time on 5m bars)
# ═════════════════════════════════════════════════════════════════════════

def compute_vwap(bars: List[Dict]) -> float:
    """Compute VWAP from today's bars. Uses t/o/h/l/c/v field names."""
    if not bars:
        return 0.0
    cum_tp_vol = 0.0
    cum_vol = 0.0
    for b in bars:
        h = float(b.get("h", b.get("high", 0)))
        l = float(b.get("l", b.get("low", 0)))
        c = float(b.get("c", b.get("close", 0)))
        v = float(b.get("v", b.get("volume", 1)))
        tp = (h + l + c) / 3.0
        cum_tp_vol += tp * v
        cum_vol += v
    return cum_tp_vol / cum_vol if cum_vol > 0 else 0.0


def compute_rsi(closes: List[float], period: int = 14) -> float:
    """Compute RSI with Wilder EWM smoothing (matches backtest lm.py)."""
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    # Seed with simple average of first `period` deltas
    avg_gain = sum(max(d, 0) for d in deltas[:period]) / period
    avg_loss = sum(max(-d, 0) for d in deltas[:period]) / period
    # Wilder smoothing: ewm(alpha=1/period, adjust=False)
    alpha = 1.0 / period
    for d in deltas[period:]:
        avg_gain = alpha * max(d, 0) + (1 - alpha) * avg_gain
        avg_loss = alpha * max(-d, 0) + (1 - alpha) * avg_loss
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _bar_close(b: dict) -> float:
    return float(b.get("c", b.get("close", 0)))

def _bar_open(b: dict) -> float:
    return float(b.get("o", b.get("open", 0)))

def _bar_high(b: dict) -> float:
    return float(b.get("h", b.get("high", 0)))

def _bar_low(b: dict) -> float:
    return float(b.get("l", b.get("low", 0)))

def _bar_time(b: dict) -> str:
    return str(b.get("t", b.get("timestamp", "")))


def _compute_day_bias(bars: List[Dict], config: dict) -> dict:
    """Compute day bias features matching backtest _session_day_features.

    Returns dict with session_open, prev_day_close, first_window_move, gap_ok, window_ok.
    Bars should be today's UTC-date bars. We need prev day's close from daily bars
    or from earlier bars in the list.
    """
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    session_start = config["session_start_min"]  # 570 = 9:30
    first_window_end = session_start + config.get("day_bias_first_window_minutes", 15)

    session_open = None
    first_window_close = None

    for b in bars:
        dt = pd.to_datetime(b["t"])
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt_et = dt.astimezone(et)
        mins = dt_et.hour * 60 + dt_et.minute

        if mins >= session_start and session_open is None:
            session_open = _bar_open(b)
        if session_start <= mins <= first_window_end:
            first_window_close = _bar_close(b)

    first_window_move = None
    if session_open is not None and first_window_close is not None:
        first_window_move = first_window_close - session_open

    return {
        "session_open": session_open,
        "first_window_move": first_window_move,
    }


def check_vwap_mr_entry(bars: List[Dict], config: dict,
                         prev_day_close: float = None) -> bool:
    """
    Check VWAP Micro MR LONG entry conditions on current 5m bar.
    Matches backtest (lm.py run_topstep_nq) exactly:
      1. c < prev_bar_close  (price dropping)
      2. dip >= 1.5 pts below VWAP  (high_prob)
      3. bearish bar >= 0.5 pt  (high_prob)
      4. RSI < 50  (disabled by default)
      5. day_bias_filter if enabled (disabled by default)
    """
    if len(bars) < 2:
        return False

    current = bars[-1]
    prev_bar = bars[-2]
    price = _bar_close(current)
    o = _bar_open(current)
    prev_bar_close = _bar_close(prev_bar)
    vwap = compute_vwap(bars)

    # Base MR condition: below VWAP and close < previous bar's close
    if price >= prev_bar_close:
        return False

    # Dip below VWAP (high_prob: >= min_dip_pts, also covers min_vwap_distance=1.0)
    dip = vwap - price
    if dip < config["min_dip_pts"]:
        return False

    # Bearish bar (close < open, down >= min_down_bar_pts)
    if price >= o:
        return False
    down_bar = o - price
    if down_bar < config["min_down_bar_pts"]:
        return False

    # RSI oversold (Wilder smoothing)
    closes = [_bar_close(b) for b in bars]
    rsi = compute_rsi(closes, 14)
    if rsi >= config["rsi_oversold_max"]:
        return False

    # Day bias filter: gap up + first window move (matches backtest)
    if config.get("day_bias_filter", False):
        bias = _compute_day_bias(bars, config)
        session_open = bias["session_open"]
        first_window_move = bias["first_window_move"]

        # Gap: session_open - prev_day_close >= min_gap_pts
        min_gap = config.get("day_bias_min_gap_pts", 3.0)
        if prev_day_close is not None and session_open is not None:
            gap = session_open - prev_day_close
            if gap < min_gap:
                return False
        else:
            return False  # can't verify day bias without prev close

        # First window move >= threshold
        min_fw_move = config.get("day_bias_min_first_window_move_pts", 0.0)
        if first_window_move is not None:
            if first_window_move < min_fw_move:
                return False
        else:
            return False  # not enough bars in session yet

    return True


def check_vwap_mr_short_entry(bars: List[Dict], config: dict,
                               prev_day_close: float = None) -> bool:
    """
    Check VWAP Micro MR SHORT entry conditions (mirror of long).
    Matches backtest (lm.py run_topstep_nq) short logic:
      1. c > prev_bar_close  (price rising)
      2. spike >= 1.5 pts above VWAP  (high_prob)
      3. bullish bar >= 0.5 pt  (high_prob)
      4. RSI > 50  (disabled by default)
      5. day_bias_filter if enabled (disabled by default)
    """
    if len(bars) < 2:
        return False

    current = bars[-1]
    prev_bar = bars[-2]
    price = _bar_close(current)
    o = _bar_open(current)
    prev_bar_close = _bar_close(prev_bar)
    vwap = compute_vwap(bars)

    # Base MR condition: above VWAP and close > previous bar's close
    if price <= prev_bar_close:
        return False

    # Spike above VWAP (high_prob: >= min_dip_pts, covers min_vwap_distance=1.0)
    spike = price - vwap
    if spike < config["min_dip_pts"]:
        return False

    # Bullish bar (close > open, up >= min_down_bar_pts)
    if price <= o:
        return False
    up_bar = price - o
    if up_bar < config["min_down_bar_pts"]:
        return False

    # RSI overbought
    closes = [_bar_close(b) for b in bars]
    rsi = compute_rsi(closes, 14)
    if rsi <= config.get("rsi_overbought_min", 55.0):
        return False

    # Day bias filter: gap DOWN + first window move DOWN (bearish day)
    if config.get("day_bias_filter", False):
        bias = _compute_day_bias(bars, config)
        session_open = bias["session_open"]
        first_window_move = bias["first_window_move"]

        # Gap: session_open - prev_day_close <= -min_gap_pts (gap down)
        min_gap = config.get("day_bias_min_gap_pts", 3.0)
        if prev_day_close is not None and session_open is not None:
            gap = session_open - prev_day_close
            if gap > -min_gap:  # need gap DOWN
                return False
        else:
            return False

        # First window move <= -threshold (bearish)
        min_fw_move = config.get("day_bias_min_first_window_move_pts", 0.0)
        if first_window_move is not None:
            if first_window_move > -min_fw_move:  # need move DOWN
                return False
        else:
            return False

    return True


def pts_to_ticks(pts: float) -> int:
    """Convert points to NQ ticks (4 ticks per point)."""
    return int(pts / TICK_SIZE)


# ═════════════════════════════════════════════════════════════════════════
# Trading Engine
# ═════════════════════════════════════════════════════════════════════════

class TradingEngine:
    """Core trading engine managing multiple accounts."""

    POLL_INTERVAL = 5         # seconds between market checks

    def __init__(self, api: ProjectXAPI, state_file: str = "bot_state.json"):
        self.api = api
        self.state_file = state_file
        self.accounts: Dict[int, AccountState] = {}
        self.contract_id: str = ""
        self.contract_name: str = ""
        self.running = False
        self._daily_bars: pd.DataFrame = pd.DataFrame()

    # ── Setup ─────────────────────────────────────────────────────────

    def setup(self):
        """Initialize: auth, find accounts, find NQ contract."""
        log.info("Setting up trading engine...")

        if not self.api.authenticate():
            raise ConnectionError("Failed to authenticate with TopstepX")

        # Find NQ contract
        nq = self.api.get_nq_contract()
        if not nq:
            raise RuntimeError("Could not find NQ contract")
        self.contract_id = nq.get("id", "")
        self.contract_name = nq.get("name", "")
        log.info(f"NQ contract: {self.contract_id} ({self.contract_name})")

        # Load accounts
        raw_accounts = self.api.get_accounts()
        if not raw_accounts:
            raise RuntimeError("No trading accounts found")

        for acc in raw_accounts:
            acc_id = acc["id"]
            name = acc.get("name", "")
            can_trade = acc.get("canTrade", True)

            # Auto-detect phase from account name
            name_lower = name.lower()
            is_funded = any(
                kw in name_lower
                for kw in ["funded", "live", "express", "payout"]
            )
            phase = Phase.FUNDED if is_funded else Phase.EVAL

            self.accounts[acc_id] = AccountState(
                account_id=acc_id,
                name=name,
                phase=phase,
                contract_id=self.contract_id,
                can_trade=can_trade,
            )
            log.info(f"Account {acc_id}: '{name}' ->{phase.value} (canTrade={can_trade})")

        # Load saved state
        self._load_state()

        log.info(f"Engine ready: {len(self.accounts)} account(s), contract={self.contract_id}")

    # ── State Persistence ─────────────────────────────────────────────

    def _load_state(self):
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file) as f:
                saved = json.load(f)
            for acc_id_str, data in saved.get("accounts", {}).items():
                acc_id = int(acc_id_str)
                if acc_id in self.accounts:
                    acc = self.accounts[acc_id]
                    acc.equity = data.get("equity", 0)
                    acc.peak = data.get("peak", 0)
                    acc.trading_days = data.get("trading_days", 0)
                    acc.winning_days = data.get("winning_days", 0)
                    acc.profit_since_payout = data.get("profit_since_payout", 0)
                    acc.lifetime_extracted = data.get("lifetime_extracted", 0)
                    acc.total_payouts = data.get("total_payouts", 0)
                    acc.total_extracted = data.get("total_extracted", 0)
                    acc.today = data.get("today", "")
                    acc.daily_pnl = data.get("daily_pnl", 0)
                    acc.trades_today = data.get("trades_today", 0)
                    phase_str = data.get("phase", "eval")
                    acc.phase = Phase(phase_str)
            log.info("Loaded saved state")
        except Exception as e:
            log.warning(f"Could not load state: {e}")

    def _save_state(self):
        data = {"accounts": {}}
        for acc_id, acc in self.accounts.items():
            data["accounts"][str(acc_id)] = {
                "phase": acc.phase.value,
                "equity": acc.equity,
                "peak": acc.peak,
                "trading_days": acc.trading_days,
                "winning_days": acc.winning_days,
                "profit_since_payout": acc.profit_since_payout,
                "lifetime_extracted": acc.lifetime_extracted,
                "total_payouts": acc.total_payouts,
                "total_extracted": acc.total_extracted,
                "today": acc.today,
                "daily_pnl": acc.daily_pnl,
                "trades_today": acc.trades_today,
            }
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)

    # ── Time Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _now_et() -> datetime:
        """Current time in US/Eastern."""
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York"))

    @staticmethod
    def _is_market_hours(dt_et: datetime) -> bool:
        """NQ futures: Sun 6pm - Fri 5pm ET, with daily halt 5pm-6pm ET."""
        wd = dt_et.weekday()  # 0=Mon
        mins = dt_et.hour * 60 + dt_et.minute

        if wd == 5:                          # Saturday: closed
            return False
        if wd == 6:                          # Sunday: open from 6pm
            return mins >= 18 * 60
        if wd == 4 and mins >= 17 * 60:      # Friday: close at 5pm
            return False
        if 17 * 60 <= mins < 18 * 60:        # Mon-Thu: halt 5-6pm
            return False
        return True

    @staticmethod
    def _is_eval_session(dt_et: datetime) -> bool:
        """Eval trades during configured session window."""
        mins = dt_et.hour * 60 + dt_et.minute
        return EVAL_CONFIG["session_start_min"] <= mins <= EVAL_CONFIG["session_end_min"]

    @staticmethod
    def _is_funded_entry_window(dt_et: datetime) -> bool:
        """Alpha#44: enter at market open ~9:30-9:35 AM ET."""
        mins = dt_et.hour * 60 + dt_et.minute
        return 9 * 60 + 30 <= mins <= 9 * 60 + 35

    @staticmethod
    def _is_funded_exit_window(dt_et: datetime) -> bool:
        """Alpha#44: exit at close ~3:55-4:00 PM ET."""
        mins = dt_et.hour * 60 + dt_et.minute
        return 15 * 60 + 55 <= mins <= 16 * 60

    # ── API Sync (source of truth) ───────────────────────────────────

    def _sync_from_api(self, acc: AccountState):
        """Sync position, PnL, and equity from the platform API."""
        try:
            # Sync position from open positions
            positions = self.api.get_positions(acc.account_id)
            if positions:
                nq_pos = None
                for p in positions:
                    if p.get("contractId") == acc.contract_id:
                        nq_pos = p
                        break
                if nq_pos:
                    size = int(nq_pos.get("size", 0))
                    if acc.position == 0 and size != 0:
                        log.warning(
                            f"API sync: detected position {size} "
                            f"but bot thinks flat"
                        )
                    acc.position = size
                else:
                    if acc.position != 0:
                        log.warning(
                            "API sync: no NQ position, syncing to flat"
                        )
                    acc.position = 0
                    acc.entry_price = 0.0
            else:
                if acc.position != 0:
                    log.warning(
                        "API sync: no positions, syncing to flat"
                    )
                acc.position = 0
                acc.entry_price = 0.0

            # Sync equity from account balance
            accounts = self.api.get_accounts()
            for a in accounts:
                if a.get("id") == acc.account_id:
                    api_balance = a.get("balance")
                    if api_balance is not None:
                        bal = float(api_balance)
                        acc.equity = bal - START_BALANCE
                        acc.peak = max(acc.peak, acc.equity)
                    break

            # Sync daily PnL from trades (source of truth)
            today_start = datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT00:00:00Z"
            )
            trades = self.api.get_trades(acc.account_id, today_start)
            total_pnl = 0.0
            total_fees = 0.0
            trade_count = 0
            for t in trades:
                pnl = t.get("profitAndLoss")
                if pnl is not None:
                    total_pnl += float(pnl)
                    trade_count += 1
                total_fees += float(t.get("fees", 0))
            acc.daily_pnl = total_pnl - total_fees
            # Trade count: completed exits + 1 if currently in a position
            acc.trades_today = trade_count + (1 if acc.position != 0 else 0)

        except Exception as e:
            log.debug(f"API sync failed for acct {acc.account_id}: {e}")

    # ── Daily Reset ───────────────────────────────────────────────────

    def _check_new_day(self):
        """Reset daily state when the date rolls over."""
        today_str = str(self._now_et().date())
        for acc in self.accounts.values():
            if acc.today != today_str:
                if acc.today:  # not first run
                    if acc.daily_pnl >= WINNING_DAY_THRESHOLD:
                        acc.winning_days += 1
                    acc.trading_days += 1

                acc.today = today_str
                acc.daily_pnl = 0.0
                acc.trades_today = 0
                acc.entered_today = False
                # Reset day start balance for daily PnL tracking
                if hasattr(acc, '_day_start_balance'):
                    del acc._day_start_balance
                log.info(f"Account {acc.account_id}: new day {today_str}")

    # ── Build Daily Bars for Alpha Signal ─────────────────────────────

    @staticmethod
    def _futures_session_date(dt: datetime) -> date:
        """
        Map a bar timestamp to the correct futures trading session date.
        NQ session: 6pm ET (prev day) -> 5pm ET (current day).
        Bars from 6pm-midnight belong to the NEXT calendar date's session.
        Bars from midnight-5pm belong to the current calendar date's session.
        Bars from 5pm-6pm (halt) belong to the current calendar date's session.
        """
        from zoneinfo import ZoneInfo
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        et = dt.astimezone(ZoneInfo("America/New_York"))
        # If after 5:00 PM ET (17:00+), this bar belongs to tomorrow's session
        if et.hour >= 17:
            return (et + timedelta(days=1)).date()
        return et.date()

    def _refresh_daily_bars(self):
        """Fetch historical 5m bars and aggregate to daily OHLCV for alpha computation."""
        try:
            raw_bars = self.api.get_bars(
                self.contract_id,
                unit=BarUnit.MINUTE,
                unit_number=5,
                limit=20000,
            )
            if not raw_bars:
                log.warning("No bars returned from API")
                return

            rows = []
            for b in raw_bars:
                dt = pd.to_datetime(b["t"])
                # Use UTC calendar date to match backtest (alpha_test_101.py load_daily)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                rows.append({
                    "datetime": dt,
                    "date": dt.date(),
                    "open": float(b["o"]),
                    "high": float(b["h"]),
                    "low": float(b["l"]),
                    "close": float(b["c"]),
                    "volume": float(b["v"]),
                })

            df = pd.DataFrame(rows).sort_values("datetime")

            daily = df.groupby("date").agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
            ).reset_index()
            daily = daily.sort_values("date").reset_index(drop=True)
            self._daily_bars = daily
            log.info(f"Refreshed daily bars: {len(daily)} days")
        except Exception as e:
            log.error(f"Failed to refresh daily bars: {e}")

    # ── Fetch Today's 5m Bars ─────────────────────────────────────────

    def _get_today_bars(self, now_et: datetime) -> List[Dict]:
        """Fetch today's 5m bars for VWAP calculation.

        Uses UTC calendar date to match backtest (lm.py trading_day = dt.date).
        Fetches enough bars to cover a full day.
        """
        try:
            raw_bars = self.api.get_bars(
                self.contract_id,
                unit=BarUnit.MINUTE,
                unit_number=5,
                limit=300,
            )
            if not raw_bars:
                return []

            today_utc = datetime.now(timezone.utc).date()
            session_bars = []
            for b in raw_bars:
                dt = pd.to_datetime(b["t"])
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt.date() == today_utc:
                    session_bars.append(b)

            if not session_bars:
                return []
            # API returns bars in reverse chronological order — sort ascending
            session_bars.sort(key=lambda b: b.get("t", ""))
            return session_bars
        except Exception as e:
            log.error(f"Failed to get today's bars: {e}")
            return []

    # ── Core Trading Logic ────────────────────────────────────────────

    def _trade_eval(self, acc: AccountState):
        """VWAP Micro MR strategy for eval accounts."""
        now = self._now_et()

        # Always fetch bars so dashboard has data even outside session
        bars = self._get_today_bars(now)
        if bars:
            acc.bars_5m = bars
            acc.vwap = compute_vwap(bars)

        # Close position outside session window
        if not self._is_eval_session(now):
            if acc.position != 0:
                self._close_position(acc, "session_end")
            return

        # Daily loss limit — stop trading but don't bust
        if acc.daily_pnl <= -DAILY_LOSS_LIMIT:
            if acc.position != 0:
                self._close_position(acc, "daily_loss_cap")
            return

        # Max trades per day
        if acc.trades_today >= EVAL_CONFIG["max_trades_per_day"]:
            return

        if not bars:
            return

        current_price = _bar_close(bars[-1])

        # If in position, check stop/target (bracket orders handle this on server,
        # but we also track locally for state management)
        if acc.position != 0:
            trail_pts = EVAL_CONFIG.get("trailing_stop_pts")
            if acc.position > 0:
                # Long: trailing stop ratchets UP
                bar_high = _bar_high(bars[-1])
                acc.highest_price_since_entry = max(acc.highest_price_since_entry, bar_high)
                if trail_pts is not None:
                    trail_stop = acc.highest_price_since_entry - trail_pts
                    if trail_stop > acc.stop_price:
                        acc.stop_price = trail_stop
                        log.debug(f"Long trail stop updated: {acc.stop_price:.2f} (high={acc.highest_price_since_entry:.2f})")
                if current_price <= acc.stop_price:
                    self._close_position(acc, "stop_hit")
                elif current_price >= acc.target_price:
                    self._close_position(acc, "target_hit")
            else:
                # Short: trailing stop ratchets DOWN
                bar_low = _bar_low(bars[-1])
                acc.highest_price_since_entry = min(acc.highest_price_since_entry, bar_low)
                if trail_pts is not None:
                    trail_stop = acc.highest_price_since_entry + trail_pts
                    if trail_stop < acc.stop_price:
                        acc.stop_price = trail_stop
                        log.debug(f"Short trail stop updated: {acc.stop_price:.2f} (low={acc.highest_price_since_entry:.2f})")
                if current_price >= acc.stop_price:
                    self._close_position(acc, "stop_hit")
                elif current_price <= acc.target_price:
                    self._close_position(acc, "target_hit")
            return

        # Get previous day's close for day bias filter (matches backtest)
        prev_day_close = None
        if self._daily_bars is not None and len(self._daily_bars) >= 2:
            today_utc = datetime.now(timezone.utc).date()
            prev_days = self._daily_bars[self._daily_bars["date"] < today_utc]
            if not prev_days.empty:
                prev_day_close = float(prev_days.iloc[-1]["close"])

        # Check for new entry — long on bullish days, short on bearish days
        stop_ticks = pts_to_ticks(EVAL_CONFIG["stop_pts"])
        target_ticks = pts_to_ticks(EVAL_CONFIG["target_pts"])

        if check_vwap_mr_entry(bars, EVAL_CONFIG, prev_day_close=prev_day_close):
            self._open_position(
                acc,
                side=OrderSide.BID,
                contracts=EVAL_CONFIG["contracts"],
                entry_price=current_price,
                stop_pts=EVAL_CONFIG["stop_pts"],
                target_pts=EVAL_CONFIG["target_pts"],
                stop_ticks=-stop_ticks,
                target_ticks=target_ticks,
            )
        elif (not EVAL_CONFIG.get("long_only", True)
              and check_vwap_mr_short_entry(bars, EVAL_CONFIG, prev_day_close=prev_day_close)):
            self._open_position(
                acc,
                side=OrderSide.ASK,
                contracts=EVAL_CONFIG["contracts"],
                entry_price=current_price,
                stop_pts=EVAL_CONFIG["stop_pts"],
                target_pts=EVAL_CONFIG["target_pts"],
                stop_ticks=stop_ticks,
                target_ticks=-target_ticks,
            )

    def _trade_funded(self, acc: AccountState):
        """Alpha#44 strategy for funded accounts."""
        now = self._now_et()

        # Daily loss limit — stop trading but don't bust
        if acc.daily_pnl <= -DAILY_LOSS_LIMIT:
            if acc.position != 0:
                self._close_position(acc, "daily_loss_cap")
            return

        # Compute alpha signal once per day
        if not acc.entered_today and len(self._daily_bars) > 0:
            acc.alpha_signal = compute_alpha044(self._daily_bars)
            acc.alpha_direction = "LONG" if acc.alpha_signal > 0 else "FLAT"

        # Entry: at open 9:30 if signal is LONG
        if (acc.position == 0
                and acc.alpha_direction == "LONG"
                and not acc.entered_today
                and self._is_funded_entry_window(now)):
            bars = self._get_today_bars(now)
            if bars:
                price = _bar_close(bars[-1])
                self._open_position(
                    acc,
                    side=OrderSide.BID,
                    contracts=FUNDED_CONFIG["contracts"],
                    entry_price=price,
                )
                acc.entered_today = True

        # Exit: at close ~3:55 PM
        if acc.position != 0 and self._is_funded_exit_window(now):
            self._close_position(acc, "eod_exit")

    # ── Order Execution ───────────────────────────────────────────────

    def _open_position(
        self,
        acc: AccountState,
        side: int,
        contracts: int,
        entry_price: float,
        stop_pts: Optional[float] = None,
        target_pts: Optional[float] = None,
        stop_ticks: Optional[int] = None,
        target_ticks: Optional[int] = None,
    ):
        """Open a position via market order with optional brackets."""
        try:
            result = self.api.place_market_order(
                account_id=acc.account_id,
                contract_id=acc.contract_id,
                side=side,
                size=contracts,
                stop_loss_ticks=stop_ticks,
                take_profit_ticks=target_ticks,
            )

            acc.position = contracts if side == OrderSide.BID else -contracts
            acc.entry_price = entry_price
            if side == OrderSide.BID:  # long
                acc.stop_price = entry_price - stop_pts if stop_pts else 0.0
                acc.target_price = entry_price + target_pts if target_pts else 999999.0
            else:  # short
                acc.stop_price = entry_price + stop_pts if stop_pts else 999999.0
                acc.target_price = entry_price - target_pts if target_pts else 0.0
            acc.highest_price_since_entry = entry_price
            acc.trades_today += 1

            side_str = "BUY" if side == OrderSide.BID else "SELL"
            log.info(
                f"OPENED: Acct {acc.account_id} | {side_str} {contracts}x NQ @ ~{entry_price:.2f} | "
                f"SL={acc.stop_price:.2f} TP={acc.target_price:.2f}"
            )

            # Sync actual position from API
            time.sleep(0.5)  # brief wait for fill
            self._sync_from_api(acc)
        except Exception as e:
            log.error(f"Order failed on acct {acc.account_id}: {e}")
            acc.last_error = str(e)

    def _close_position(self, acc: AccountState, reason: str):
        """Flatten position via market order."""
        if acc.position == 0:
            return

        try:
            side = OrderSide.ASK if acc.position > 0 else OrderSide.BID
            contracts = abs(acc.position)
            side_str = "SELL" if acc.position > 0 else "BUY"

            self.api.place_market_order(
                account_id=acc.account_id,
                contract_id=acc.contract_id,
                side=side,
                size=contracts,
            )

            log.info(
                f"CLOSED: Acct {acc.account_id} | {reason} | "
                f"{side_str} {contracts}x NQ"
            )

            acc.position = 0
            acc.entry_price = 0.0
            acc.stop_price = 0.0
            acc.highest_price_since_entry = 0.0
            acc.target_price = 0.0

            # Sync real PnL/equity from API
            time.sleep(0.5)  # brief wait for fill
            self._sync_from_api(acc)

            # Check trailing DD bust
            drawdown = acc.peak - acc.equity
            max_dd = EVAL_TRAILING_DD if acc.phase == Phase.EVAL else FUNDED_TRAILING_DD
            if drawdown >= max_dd:
                acc.active = False
                log.warning(f"BUST: Acct {acc.account_id} | DD ${drawdown:,.2f} >= ${max_dd:,.2f}")

            # Check eval pass
            if (acc.phase == Phase.EVAL
                    and acc.equity >= EVAL_PROFIT_TARGET
                    and acc.trading_days >= 2):
                log.info(f"EVAL PASSED: Acct {acc.account_id} | Equity=${acc.equity:,.2f}")

        except Exception as e:
            log.error(f"Close failed on acct {acc.account_id}: {e}")
            acc.last_error = str(e)

    # ── Dashboard ─────────────────────────────────────────────────────

    def _build_dashboard(self) -> Table:
        """Build rich dashboard as a single table with market conditions."""
        from rich.columns import Columns
        from rich.layout import Layout

        now = self._now_et()
        market_open = self._is_market_hours(now)
        in_session = self._is_eval_session(now)
        mins = now.hour * 60 + now.minute

        # ── Market conditions from latest bars ──
        price_str = "-"
        vwap_str = "-"
        dip_str = "-"
        rsi_str = "-"
        bias_str = "-"
        entry_str = "[dim]NO[/]"
        dist_str = "-"

        # Grab from first active eval account
        for acc in self.accounts.values():
            if acc.bars_5m:
                price = _bar_close(acc.bars_5m[-1])
                vwap = acc.vwap
                price_str = f"{price:,.2f}"
                vwap_str = f"{vwap:,.2f}"

                if vwap > 0:
                    dip = vwap - price
                    dip_color = "green" if dip >= 2.5 else "red"
                    dip_str = f"[{dip_color}]{dip:+.2f} pts[/]"

                    # Distance to entry: how much price must drop to reach VWAP - 2.5
                    entry_level = vwap - EVAL_CONFIG["min_dip_pts"]
                    distance = price - entry_level
                    if distance > 0:
                        dist_str = f"[yellow]{distance:.2f} pts above[/] ({entry_level:,.2f})"
                    else:
                        dist_str = f"[green]IN ZONE[/] ({abs(distance):.2f} below)"

                closes = [_bar_close(b) for b in acc.bars_5m]
                rsi = compute_rsi(closes, 14)
                rsi_color = "green" if rsi < 45 else "red"
                rsi_str = f"[{rsi_color}]{rsi:.1f}[/]"

                # Day bias: gap + first window move (matches backtest)
                bias = _compute_day_bias(acc.bars_5m, EVAL_CONFIG)
                prev_day_close = None
                if self._daily_bars is not None and len(self._daily_bars) >= 2:
                    today_utc = datetime.now(timezone.utc).date()
                    prev_days = self._daily_bars[self._daily_bars["date"] < today_utc]
                    if not prev_days.empty:
                        prev_day_close = float(prev_days.iloc[-1]["close"])

                if EVAL_CONFIG.get("day_bias_filter", False):
                    gap = None
                    if bias["session_open"] is not None and prev_day_close is not None:
                        gap = bias["session_open"] - prev_day_close
                    min_gap = EVAL_CONFIG.get("day_bias_min_gap_pts", 3.0)
                    min_fw = EVAL_CONFIG.get("day_bias_min_first_window_move_pts", 0.0)
                    gap_bull = gap is not None and gap >= min_gap
                    fw_bull = bias["first_window_move"] is not None and bias["first_window_move"] >= min_fw
                    gap_bear = gap is not None and gap <= -min_gap
                    fw_bear = bias["first_window_move"] is not None and bias["first_window_move"] <= -min_fw
                    if gap_bull and fw_bull:
                        bias_str = f"[green]BULL[/] (gap {gap:+.1f})"
                    elif gap_bear and fw_bear:
                        bias_str = f"[yellow]BEAR[/] (gap {gap:+.1f})"
                    elif gap is not None:
                        bias_str = f"[red]FLAT[/] (gap {gap:+.1f})"
                    else:
                        bias_str = "[dim]N/A[/]"
                else:
                    bias_str = "[green]ANY[/] (no bias filter)"

                # Check all conditions — long or short
                long_met = check_vwap_mr_entry(acc.bars_5m, EVAL_CONFIG, prev_day_close=prev_day_close)
                short_met = (not EVAL_CONFIG.get("long_only", True)
                             and check_vwap_mr_short_entry(acc.bars_5m, EVAL_CONFIG, prev_day_close=prev_day_close))
                if long_met:
                    entry_str = "[bold green]LONG[/]"
                elif short_met:
                    entry_str = "[bold yellow]SHORT[/]"
                else:
                    entry_str = "[dim]NO[/]"
                break

        # ── Build output ──
        outer = Table.grid(padding=(0, 0))

        # Header row
        market_str = "[bold green]OPEN[/]" if market_open else "[bold red]CLOSED[/]"
        session_str = "[green]IN SESSION[/]" if in_session else "[dim]OUT[/]"
        # Position indicator from first account
        pos_str = "[dim]FLAT[/]"
        for acc in self.accounts.values():
            if acc.position > 0:
                pos_str = f"[bold green]{acc.position}x LONG @ {acc.entry_price:,.2f}[/]"
            elif acc.position < 0:
                pos_str = f"[bold red]{abs(acc.position)}x SHORT @ {acc.entry_price:,.2f}[/]"
            break

        header = Text.from_markup(
            f" [bold]TopstepX Bot[/]  |  {now.strftime('%Y-%m-%d %H:%M:%S ET')}  |  "
            f"Market: {market_str}  |  Session: {session_str}  |  "
            f"Position: {pos_str}  |  {self.contract_name}"
        )
        outer.add_row(header)
        outer.add_row(Text(""))

        # Market conditions panel
        cond = Table.grid(padding=(0, 4))
        cond.add_column(style="bold", width=16)
        cond.add_column(width=24)
        cond.add_column(style="bold", width=16)
        cond.add_column(width=24)
        cond.add_row("Price:", price_str, "VWAP:", vwap_str)
        cond.add_row("Dip from VWAP:", dip_str, "RSI(14):", rsi_str)
        cond.add_row("Day Bias:", bias_str, "Entry Signal:", entry_str)
        cond.add_row("Dist to Entry:", dist_str, "", "")

        outer.add_row(Panel(cond, title="Market Conditions", border_style="cyan", expand=True))
        outer.add_row(Text(""))

        # Accounts table
        acct_table = Table(show_lines=True, expand=True, show_header=True, padding=(0, 2))
        acct_table.add_column("Account", style="bold", ratio=4, no_wrap=True)
        acct_table.add_column("Phase", justify="center", ratio=1)
        acct_table.add_column("Equity", justify="right", ratio=2)
        acct_table.add_column("DD Left", justify="right", ratio=2)
        acct_table.add_column("Day PnL", justify="right", ratio=2)
        acct_table.add_column("Position", justify="center", ratio=2)
        acct_table.add_column("Trades", justify="center", ratio=1)
        acct_table.add_column("Status", justify="center", ratio=2)

        for acc in self.accounts.values():
            max_dd = EVAL_TRAILING_DD if acc.phase == Phase.EVAL else FUNDED_TRAILING_DD
            dd_left = max_dd - (acc.peak - acc.equity)

            phase_text = Text(acc.phase.value.upper())
            phase_text.stylize("green" if acc.phase == Phase.EVAL else "blue")

            equity_text = Text(f"${acc.equity:,.2f}")
            equity_text.stylize("green" if acc.equity >= 0 else "red")

            dd_text = Text(f"${dd_left:,.2f}")
            dd_text.stylize("green" if dd_left > 500 else "yellow" if dd_left > 200 else "red")

            day_text = Text(f"${acc.daily_pnl:+,.2f}")
            day_text.stylize("green" if acc.daily_pnl >= 0 else "red")

            pos = f"{acc.position}x LONG" if acc.position > 0 else f"{abs(acc.position)}x SHORT" if acc.position < 0 else "flat"
            pos_text = Text(pos)
            pos_text.stylize("cyan" if acc.position != 0 else "dim")

            if not acc.active:
                status = Text("BUST", style="bold red")
            elif acc.phase == Phase.EVAL and acc.equity >= EVAL_PROFIT_TARGET:
                status = Text("PASSED!", style="bold green")
            elif acc.position != 0:
                unrealized = ((_bar_close(acc.bars_5m[-1]) if acc.bars_5m else acc.entry_price) - acc.entry_price) * POINT_VALUE * abs(acc.position)
                ucolor = "green" if unrealized >= 0 else "red"
                status = Text(f"OPEN ${unrealized:+,.0f}", style=f"bold {ucolor}")
            elif not market_open:
                status = Text("MKT CLOSED", style="dim")
            elif not in_session and acc.phase == Phase.EVAL:
                status = Text("WAIT SESSION", style="dim")
            else:
                status = Text("SCANNING", style="yellow")

            # Funded extra row info
            extra = ""
            if acc.phase == Phase.FUNDED:
                extra = f"\nW-days: {acc.winning_days}/{WINNING_DAYS_REQUIRED} | Extracted: ${acc.total_extracted:,.0f}"

            acct_table.add_row(
                f"{acc.name} ({acc.account_id}){extra}",
                phase_text,
                equity_text,
                dd_text,
                day_text,
                pos_text,
                f"{acc.trades_today}/{EVAL_CONFIG['max_trades_per_day']}",
                status,
            )

        outer.add_row(acct_table)
        outer.add_row(Text(""))

        # Trade history from API
        try:
            today_start = datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT00:00:00Z"
            )
            for acc in self.accounts.values():
                trades = self.api.get_trades(
                    acc.account_id, today_start
                )
                if trades:
                    trade_table = Table(
                        show_lines=False, expand=True,
                        title="Today's Trades",
                        title_style="bold cyan",
                    )
                    trade_table.add_column("Time (ET)", ratio=2)
                    trade_table.add_column("Side", justify="center", ratio=1)
                    trade_table.add_column("Size", justify="center", ratio=1)
                    trade_table.add_column("Price", justify="right", ratio=2)
                    trade_table.add_column("PnL", justify="right", ratio=2)
                    trade_table.add_column("Fees", justify="right", ratio=1)

                    from zoneinfo import ZoneInfo
                    et = ZoneInfo("America/New_York")
                    # Only show trades with PnL (exits)
                    exit_trades = [
                        t for t in trades
                        if t.get("profitAndLoss") is not None
                    ]
                    for t in exit_trades:
                        ts = pd.to_datetime(t["creationTimestamp"])
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        ts_et = ts.astimezone(et)
                        time_str = ts_et.strftime("%H:%M:%S")
                        side_str = "BUY" if t["side"] == 0 else "SELL"
                        side_style = "green" if t["side"] == 0 else "red"
                        price_str = f"{t['price']:,.2f}"
                        pnl = float(t["profitAndLoss"])
                        fees = float(t.get("fees", 0))
                        net = pnl - fees
                        net_style = "green" if net >= 0 else "red"
                        trade_table.add_row(
                            time_str,
                            f"[{side_style}]{side_str}[/]",
                            str(t["size"]),
                            price_str,
                            f"[{net_style}]${pnl:+,.2f}[/]",
                            f"${fees:.2f}",
                        )
                    # Summary row
                    if exit_trades:
                        total_pnl = sum(
                            float(t["profitAndLoss"])
                            for t in exit_trades
                        )
                        total_fees = sum(
                            float(t.get("fees", 0))
                            for t in exit_trades
                        )
                        net_total = total_pnl - total_fees
                        ns = "green" if net_total >= 0 else "red"
                        trade_table.add_row(
                            "[bold]TOTAL[/]", "", "",  "",
                            f"[bold {ns}]${net_total:+,.2f}[/]",
                            f"${total_fees:.2f}",
                        )

                    outer.add_row(trade_table)
                break  # only first account
        except Exception:
            pass  # don't crash dashboard if trade fetch fails

        return outer

    # ── Main Loop ─────────────────────────────────────────────────────

    def run(self):
        """Main trading loop — runs 24/7."""
        self.running = True
        log.info("Bot started — running 24/7")

        self._refresh_daily_bars()
        last_daily_refresh = time.time()

        try:
            with Live(self._build_dashboard(), console=console, refresh_per_second=1, vertical_overflow="crop") as live:
                while self.running:
                    try:
                        now = self._now_et()
                        self._check_new_day()

                        # Refresh daily bars every 30 min
                        if time.time() - last_daily_refresh > 1800:
                            self._refresh_daily_bars()
                            last_daily_refresh = time.time()

                        # Re-auth if needed
                        self.api.ensure_auth()

                        if self._is_market_hours(now):
                            for acc in self.accounts.values():
                                if not acc.active or not acc.can_trade:
                                    continue
                                try:
                                    if acc.phase == Phase.EVAL:
                                        self._trade_eval(acc)
                                    else:
                                        self._trade_funded(acc)
                                except Exception as e:
                                    log.error(f"Trade error on acct {acc.account_id}: {e}")
                                    acc.last_error = str(e)

                        # Sync account state from API each cycle
                        for acc in self.accounts.values():
                            self._sync_from_api(acc)

                        live.update(self._build_dashboard())
                        self._save_state()
                        time.sleep(self.POLL_INTERVAL)

                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        log.error(f"Loop error: {e}\n{traceback.format_exc()}")
                        time.sleep(60)

        except KeyboardInterrupt:
            log.info("Shutting down — closing all positions...")
            for acc in self.accounts.values():
                if acc.position != 0:
                    log.info(f"Closing position on acct {acc.account_id}")
                    self._close_position(acc, "shutdown")
            self._save_state()
            log.info("Bot stopped.")


# ═════════════════════════════════════════════════════════════════════════
# Entry Point
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="TopstepX Automated Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python topstep_bot.py
  python topstep_bot.py --config config.json
  python topstep_bot.py --poll 15

Reads API_KEY from .env file (or --config / TOPSTEPX_API_KEY env var).
        """,
    )
    parser.add_argument("--config", help="Path to config JSON file")
    parser.add_argument("--poll", type=int, default=5, help="Poll interval in seconds (default: 5)")
    args = parser.parse_args()

    # Load API key: .env > config.json > env var
    api_key = os.environ.get("API_KEY") or os.environ.get("TOPSTEPX_API_KEY")
    username = os.environ.get("TOPSTEPX_USERNAME", "")
    account_phases = {}

    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        api_key = api_key or cfg.get("api_key")
        username = username or cfg.get("username", "")
        account_phases = cfg.get("accounts", {})

    if not api_key:
        console.print("[bold red]No API key found.[/]")
        console.print("Add API_KEY=your_key to .env file, or use --config.")
        console.print("\nTo get an API key:")
        console.print("  1. TopstepX dashboard ->Settings ->API Access")
        console.print("  2. Generate API key")
        sys.exit(1)

    # Build and run
    api = ProjectXAPI(api_key, username)
    engine = TradingEngine(api)
    engine.POLL_INTERVAL = args.poll

    console.print(Panel(
        "[bold]TopstepX Automated Trading Bot[/]\n\n"
        f"Eval:   VWAP Micro Mean-Reversion ({EVAL_CONFIG['session_start_min']//60}:{EVAL_CONFIG['session_start_min']%60:02d}"
        f"-{EVAL_CONFIG['session_end_min']//60}:{EVAL_CONFIG['session_end_min']%60:02d} ET)\n"
        "Funded: Alpha#44 (long at open, exit at close)\n\n"
        f"Poll interval: {args.poll}s\n"
        "Press Ctrl+C to stop (positions closed safely)",
        title="Starting",
        border_style="green",
    ))

    try:
        engine.setup()

        # Override phases from config
        for acc_id_str, acc_cfg in account_phases.items():
            acc_id = int(acc_id_str)
            if acc_id in engine.accounts:
                engine.accounts[acc_id].phase = Phase(acc_cfg.get("phase", "eval"))
                log.info(f"Config override: acct {acc_id} ->{acc_cfg.get('phase')}")

        engine.run()
    except Exception as e:
        log.error(f"Fatal error: {e}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
