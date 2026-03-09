"""
TopstepX Automated Trading Bot
═══════════════════════════════
Runs 24/7 monitoring the market and trading NQ futures automatically.

Phase 1 (EVAL):   VWAP Micro Mean-Reversion — pass the 50K combine
Phase 2 (FUNDED): Alpha#44 — extract payouts before bust

Supports multiple accounts simultaneously, auto-detects eval vs funded.

Usage:
  pip install requests websocket-client rich pandas numpy
  python topstep_bot.py --api-key YOUR_API_KEY
  python topstep_bot.py --config config.json
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ═════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════

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
POINT_VALUE = 20.0            # NQ point value
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
    "max_trades_per_day": 3,
    "session_start": "09:30",      # ET
    "session_end": "11:30",        # ET
    "min_dip_pts": 2.5,
    "min_down_bar_pts": 1.0,
    "rsi_oversold_max": 45.0,
    "day_bias_filter": True,
}

FUNDED_CONFIG = {
    "contracts": 2,
    "signal": "alpha044",          # signal > 0 → long next day open→close
}


# ═════════════════════════════════════════════════════════════════════════
# ProjectX / TopstepX API Client
# ═════════════════════════════════════════════════════════════════════════

class ProjectXAPI:
    """REST client for TopstepX (ProjectX Gateway API)."""

    BASE_URL = "https://gateway.topstepx.com"

    def __init__(self, api_key: str):
        self.api_key = api_key
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
                json={"apiKey": self.api_key},
            )
            resp.raise_for_status()
            data = resp.json()
            self.token = data.get("token") or data.get("access_token")
            if not self.token:
                log.error(f"Auth response missing token: {data}")
                return False
            self.session.headers["Authorization"] = f"Bearer {self.token}"
            self.token_expiry = datetime.now(timezone.utc) + timedelta(hours=12)
            log.info("Authenticated with TopstepX")
            return True
        except Exception as e:
            log.error(f"Auth failed: {e}")
            return False

    def ensure_auth(self):
        """Re-auth if token expired."""
        if not self.token or (self.token_expiry and datetime.now(timezone.utc) > self.token_expiry):
            if not self.authenticate():
                raise ConnectionError("Failed to re-authenticate")

    # ── Accounts ──────────────────────────────────────────────────────

    def get_accounts(self) -> List[Dict]:
        """Get all trading accounts."""
        self.ensure_auth()
        resp = self.session.post(f"{self.BASE_URL}/api/Account/search", json={})
        resp.raise_for_status()
        return resp.json()

    def get_account_balance(self, account_id: int) -> Dict:
        """Get account balance/equity info."""
        self.ensure_auth()
        resp = self.session.post(
            f"{self.BASE_URL}/api/Account/search",
            json={"accountId": account_id},
        )
        resp.raise_for_status()
        accounts = resp.json()
        for acc in accounts:
            if acc.get("id") == account_id:
                return acc
        return {}

    # ── Contracts ─────────────────────────────────────────────────────

    def search_contracts(self, text: str = "NQ") -> List[Dict]:
        """Search for futures contracts."""
        self.ensure_auth()
        resp = self.session.post(
            f"{self.BASE_URL}/api/Contract/search",
            json={"text": text},
        )
        resp.raise_for_status()
        return resp.json()

    def get_nq_contract(self) -> Optional[Dict]:
        """Find the active front-month NQ contract."""
        contracts = self.search_contracts("NQ")
        # Filter for E-mini NQ (not micro)
        nq_contracts = [
            c for c in contracts
            if "ENQ" in c.get("id", "") or "NQ" in c.get("name", "")
        ]
        if nq_contracts:
            return nq_contracts[0]
        return contracts[0] if contracts else None

    # ── Orders ────────────────────────────────────────────────────────

    def place_order(
        self,
        account_id: int,
        contract_id: str,
        side: str,          # "Buy" or "Sell"
        size: int,
        order_type: int = 1,  # 1=Market, 2=Limit, 3=Stop
        limit_price: float = None,
        stop_price: float = None,
    ) -> Dict:
        """Place an order."""
        self.ensure_auth()
        payload = {
            "accountId": account_id,
            "contractId": contract_id,
            "type": order_type,
            "side": side,
            "size": size,
        }
        if limit_price is not None:
            payload["limitPrice"] = limit_price
        if stop_price is not None:
            payload["stopPrice"] = stop_price

        log.info(f"Placing order: {side} {size}x {contract_id} on account {account_id}")
        resp = self.session.post(f"{self.BASE_URL}/api/Order/place", json=payload)
        resp.raise_for_status()
        result = resp.json()
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
        return resp.json()

    # ── Positions ─────────────────────────────────────────────────────

    def get_positions(self, account_id: int) -> List[Dict]:
        """Get open positions."""
        self.ensure_auth()
        resp = self.session.post(
            f"{self.BASE_URL}/api/Position/search",
            json={"accountId": account_id},
        )
        resp.raise_for_status()
        return resp.json()

    # ── Market Data ───────────────────────────────────────────────────

    def get_bars(self, contract_id: str, bar_type: int = 5, count: int = 100) -> List[Dict]:
        """Get historical bars. bar_type: minutes per bar."""
        self.ensure_auth()
        resp = self.session.post(
            f"{self.BASE_URL}/api/History/bars",
            json={
                "contractId": contract_id,
                "barType": bar_type,
                "count": count,
            },
        )
        resp.raise_for_status()
        return resp.json()

    def get_quote(self, contract_id: str) -> Dict:
        """Get current quote."""
        self.ensure_auth()
        resp = self.session.post(
            f"{self.BASE_URL}/api/MarketData/quote",
            json={"contractId": contract_id},
        )
        resp.raise_for_status()
        return resp.json()


# ═════════════════════════════════════════════════════════════════════════
# Account Tracker — tracks state per account
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class AccountState:
    account_id: int
    name: str = ""
    phase: Phase = Phase.EVAL
    contract_id: str = ""

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
    if len(daily_df) < 252:
        return 0.0

    high = daily_df["high"]
    volume = daily_df["volume"]

    # rank = rolling percentile rank over 252 days
    vol_rank = volume.rolling(252, min_periods=1).apply(
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
    """Compute VWAP from today's bars."""
    if not bars:
        return 0.0
    cum_tp_vol = 0.0
    cum_vol = 0.0
    for b in bars:
        tp = (b["high"] + b["low"] + b["close"]) / 3.0
        vol = b.get("volume", 1)
        cum_tp_vol += tp * vol
        cum_vol += vol
    return cum_tp_vol / cum_vol if cum_vol > 0 else 0.0


def compute_rsi(closes: List[float], period: int = 14) -> float:
    """Compute RSI from a list of close prices."""
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def check_vwap_mr_entry(bars: List[Dict], config: dict) -> bool:
    """
    Check VWAP Micro MR entry conditions on current 5m bar.
    Entry: price dips below VWAP, RSI oversold, bearish bar, bullish day bias.
    """
    if len(bars) < 2:
        return False

    current = bars[-1]
    prev = bars[-2]
    price = current["close"]
    vwap = compute_vwap(bars)

    # Dip below VWAP
    dip = vwap - price
    if dip < config["min_dip_pts"]:
        return False

    # Bearish bar (close < open)
    if current["close"] >= current["open"]:
        return False
    down_bar = current["open"] - current["close"]
    if down_bar < config["min_down_bar_pts"]:
        return False

    # RSI oversold
    closes = [b["close"] for b in bars]
    rsi = compute_rsi(closes, 14)
    if rsi >= config["rsi_oversold_max"]:
        return False

    # Day bias filter: bullish if current price > day open
    if config.get("day_bias_filter", False):
        day_open = bars[0]["open"]
        if price <= day_open:
            return False

    return True


# ═════════════════════════════════════════════════════════════════════════
# Trading Engine
# ═════════════════════════════════════════════════════════════════════════

class TradingEngine:
    """Core trading engine managing multiple accounts."""

    POLL_INTERVAL = 30        # seconds between market checks
    BAR_INTERVAL = 300        # 5 minutes in seconds

    def __init__(self, api: ProjectXAPI, state_file: str = "bot_state.json"):
        self.api = api
        self.state_file = state_file
        self.accounts: Dict[int, AccountState] = {}
        self.contract_id: str = ""
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
        log.info(f"NQ contract: {self.contract_id} ({nq.get('name', '')})")

        # Load accounts
        accounts = self.api.get_accounts()
        if not accounts:
            raise RuntimeError("No trading accounts found")

        for acc in accounts:
            acc_id = acc["id"]
            # Determine phase from account name/type
            name = acc.get("name", "")
            is_funded = any(
                kw in name.lower()
                for kw in ["funded", "live", "express"]
            )
            phase = Phase.FUNDED if is_funded else Phase.EVAL

            self.accounts[acc_id] = AccountState(
                account_id=acc_id,
                name=name,
                phase=phase,
                contract_id=self.contract_id,
            )
            log.info(f"Account {acc_id}: {name} → {phase.value}")

        # Load saved state
        self._load_state()

        log.info(f"Engine ready: {len(self.accounts)} accounts, contract={self.contract_id}")

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
        h, m = dt_et.hour, dt_et.minute
        mins = h * 60 + m

        # Saturday: closed
        if wd == 5:
            return False
        # Sunday: open from 6pm
        if wd == 6:
            return mins >= 18 * 60
        # Friday: close at 5pm
        if wd == 4:
            return mins < 17 * 60
        # Mon-Thu: closed 5pm-6pm daily halt
        if 17 * 60 <= mins < 18 * 60:
            return False
        return True

    @staticmethod
    def _is_eval_session(dt_et: datetime) -> bool:
        """Eval trades only 9:30 AM - 11:30 AM ET."""
        mins = dt_et.hour * 60 + dt_et.minute
        return 9 * 60 + 30 <= mins <= 11 * 60 + 30

    @staticmethod
    def _is_funded_entry_time(dt_et: datetime) -> bool:
        """Alpha#44: enter at market open ~9:30 AM ET."""
        mins = dt_et.hour * 60 + dt_et.minute
        return 9 * 60 + 30 <= mins <= 9 * 60 + 35

    @staticmethod
    def _is_funded_exit_time(dt_et: datetime) -> bool:
        """Alpha#44: exit at market close ~3:55 PM ET."""
        mins = dt_et.hour * 60 + dt_et.minute
        return 15 * 60 + 55 <= mins <= 16 * 60

    # ── Daily Reset ───────────────────────────────────────────────────

    def _check_new_day(self):
        """Reset daily state at midnight ET."""
        today_str = str(self._now_et().date())
        for acc in self.accounts.values():
            if acc.today != today_str:
                if acc.today:  # not first run
                    # Record previous day stats
                    if acc.daily_pnl >= WINNING_DAY_THRESHOLD:
                        acc.winning_days += 1
                    acc.trading_days += 1

                acc.today = today_str
                acc.daily_pnl = 0.0
                acc.trades_today = 0
                acc.entered_today = False
                log.info(f"Account {acc.account_id}: new day {today_str}")

    # ── Build Daily Bars for Alpha ────────────────────────────────────

    def _refresh_daily_bars(self):
        """Fetch recent bars and build daily OHLCV for alpha computation."""
        try:
            raw_bars = self.api.get_bars(self.contract_id, bar_type=5, count=2000)
            if not raw_bars:
                return

            df = pd.DataFrame(raw_bars)
            if "timestamp" in df.columns:
                df["datetime"] = pd.to_datetime(df["timestamp"])
            elif "t" in df.columns:
                df["datetime"] = pd.to_datetime(df["t"])

            for col in ["open", "high", "low", "close", "volume"]:
                alt = col[0]  # o, h, l, c, v
                if col not in df.columns and alt in df.columns:
                    df[col] = df[alt]

            df["date"] = df["datetime"].dt.date
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

    # ── Core Trading Logic ────────────────────────────────────────────

    def _trade_eval(self, acc: AccountState):
        """VWAP Micro MR strategy for eval accounts."""
        now = self._now_et()

        if not self._is_eval_session(now):
            # If we have a position outside session, close it
            if acc.position != 0:
                self._close_position(acc, "session_end")
            return

        # Check daily loss limit
        if acc.daily_pnl <= -DAILY_LOSS_LIMIT:
            log.info(f"Account {acc.account_id}: daily loss limit hit, no more trades today")
            if acc.position != 0:
                self._close_position(acc, "daily_loss_limit")
            return

        # Max trades per day
        if acc.trades_today >= EVAL_CONFIG["max_trades_per_day"]:
            return

        # Fetch recent 5m bars for today
        try:
            bars = self.api.get_bars(self.contract_id, bar_type=5, count=50)
            if not bars:
                return

            # Filter to today's session bars
            today_str = str(now.date())
            today_bars = []
            for b in bars:
                ts = b.get("timestamp", b.get("t", ""))
                if today_str in str(ts):
                    today_bars.append(b)

            if not today_bars:
                today_bars = bars[-20:]  # fallback

            acc.bars_5m = today_bars
            acc.vwap = compute_vwap(today_bars)
        except Exception as e:
            log.error(f"Failed to get bars: {e}")
            return

        current_price = acc.bars_5m[-1]["close"] if acc.bars_5m else 0

        # If we have a position, check stop/target
        if acc.position != 0:
            if current_price <= acc.stop_price:
                self._close_position(acc, "stop_hit")
            elif current_price >= acc.target_price:
                self._close_position(acc, "target_hit")
            return

        # Check for new entry
        if check_vwap_mr_entry(acc.bars_5m, EVAL_CONFIG):
            self._open_position(
                acc,
                side="Buy",
                contracts=EVAL_CONFIG["contracts"],
                entry_price=current_price,
                stop_pts=EVAL_CONFIG["stop_pts"],
                target_pts=EVAL_CONFIG["target_pts"],
            )

    def _trade_funded(self, acc: AccountState):
        """Alpha#44 strategy for funded accounts."""
        now = self._now_et()

        # Check daily loss limit
        if acc.daily_pnl <= -DAILY_LOSS_LIMIT:
            if acc.position != 0:
                self._close_position(acc, "daily_loss_limit")
            return

        # Compute alpha signal once per day (before market open)
        if not acc.entered_today and len(self._daily_bars) > 0:
            acc.alpha_signal = compute_alpha044(self._daily_bars)
            acc.alpha_direction = "LONG" if acc.alpha_signal > 0 else "FLAT"

        # Entry: at open if signal is LONG
        if (acc.position == 0
                and acc.alpha_direction == "LONG"
                and not acc.entered_today
                and self._is_funded_entry_time(now)):
            try:
                quote = self.api.get_quote(self.contract_id)
                price = quote.get("last", quote.get("lastPrice", 0))
                if price > 0:
                    self._open_position(
                        acc,
                        side="Buy",
                        contracts=FUNDED_CONFIG["contracts"],
                        entry_price=price,
                        stop_pts=None,
                        target_pts=None,
                    )
                    acc.entered_today = True
            except Exception as e:
                log.error(f"Funded entry failed: {e}")

        # Exit: at close
        if acc.position != 0 and self._is_funded_exit_time(now):
            self._close_position(acc, "eod_exit")

    # ── Order Execution ───────────────────────────────────────────────

    def _open_position(
        self,
        acc: AccountState,
        side: str,
        contracts: int,
        entry_price: float,
        stop_pts: Optional[float],
        target_pts: Optional[float],
    ):
        """Open a position via market order."""
        try:
            result = self.api.place_order(
                account_id=acc.account_id,
                contract_id=acc.contract_id,
                side=side,
                size=contracts,
                order_type=1,  # Market
            )
            acc.position = contracts if side == "Buy" else -contracts
            acc.entry_price = entry_price
            acc.stop_price = entry_price - stop_pts if stop_pts else 0
            acc.target_price = entry_price + target_pts if target_pts else 999999
            acc.trades_today += 1

            log.info(
                f"OPENED: Account {acc.account_id} | {side} {contracts}x NQ @ {entry_price:.2f} | "
                f"Stop={acc.stop_price:.2f} Target={acc.target_price:.2f}"
            )
        except Exception as e:
            log.error(f"Order failed: {e}")
            acc.last_error = str(e)

    def _close_position(self, acc: AccountState, reason: str):
        """Close position via market order."""
        if acc.position == 0:
            return

        try:
            side = "Sell" if acc.position > 0 else "Buy"
            contracts = abs(acc.position)

            quote = self.api.get_quote(acc.contract_id)
            exit_price = quote.get("last", quote.get("lastPrice", 0))

            result = self.api.place_order(
                account_id=acc.account_id,
                contract_id=acc.contract_id,
                side=side,
                size=contracts,
                order_type=1,  # Market
            )

            # Calc PnL
            pnl = (exit_price - acc.entry_price) * POINT_VALUE * contracts
            if acc.position < 0:
                pnl = -pnl

            acc.daily_pnl += pnl
            acc.equity += pnl
            acc.peak = max(acc.peak, acc.equity)
            acc.profit_since_payout += pnl

            log.info(
                f"CLOSED: Account {acc.account_id} | {reason} | {side} {contracts}x NQ @ {exit_price:.2f} | "
                f"PnL=${pnl:+,.2f} | Equity=${acc.equity:,.2f}"
            )

            acc.position = 0
            acc.entry_price = 0
            acc.stop_price = 0
            acc.target_price = 0

            # Check if account busted
            drawdown = acc.peak - acc.equity
            max_dd = EVAL_TRAILING_DD if acc.phase == Phase.EVAL else FUNDED_TRAILING_DD
            if drawdown >= max_dd:
                acc.active = False
                log.warning(f"BUST: Account {acc.account_id} trailing DD ${drawdown:,.2f} >= ${max_dd:,.2f}")

            # Check eval pass
            if acc.phase == Phase.EVAL and acc.equity >= EVAL_PROFIT_TARGET and acc.trading_days >= 2:
                log.info(f"EVAL PASSED: Account {acc.account_id}! Equity=${acc.equity:,.2f}")

        except Exception as e:
            log.error(f"Close failed: {e}")
            acc.last_error = str(e)

    # ── Dashboard ─────────────────────────────────────────────────────

    def _build_dashboard(self) -> Table:
        """Build rich dashboard table."""
        now = self._now_et()
        market_open = self._is_market_hours(now)

        table = Table(
            title=f"TopstepX Bot | {now.strftime('%Y-%m-%d %H:%M:%S ET')} | Market: {'OPEN' if market_open else 'CLOSED'}",
            show_lines=True,
        )
        table.add_column("Account", style="bold")
        table.add_column("Phase", justify="center")
        table.add_column("Equity", justify="right")
        table.add_column("DD Left", justify="right")
        table.add_column("Day PnL", justify="right")
        table.add_column("Position", justify="center")
        table.add_column("Trades", justify="center")
        table.add_column("Status", justify="center")

        for acc in self.accounts.values():
            max_dd = EVAL_TRAILING_DD if acc.phase == Phase.EVAL else FUNDED_TRAILING_DD
            dd_left = max_dd - (acc.peak - acc.equity)

            phase_text = Text(acc.phase.value.upper())
            phase_text.stylize("green" if acc.phase == Phase.EVAL else "blue")

            equity_text = Text(f"${acc.equity:,.2f}")
            equity_text.stylize("green" if acc.equity >= 0 else "red")

            dd_text = Text(f"${dd_left:,.2f}")
            dd_text.stylize("green" if dd_left > 500 else "red")

            day_text = Text(f"${acc.daily_pnl:+,.2f}")
            day_text.stylize("green" if acc.daily_pnl >= 0 else "red")

            pos = f"{acc.position}x" if acc.position != 0 else "FLAT"
            pos_text = Text(pos)
            pos_text.stylize("cyan" if acc.position != 0 else "dim")

            if not acc.active:
                status = Text("BUST", style="bold red")
            elif acc.phase == Phase.EVAL and acc.equity >= EVAL_PROFIT_TARGET:
                status = Text("PASSED", style="bold green")
            elif acc.position != 0:
                status = Text("IN TRADE", style="bold cyan")
            else:
                status = Text("WATCHING", style="dim")

            # Funded extra info
            extra = ""
            if acc.phase == Phase.FUNDED:
                extra = f" | W:{acc.winning_days}/{WINNING_DAYS_REQUIRED} | ${acc.total_extracted:,.0f} out"

            table.add_row(
                f"{acc.name} ({acc.account_id})",
                phase_text,
                equity_text,
                dd_text,
                day_text,
                pos_text,
                f"{acc.trades_today}",
                status,
            )

        return table

    # ── Main Loop ─────────────────────────────────────────────────────

    def run(self):
        """Main trading loop — runs 24/7."""
        self.running = True
        log.info("Bot started — running 24/7")

        # Refresh daily bars on startup
        self._refresh_daily_bars()
        last_daily_refresh = time.time()

        try:
            with Live(self._build_dashboard(), console=console, refresh_per_second=0.5) as live:
                while self.running:
                    try:
                        now = self._now_et()
                        self._check_new_day()

                        # Refresh daily bars every 30 minutes
                        if time.time() - last_daily_refresh > 1800:
                            self._refresh_daily_bars()
                            last_daily_refresh = time.time()

                        # Re-auth periodically
                        self.api.ensure_auth()

                        if self._is_market_hours(now):
                            for acc in self.accounts.values():
                                if not acc.active:
                                    continue
                                try:
                                    if acc.phase == Phase.EVAL:
                                        self._trade_eval(acc)
                                    else:
                                        self._trade_funded(acc)
                                except Exception as e:
                                    log.error(f"Trade error on {acc.account_id}: {e}")
                                    acc.last_error = str(e)

                        # Update dashboard
                        live.update(self._build_dashboard())

                        # Save state
                        self._save_state()

                        # Sleep between checks
                        time.sleep(self.POLL_INTERVAL)

                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        log.error(f"Loop error: {e}\n{traceback.format_exc()}")
                        time.sleep(60)

        except KeyboardInterrupt:
            log.info("Shutting down...")
            # Close all positions before exit
            for acc in self.accounts.values():
                if acc.position != 0:
                    log.info(f"Closing position on {acc.account_id} before shutdown")
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
  python topstep_bot.py --api-key abc123
  python topstep_bot.py --config config.json
  python topstep_bot.py --api-key abc123 --poll-interval 15

Config file format (config.json):
  {
    "api_key": "your-api-key",
    "poll_interval": 30,
    "accounts": {
      "12345": {"phase": "eval"},
      "67890": {"phase": "funded"}
    }
  }
        """,
    )
    parser.add_argument("--api-key", help="TopstepX/ProjectX API key")
    parser.add_argument("--config", help="Path to config JSON file")
    parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between market checks (default: 30)")
    args = parser.parse_args()

    # Load config
    api_key = args.api_key
    account_phases = {}

    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        api_key = api_key or cfg.get("api_key")
        account_phases = cfg.get("accounts", {})

    if not api_key:
        # Check env var
        api_key = os.environ.get("TOPSTEPX_API_KEY")

    if not api_key:
        console.print("[bold red]No API key provided.[/]")
        console.print("Use --api-key, --config, or set TOPSTEPX_API_KEY env var.")
        console.print("\nTo get an API key:")
        console.print("  1. Go to TopstepX dashboard")
        console.print("  2. Settings → API Access")
        console.print("  3. Generate API key ($29/month)")
        sys.exit(1)

    # Build and run
    api = ProjectXAPI(api_key)
    engine = TradingEngine(api)
    engine.POLL_INTERVAL = args.poll_interval

    console.print(Panel(
        "[bold]TopstepX Automated Trading Bot[/]\n\n"
        "Eval:   VWAP Micro Mean-Reversion (9:30-11:30 AM ET)\n"
        "Funded: Alpha#44 (long at open, exit at close)\n\n"
        "Press Ctrl+C to stop (positions will be closed safely)",
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

        engine.run()
    except Exception as e:
        log.error(f"Fatal error: {e}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
