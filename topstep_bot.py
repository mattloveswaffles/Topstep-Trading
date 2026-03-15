"""
TopstepX Automated Trading Bot
===============================
Runs 24/7 monitoring the market and trading NQ futures automatically.

Phase 1 (EVAL):   VWAP Micro Mean-Reversion -- pass the 50K combine
Phase 2 (FUNDED): VWAP Micro Mean-Reversion -- extract payouts

Supports multiple accounts simultaneously, auto-detects eval vs funded.

Uses real-time SignalR WebSocket for:
  - Account balance updates (GatewayUserAccount)
  - Position changes (GatewayUserPosition)
  - Trade executions + PnL (GatewayUserTrade)
  - Order status (GatewayUserOrder)
  - Live market quotes (GatewayQuote)
"""

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from signalrcore.hub_connection_builder import HubConnectionBuilder
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# =========================================================================
# Configuration
# =========================================================================

load_dotenv()

console = Console(width=140)
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

ET = ZoneInfo("America/New_York")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"bot_{date.today()}.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("topstep_bot")

# Suppress noisy signalrcore reconnect warnings
for _logger_name in ("signalrcore", "signalrcore.hub", "signalrcore.transport",
                      "websocket", "websockets"):
    logging.getLogger(_logger_name).setLevel(logging.CRITICAL)


class _SignalRFilter(logging.Filter):
    """Filter out 'Connection closed' spam from signalrcore hitting root logger."""
    def filter(self, record):
        msg = record.getMessage()
        return "Connection closed" not in msg and "Handshake" not in msg

for _h in logging.root.handlers:
    _h.addFilter(_SignalRFilter())


class Phase(str, Enum):
    EVAL = "eval"
    FUNDED = "funded"


# -- Topstep Rules --------------------------------------------------------
POINT_VALUE = 20.0            # NQ $20 per point
TICK_SIZE = 0.25              # NQ min tick
TICK_VALUE = 5.0              # $5 per tick
START_BALANCE = 50000.0       # Topstep 50K combine starting balance
EVAL_PROFIT_TARGET = 3000.0
EVAL_TRAILING_DD = 2000.0
DAILY_LOSS_LIMIT = 1000.0
FUNDED_TRAILING_DD = 2000.0
# Consistency Path payout rules
PAYOUT_MAX = 6000.0
PAYOUT_BAL_PCT = 0.50
WINNING_DAY_THRESHOLD = 150.0
WINNING_DAYS_REQUIRED = 3
CONSISTENCY_PCT = 0.40           # No single day > 40% of total profit

# -- Session Definitions (ET) ---------------------------------------------
# Each session: (name, start_min, end_min)
# Topstep "day" starts at Asia open (6PM ET)
SESSIONS = [
    ("Asia",   18 * 60,       2 * 60),       # 6:00 PM - 2:00 AM ET (wraps midnight)
    ("London",  3 * 60,       9 * 60 + 30),  # 3:00 AM - 9:30 AM ET
    ("NY",      9 * 60 + 30, 15 * 60 + 45),  # 9:30 AM - 3:45 PM ET
]


def _is_market_open(dt_et: datetime) -> bool:
    """NQ futures: Sun 6pm - Fri 5pm ET, with daily halt 5pm-6pm ET."""
    wd = dt_et.weekday()
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


def get_current_session(dt_et: datetime) -> Optional[str]:
    """Return the name of the active session, or None if between sessions or market closed."""
    if not _is_market_open(dt_et):
        return None
    mins = dt_et.hour * 60 + dt_et.minute
    for name, start, end in SESSIONS:
        if start <= end:
            if start <= mins <= end:
                return name
        else:
            # Overnight wrap (e.g. Asia 6PM-2AM)
            if mins >= start or mins <= end:
                return name
    return None


# -- Strategy Config ------------------------------------------------------
EVAL_CONFIG = {
    "stop_pts": 5.0,
    "target_pts": 20.0,
    "contracts": 2,
    "max_trades_per_session": 5,
    "min_dip_pts": 1.5,
    "min_down_bar_pts": 0.5,
    "rsi_oversold_max": 50.0,           # effectively disabled
    "rsi_overbought_min": 50.0,         # effectively disabled
    "day_bias_filter": False,           # no day bias -- trade any day
    "trailing_stop_pts": 2.0,           # trail stop 2 pts behind best price
    "long_only": False,                 # trade both long and short
    "max_daily_pnl": 1600.0,           # stop trading once daily PnL hits $1600
}

FUNDED_CONFIG = {
    "stop_pts": 5.0,
    "target_pts": 20.0,
    "contracts": 2,
    "max_trades_per_session": 5,
    "min_dip_pts": 1.5,
    "min_down_bar_pts": 0.5,
    "rsi_oversold_max": 50.0,
    "rsi_overbought_min": 50.0,
    "day_bias_filter": False,
    "trailing_stop_pts": 2.0,
    "long_only": False,
    "max_daily_pnl": 1600.0,
}

LOSS_COOLDOWN_MINUTES = 180    # block entries for 3 hours after a loss
STUCK_TRADE_TIMEOUT_SECS = 300 # force-clear _trade_active if stuck this long
PENDING_EXIT_TIMEOUT_SECS = 60 # fall back to REST if RT trade event missed
MIN_EXIT_GAP_SECS = 30         # min seconds between exit and next entry


# =========================================================================
# Telegram Notifications (optional -- set TELEGRAM_BOT_TOKEN in .env)
# =========================================================================

class TelegramNotifier:
    """Sends Telegram messages from a background thread so it never blocks trading."""

    def __init__(self, bot_token: str, chat_id: Optional[str] = None):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _resolve_chat_id(self) -> bool:
        """Auto-discover chat_id from the most recent message sent to the bot."""
        try:
            resp = requests.get(
                f"https://api.telegram.org/bot{self.bot_token}/getUpdates",
                params={"limit": 1, "offset": -1},
                timeout=10,
            )
            data = resp.json()
            results = data.get("result", [])
            if results:
                msg = results[0].get("message") or results[0].get("channel_post", {})
                chat = msg.get("chat", {})
                self.chat_id = str(chat.get("id", ""))
                if self.chat_id:
                    log.info(f"Telegram chat_id auto-discovered: {self.chat_id}")
                    return True
            log.warning("Telegram: no messages found. Send any message to the bot first, then restart.")
        except Exception as e:
            log.warning(f"Telegram chat_id discovery failed: {e}")
        return False

    def _worker(self):
        """Background thread that drains the message queue."""
        while True:
            msg = self._queue.get()
            if msg is None:
                break
            if not self.chat_id:
                if not self._resolve_chat_id():
                    continue
            try:
                requests.post(
                    f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": msg,
                        "parse_mode": "HTML",
                    },
                    timeout=10,
                )
            except Exception as e:
                log.warning(f"Telegram send failed: {e}")

    def send(self, text: str):
        """Enqueue a message (non-blocking)."""
        self._queue.put(text)


def _build_telegram_notifier() -> Optional[TelegramNotifier]:
    """Create notifier if TELEGRAM_BOT_TOKEN is set."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return None
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    log.info(f"Telegram notifications enabled (chat_id={'set' if chat_id else 'auto-discover'})")
    return TelegramNotifier(token, chat_id)


# Global notifier instance (set in main)
_telegram: Optional[TelegramNotifier] = None


# =========================================================================
# ProjectX / TopstepX API Client
# Docs: https://gateway.docs.projectx.com/
# =========================================================================

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

    # -- Auth --------------------------------------------------------------

    def authenticate(self) -> bool:
        """Login with API key, get JWT token."""
        try:
            resp = self.session.post(
                f"{self.BASE_URL}/api/Auth/loginKey",
                json={"userName": self.username, "apiKey": self.api_key},
            )
            resp.raise_for_status()
            data = resp.json()

            token = (
                data.get("token")
                or data.get("access_token")
                or data.get("Token")
                or data.get("accessToken")
            )

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
            log.error(f"Auth exception: {e}", exc_info=True)
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

    def _unwrap_list(self, data, key: str, context: str) -> list:
        """Extract list from API response (handles both bare lists and wrapped objects)."""
        if isinstance(data, list):
            return data
        if not self._check_response(data, context):
            return []
        return data.get(key, data.get("data", []))

    # -- Accounts ----------------------------------------------------------

    def get_accounts(self) -> List[Dict]:
        """Get all active trading accounts."""
        self.ensure_auth()
        resp = self.session.post(
            f"{self.BASE_URL}/api/Account/search",
            json={"onlyActiveAccounts": True},
        )
        resp.raise_for_status()
        return self._unwrap_list(resp.json(), "accounts", "get_accounts")

    # -- Contracts ---------------------------------------------------------

    def search_contracts(self, text: str = "NQ", live: bool = False) -> List[Dict]:
        """Search for futures contracts. Returns up to 20 results."""
        self.ensure_auth()
        resp = self.session.post(
            f"{self.BASE_URL}/api/Contract/search",
            json={"searchText": text, "live": live},
        )
        resp.raise_for_status()
        return self._unwrap_list(resp.json(), "contracts", "search_contracts")

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
            is_emini = "ENQ" in cid or ("NQ" in name and "MNQ" not in name)
            if is_emini and active:
                return c

        # Fallback: first active NQ
        for c in contracts:
            if c.get("activeContract", True):
                return c
        return contracts[0]

    # -- Orders ------------------------------------------------------------

    def place_market_order(
        self,
        account_id: int,
        contract_id: str,
        side: int,
        size: int,
        stop_loss_ticks: Optional[int] = None,
        take_profit_ticks: Optional[int] = None,
        trailing_stop: bool = False,
    ) -> Dict:
        """Place a market order with optional bracket orders.

        If trailing_stop=True, the stop loss bracket uses TrailingStop type (5)
        so the exchange automatically trails the stop.
        """
        self.ensure_auth()
        payload = {
            "accountId": account_id,
            "contractId": contract_id,
            "type": OrderType.MARKET,
            "side": side,
            "size": size,
        }

        if stop_loss_ticks is not None:
            stop_type = OrderType.TRAILING_STOP if trailing_stop else OrderType.STOP
            payload["stopLossBracket"] = {"ticks": stop_loss_ticks, "type": stop_type}
        if take_profit_ticks is not None:
            payload["takeProfitBracket"] = {"ticks": take_profit_ticks, "type": OrderType.LIMIT}

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

    def modify_order(
        self,
        account_id: int,
        order_id: int,
        stop_price: Optional[float] = None,
        limit_price: Optional[float] = None,
        size: Optional[int] = None,
    ) -> Dict:
        """Modify an existing order (e.g. move stop price for trailing stop)."""
        self.ensure_auth()
        payload: Dict[str, Any] = {
            "accountId": account_id,
            "orderId": order_id,
        }
        if stop_price is not None:
            payload["stopPrice"] = stop_price
        if limit_price is not None:
            payload["limitPrice"] = limit_price
        if size is not None:
            payload["size"] = size

        log.info(f"Modifying order {order_id}: {payload}")
        resp = self.session.post(f"{self.BASE_URL}/api/Order/modify", json=payload)
        resp.raise_for_status()
        result = resp.json()
        if not self._check_response(result, "modify_order"):
            raise RuntimeError(f"Order modify rejected: {result.get('errorMessage')}")
        return result

    def get_open_orders(self, account_id: int) -> List[Dict]:
        """Get open orders for an account."""
        self.ensure_auth()
        resp = self.session.post(
            f"{self.BASE_URL}/api/Order/searchOpen",
            json={"accountId": account_id},
        )
        resp.raise_for_status()
        return self._unwrap_list(resp.json(), "orders", "get_open_orders")

    # -- Positions ---------------------------------------------------------

    def get_positions(self, account_id: int) -> List[Dict]:
        """Get open positions for an account."""
        self.ensure_auth()
        resp = self.session.post(
            f"{self.BASE_URL}/api/Position/searchOpen",
            json={"accountId": account_id},
        )
        resp.raise_for_status()
        if not resp.text or not resp.text.strip():
            return []
        return self._unwrap_list(resp.json(), "positions", "get_positions")

    def get_trades(self, account_id: int, start_timestamp: str) -> List[Dict]:
        """Search trades for an account from a given start time."""
        self.ensure_auth()
        resp = self.session.post(
            f"{self.BASE_URL}/api/Trade/search",
            json={"accountId": account_id, "startTimestamp": start_timestamp},
        )
        resp.raise_for_status()
        if not resp.text or not resp.text.strip():
            return []
        data = resp.json()
        return data.get("trades", [])

    # -- Market Data -------------------------------------------------------

    def get_bars(
        self,
        contract_id: str,
        unit: int = BarUnit.MINUTE,
        unit_number: int = 5,
        limit: int = 100,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[Dict]:
        """Retrieve historical bars. Max 20,000 per request.

        Response bars have fields: t (timestamp), o (open), h (high), l (low), c (close), v (volume)
        """
        self.ensure_auth()

        if not end_time:
            end_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if not start_time:
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
        resp = self.session.post(f"{self.BASE_URL}/api/History/retrieveBars", json=inner)
        if resp.status_code != 200:
            log.error(f"Bars response {resp.status_code}: {resp.text[:500]}")
        resp.raise_for_status()
        return self._unwrap_list(resp.json(), "bars", "get_bars")


# =========================================================================
# Real-Time WebSocket Client (SignalR)
# Docs: https://gateway.docs.projectx.com/docs/realtime/
# =========================================================================

RTC_BASE = "https://rtc.topstepx.com"
USER_HUB_URL = f"{RTC_BASE}/hubs/user"
MARKET_HUB_URL = f"{RTC_BASE}/hubs/market"


class RealtimeClient:
    """SignalR WebSocket client for real-time account + market data.

    User Hub:
      - Account balance updates (GatewayUserAccount)
      - Position changes (GatewayUserPosition)
      - Trade executions + PnL (GatewayUserTrade)
      - Order status (GatewayUserOrder)

    Market Hub:
      - Live quotes with bid/ask/last (GatewayQuote)
    """

    def __init__(self, token_factory: Callable[[], str]):
        self._token_factory = token_factory
        self._user_conn = None
        self._market_conn = None
        self._running = False

        # Callbacks set by TradingEngine
        self.on_account_update: Optional[Callable] = None
        self.on_position_update: Optional[Callable] = None
        self.on_trade_update: Optional[Callable] = None
        self.on_order_update: Optional[Callable] = None
        self.on_quote_update: Optional[Callable] = None

        # Subscriptions to restore on reconnect
        self._account_ids: List[int] = []
        self._contract_ids: List[str] = []

    # -- Event Handlers ----------------------------------------------------

    def _handle_account(self, args):
        data = args[0] if isinstance(args, (list, tuple)) and args else args
        log.debug(f"RT Account: {data}")
        if self.on_account_update:
            self.on_account_update(data)

    def _handle_position(self, args):
        data = args[0] if isinstance(args, (list, tuple)) and args else args
        log.debug(f"RT Position: {data}")
        if self.on_position_update:
            self.on_position_update(data)

    def _handle_trade(self, args):
        data = args[0] if isinstance(args, (list, tuple)) and args else args
        log.debug(f"RT Trade: {data}")
        if self.on_trade_update:
            self.on_trade_update(data)

    def _handle_order(self, args):
        data = args[0] if isinstance(args, (list, tuple)) and args else args
        log.debug(f"RT Order: {data}")
        if self.on_order_update:
            self.on_order_update(data)

    def _handle_quote(self, args):
        data = args[1] if isinstance(args, (list, tuple)) and len(args) > 1 else args
        if self.on_quote_update:
            self.on_quote_update(data)

    # -- Connection Callbacks ----------------------------------------------

    def _on_user_open(self):
        log.info("RT User Hub connected - subscribing...")
        self._user_conn.send("SubscribeAccounts", [])
        for acc_id in self._account_ids:
            self._user_conn.send("SubscribeOrders", [acc_id])
            self._user_conn.send("SubscribePositions", [acc_id])
            self._user_conn.send("SubscribeTrades", [acc_id])
        log.info(f"RT User Hub subscribed: {len(self._account_ids)} account(s)")

    def _on_market_open(self):
        log.info("RT Market Hub connected - subscribing...")
        for cid in self._contract_ids:
            self._market_conn.send("SubscribeContractQuotes", [cid])
        log.info(f"RT Market Hub subscribed: {len(self._contract_ids)} contract(s)")

    def _on_close(self):
        log.debug("RT WebSocket closed - will auto-reconnect")

    def _on_error(self, data):
        log.error(f"RT WebSocket error: {data}")

    # -- Start / Stop ------------------------------------------------------

    def subscribe(self, account_ids: List[int], contract_ids: List[str]):
        self._account_ids = account_ids
        self._contract_ids = contract_ids

    def _build_connection(self, hub_url: str):
        """Build a SignalR connection with skip_negotiation + WebSocket transport."""
        token = self._token_factory()
        url = f"{hub_url}?access_token={token}"
        return HubConnectionBuilder() \
            .with_url(url, options={
                "access_token_factory": self._token_factory,
                "skip_negotiation": True,
                "type": 1,
            }) \
            .with_automatic_reconnect({
                "type": "interval",
                "keep_alive_interval": 10,
                "intervals": [1, 2, 5, 10, 30, 60],
            }) \
            .build()

    def start(self):
        """Start both SignalR connections (user hub + market hub)."""
        if self._running:
            return

        self._user_conn = self._build_connection(USER_HUB_URL)
        self._user_conn.on("GatewayUserAccount", self._handle_account)
        self._user_conn.on("GatewayUserPosition", self._handle_position)
        self._user_conn.on("GatewayUserTrade", self._handle_trade)
        self._user_conn.on("GatewayUserOrder", self._handle_order)
        self._user_conn.on_open(self._on_user_open)
        self._user_conn.on_close(self._on_close)
        self._user_conn.on_error(self._on_error)

        self._market_conn = self._build_connection(MARKET_HUB_URL)
        self._market_conn.on("GatewayQuote", self._handle_quote)
        self._market_conn.on_open(self._on_market_open)
        self._market_conn.on_close(self._on_close)
        self._market_conn.on_error(self._on_error)

        self._user_conn.start()
        self._market_conn.start()
        self._running = True
        log.info("RT WebSocket started (user + market hubs)")

    def stop(self):
        if self._running:
            if self._user_conn:
                self._user_conn.stop()
            if self._market_conn:
                self._market_conn.stop()
            self._running = False
            log.info("RT WebSocket stopped")


# =========================================================================
# Account State -- tracks per-account trading state
# =========================================================================

class EntryGuard(str, Enum):
    """Why entries are currently blocked for an account.

    CLEAR          -- ready to trade
    TRADE_ACTIVE   -- order placed, waiting for RT trade to confirm exit PnL
    PENDING_EXIT   -- position went flat on exchange, waiting for RT trade event
    COOLDOWN       -- post-loss cooldown (180 min)
    EXIT_GAP       -- minimum 30s gap after any exit
    """
    CLEAR = "clear"
    TRADE_ACTIVE = "trade_active"
    PENDING_EXIT = "pending_exit"
    COOLDOWN = "cooldown"
    EXIT_GAP = "exit_gap"


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
    position: int = 0             # signed contract count (positive=long, negative=short, 0=flat)
    entry_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    highest_price_since_entry: float = 0.0
    stop_order_id: Optional[int] = None

    # Entry guard state (replaces overlapping _trade_active / _pending_exit_cooldown / cooldown_until)
    trades_today: int = 0
    trades_this_session: int = 0
    current_session: Optional[str] = None
    cooldown_until: Optional[datetime] = None
    last_exit_time: Optional[datetime] = None
    _trade_active: bool = False
    _pending_exit_cooldown: bool = False

    # VWAP MR state (eval)
    bars_5m: list = field(default_factory=list)
    vwap: float = 0.0

    # Funded payout tracking (Consistency Path)
    winning_days: int = 0
    profit_since_payout: float = 0.0
    max_day_pnl_since_payout: float = 0.0  # Largest single day for 40% rule
    lifetime_extracted: float = 0.0
    total_payouts: int = 0
    total_extracted: float = 0.0
    has_taken_payout: bool = False  # After first payout, DD floor = $0
    _payout_notified: bool = False

    # Real-time trade log (from WebSocket)
    rt_trades_today: list = field(default_factory=list)
    session_trades_log: list = field(default_factory=list)  # trades in current session for TG summary

    # Status
    active: bool = True
    last_error: str = ""

    def entry_blocked(self, now: datetime) -> EntryGuard:
        """Single source of truth for whether entries are blocked and why.

        Checks guards in priority order. Returns EntryGuard.CLEAR when ready to trade.
        """
        if self._trade_active:
            # Safety release: if position is flat and stuck for too long,
            # the caller should force-clear before re-checking.
            return EntryGuard.TRADE_ACTIVE

        if self._pending_exit_cooldown:
            return EntryGuard.PENDING_EXIT

        if self.last_exit_time is not None:
            if (now - self.last_exit_time).total_seconds() < MIN_EXIT_GAP_SECS:
                return EntryGuard.EXIT_GAP

        if self.cooldown_until is not None and now < self.cooldown_until:
            return EntryGuard.COOLDOWN

        return EntryGuard.CLEAR

    def clear_position_state(self):
        """Reset all position-related fields to flat."""
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.target_price = 0.0
        self.highest_price_since_entry = 0.0
        self.stop_order_id = None

    def apply_loss_cooldown(self, now: datetime):
        """Set the post-loss cooldown timer."""
        self.cooldown_until = now + timedelta(minutes=LOSS_COOLDOWN_MINUTES)

    def cooldown_remaining(self, now: datetime) -> Optional[timedelta]:
        """Return remaining cooldown duration, or None if not in cooldown."""
        if self.cooldown_until is not None and now < self.cooldown_until:
            return self.cooldown_until - now
        return None



# =========================================================================
# VWAP MR Signal Computation (real-time on 5m bars)
# =========================================================================

# -- Bar field accessors (API returns t/o/h/l/c/v shorthand) --------------

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


def compute_vwap(bars: List[Dict]) -> float:
    """Compute VWAP from today's bars."""
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
    avg_gain = sum(max(d, 0) for d in deltas[:period]) / period
    avg_loss = sum(max(-d, 0) for d in deltas[:period]) / period
    alpha = 1.0 / period
    for d in deltas[period:]:
        avg_gain = alpha * max(d, 0) + (1 - alpha) * avg_gain
        avg_loss = alpha * max(-d, 0) + (1 - alpha) * avg_loss
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _compute_day_bias(bars: List[Dict], config: dict) -> dict:
    """Compute day bias features matching backtest _session_day_features.

    Returns dict with session_open, first_window_move.
    """
    session_start = config.get("session_start_min", 9 * 60 + 30)  # default NY open
    first_window_end = session_start + config.get("day_bias_first_window_minutes", 15)

    session_open = None
    first_window_close = None

    for b in bars:
        dt = pd.to_datetime(b["t"])
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt_et = dt.astimezone(ET)
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


def _check_day_bias(bars: List[Dict], config: dict, prev_day_close: Optional[float],
                     bearish: bool = False) -> bool:
    """Check day bias filter conditions. Returns True if bias is favorable.

    For longs: gap UP + first window move UP.
    For shorts (bearish=True): gap DOWN + first window move DOWN.
    Returns True (pass) if day_bias_filter is disabled.
    """
    if not config.get("day_bias_filter", False):
        return True

    bias = _compute_day_bias(bars, config)
    session_open = bias["session_open"]
    first_window_move = bias["first_window_move"]
    min_gap = config.get("day_bias_min_gap_pts", 3.0)
    min_fw_move = config.get("day_bias_min_first_window_move_pts", 0.0)

    if prev_day_close is None or session_open is None or first_window_move is None:
        return False

    gap = session_open - prev_day_close

    if bearish:
        return gap <= -min_gap and first_window_move <= -min_fw_move
    else:
        return gap >= min_gap and first_window_move >= min_fw_move


def check_vwap_mr_entry(bars: List[Dict], config: dict,
                         prev_day_close: float = None) -> bool:
    """Check VWAP Micro MR LONG entry conditions on current 5m bar.

    Matches backtest (lm.py run_topstep_nq) exactly:
      1. c < prev_bar_close  (price dropping)
      2. dip >= 1.5 pts below VWAP  (high_prob)
      3. bearish bar >= 0.5 pt  (high_prob)
      4. RSI < 50  (disabled by default)
      5. day_bias_filter if enabled (disabled by default)
    """
    if len(bars) < 2:
        return False

    price = _bar_close(bars[-1])
    o = _bar_open(bars[-1])
    prev_bar_close = _bar_close(bars[-2])
    vwap = compute_vwap(bars)

    if price >= prev_bar_close:
        return False
    if vwap - price < config["min_dip_pts"]:
        return False
    if price >= o or o - price < config["min_down_bar_pts"]:
        return False

    closes = [_bar_close(b) for b in bars]
    if compute_rsi(closes, 14) >= config["rsi_oversold_max"]:
        return False

    return _check_day_bias(bars, config, prev_day_close, bearish=False)


def check_vwap_mr_short_entry(bars: List[Dict], config: dict,
                               prev_day_close: float = None) -> bool:
    """Check VWAP Micro MR SHORT entry conditions (mirror of long).

    Matches backtest (lm.py run_topstep_nq) short logic:
      1. c > prev_bar_close  (price rising)
      2. spike >= 1.5 pts above VWAP  (high_prob)
      3. bullish bar >= 0.5 pt  (high_prob)
      4. RSI > 50  (disabled by default)
      5. day_bias_filter if enabled (disabled by default)
    """
    if len(bars) < 2:
        return False

    price = _bar_close(bars[-1])
    o = _bar_open(bars[-1])
    prev_bar_close = _bar_close(bars[-2])
    vwap = compute_vwap(bars)

    if price <= prev_bar_close:
        return False
    if price - vwap < config["min_dip_pts"]:
        return False
    if price <= o or price - o < config["min_down_bar_pts"]:
        return False

    closes = [_bar_close(b) for b in bars]
    if compute_rsi(closes, 14) <= config.get("rsi_overbought_min", 55.0):
        return False

    return _check_day_bias(bars, config, prev_day_close, bearish=True)


def pts_to_ticks(pts: float) -> int:
    """Convert points to NQ ticks (4 ticks per point)."""
    return int(pts / TICK_SIZE)


# =========================================================================
# Dashboard Builder
# =========================================================================

class DashboardBuilder:
    """Renders the Rich TUI dashboard. Separated from trading logic for clarity."""

    def __init__(self, engine: "TradingEngine"):
        self._engine = engine

    def build(self) -> Table:
        """Build the full dashboard as a Rich Table grid."""
        eng = self._engine
        now = eng._now_et()
        market_open = eng._is_market_hours(now)
        in_session = eng._is_eval_session(now)

        outer = Table.grid(padding=(0, 0))
        outer.add_row(self._build_header(now, market_open, in_session))
        outer.add_row(Text(""))
        outer.add_row(self._build_market_conditions(market_open, in_session))
        outer.add_row(Text(""))
        outer.add_row(self._build_accounts_table(now, market_open, in_session))
        outer.add_row(Text(""))

        trade_table = self._build_trade_history()
        if trade_table is not None:
            outer.add_row(trade_table)

        if eng._activity_log:
            outer.add_row(Text(""))
            outer.add_row(self._build_activity_log())

        return outer

    def _build_header(self, now: datetime, market_open: bool, in_session: bool) -> Text:
        eng = self._engine
        market_str = "[bold green]OPEN[/]" if market_open else "[bold red]CLOSED[/]"
        cur_session = get_current_session(now)
        session_str = f"[green]{cur_session}[/]" if cur_session else "[dim]OUT[/]"

        pos_str = "[dim]FLAT[/]"
        for acc in eng.accounts.values():
            if acc.position > 0:
                pos_str = f"[bold green]{acc.position}x LONG @ {acc.entry_price:,.2f}[/]"
            elif acc.position < 0:
                pos_str = f"[bold red]{abs(acc.position)}x SHORT @ {acc.entry_price:,.2f}[/]"
            break

        rt_str = "[green]WS[/]" if eng._rt_connected else "[red]REST[/]"
        rt_price = getattr(eng, '_last_price', None)
        rt_price_str = f"  |  RT: ${rt_price:,.2f}" if rt_price else ""

        return Text.from_markup(
            f" [bold]TopstepX Bot[/]  |  {now.strftime('%Y-%m-%d %H:%M:%S ET')}  |  "
            f"Market: {market_str}  |  Session: {session_str}  |  "
            f"Position: {pos_str}  |  {eng.contract_name}  |  "
            f"Feed: {rt_str}{rt_price_str}"
        )

    def _build_market_conditions(self, market_open: bool, in_session: bool) -> Panel:
        """Build the market conditions panel using data from the first eval account."""
        eng = self._engine
        price_str = vwap_str = dip_str = rsi_str = dist_str = "-"
        bias_str = "-"
        entry_str = "[dim]NO[/]"

        bars = None
        prev_day_close = self._get_prev_day_close()

        for acc in eng.accounts.values():
            if not acc.bars_5m:
                continue
            bars = acc.bars_5m
            price = _bar_close(bars[-1])
            vwap = acc.vwap
            price_str = f"{price:,.2f}"
            vwap_str = f"{vwap:,.2f}"

            if vwap > 0:
                dip = vwap - price
                dip_color = "green" if dip >= 2.5 else "red"
                dip_str = f"[{dip_color}]{dip:+.2f} pts[/]"

                entry_level = vwap - EVAL_CONFIG["min_dip_pts"]
                distance = price - entry_level
                if distance > 0:
                    dist_str = f"[yellow]{distance:.2f} pts above[/] ({entry_level:,.2f})"
                else:
                    dist_str = f"[green]IN ZONE[/] ({abs(distance):.2f} below)"

            closes = [_bar_close(b) for b in bars]
            rsi = compute_rsi(closes, 14)
            rsi_color = "green" if rsi < 45 else "red"
            rsi_str = f"[{rsi_color}]{rsi:.1f}[/]"

            bias_str = self._format_day_bias(bars, prev_day_close)
            entry_str = self._format_entry_signal(bars, prev_day_close)
            break

        cond = Table.grid(padding=(0, 4))
        cond.add_column(style="bold", width=16)
        cond.add_column(width=24)
        cond.add_column(style="bold", width=16)
        cond.add_column(width=24)
        cond.add_row("Price:", price_str, "VWAP:", vwap_str)
        cond.add_row("Dip from VWAP:", dip_str, "RSI(14):", rsi_str)
        cond.add_row("Day Bias:", bias_str, "Entry Signal:", entry_str)
        cond.add_row("Dist to Entry:", dist_str, "", "")

        return Panel(cond, title="Market Conditions", border_style="cyan", expand=True)

    def _format_day_bias(self, bars: List[Dict], prev_day_close: Optional[float]) -> str:
        if not EVAL_CONFIG.get("day_bias_filter", False):
            return "[green]ANY[/] (no bias filter)"

        bias = _compute_day_bias(bars, EVAL_CONFIG)
        min_gap = EVAL_CONFIG.get("day_bias_min_gap_pts", 3.0)
        min_fw = EVAL_CONFIG.get("day_bias_min_first_window_move_pts", 0.0)

        if bias["session_open"] is None or prev_day_close is None:
            return "[dim]N/A[/]"

        gap = bias["session_open"] - prev_day_close
        gap_bull = gap >= min_gap
        fw_bull = bias["first_window_move"] is not None and bias["first_window_move"] >= min_fw
        gap_bear = gap <= -min_gap
        fw_bear = bias["first_window_move"] is not None and bias["first_window_move"] <= -min_fw

        if gap_bull and fw_bull:
            return f"[green]BULL[/] (gap {gap:+.1f})"
        elif gap_bear and fw_bear:
            return f"[yellow]BEAR[/] (gap {gap:+.1f})"
        else:
            return f"[red]FLAT[/] (gap {gap:+.1f})"

    def _format_entry_signal(self, bars: List[Dict], prev_day_close: Optional[float]) -> str:
        long_met = check_vwap_mr_entry(bars, EVAL_CONFIG, prev_day_close=prev_day_close)
        short_met = (
            not EVAL_CONFIG.get("long_only", True)
            and check_vwap_mr_short_entry(bars, EVAL_CONFIG, prev_day_close=prev_day_close)
        )
        if long_met:
            return "[bold green]LONG[/]"
        elif short_met:
            return "[bold yellow]SHORT[/]"
        return "[dim]NO[/]"

    def _get_prev_day_close(self) -> Optional[float]:
        eng = self._engine
        if eng._daily_bars is not None and len(eng._daily_bars) >= 2:
            today_utc = datetime.now(timezone.utc).date()
            prev_days = eng._daily_bars[eng._daily_bars["date"] < today_utc]
            if not prev_days.empty:
                return float(prev_days.iloc[-1]["close"])
        return None

    def _build_accounts_table(self, now: datetime, market_open: bool, in_session: bool) -> Table:
        eng = self._engine
        tbl = Table(show_lines=True, expand=True, show_header=True, padding=(0, 2))
        tbl.add_column("Account", style="bold", ratio=4, no_wrap=True)
        tbl.add_column("Phase", justify="center", ratio=1)
        tbl.add_column("Equity", justify="right", ratio=2)
        tbl.add_column("DD Left", justify="right", ratio=2)
        tbl.add_column("Day PnL", justify="right", ratio=2)
        tbl.add_column("Position", justify="center", ratio=2)
        tbl.add_column("Trades", justify="center", ratio=2)
        tbl.add_column("Status", justify="center", ratio=2)

        for acc in eng.accounts.values():
            if acc.phase == Phase.FUNDED and acc.has_taken_payout:
                # After first payout, DD floor = $0 — can only lose remaining equity
                dd_left = min(acc.equity, FUNDED_TRAILING_DD - (acc.peak - acc.equity))
            else:
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

            if acc.position > 0:
                pos = f"{acc.position}x LONG"
            elif acc.position < 0:
                pos = f"{abs(acc.position)}x SHORT"
            else:
                pos = "flat"
            pos_text = Text(pos)
            pos_text.stylize("cyan" if acc.position != 0 else "dim")

            status = self._account_status(acc, now, market_open, in_session)

            extra = ""
            if acc.phase == Phase.FUNDED:
                max_day_limit = acc.profit_since_payout * CONSISTENCY_PCT if acc.profit_since_payout > 0 else 0
                consistency_ok = acc.max_day_pnl_since_payout <= max_day_limit if max_day_limit > 0 else True
                cons_mark = "✓" if consistency_ok else "✗"
                extra = (f"\nW-days: {acc.winning_days}/{WINNING_DAYS_REQUIRED} | "
                         f"40%: ${acc.max_day_pnl_since_payout:,.0f}/${max_day_limit:,.0f} {cons_mark} | "
                         f"Ext: ${acc.total_extracted:,.0f}")

            tbl.add_row(
                f"{acc.name} ({acc.account_id}){extra}",
                phase_text,
                equity_text,
                dd_text,
                day_text,
                pos_text,
                f"{acc.trades_this_session}/{EVAL_CONFIG['max_trades_per_session']} ({acc.trades_today}d)",
                status,
            )
        return tbl

    def _account_status(self, acc: AccountState, now: datetime,
                        market_open: bool, in_session: bool) -> Text:
        if not acc.active:
            return Text("BUST", style="bold red")

        if acc.phase == Phase.EVAL and acc.equity >= EVAL_PROFIT_TARGET:
            return Text("PASSED!", style="bold green")

        if acc._payout_notified:
            return Text("PAYOUT READY", style="bold green")

        if acc.position != 0:
            cur_px = _bar_close(acc.bars_5m[-1]) if acc.bars_5m else acc.entry_price
            # Use signed position for correct unrealized P&L (negative for shorts)
            unrealized = (cur_px - acc.entry_price) * POINT_VALUE * acc.position
            ucolor = "green" if unrealized >= 0 else "red"
            return Text(f"OPEN ${unrealized:+,.0f}", style=f"bold {ucolor}")

        if not market_open:
            return Text("MKT CLOSED", style="dim")

        if not in_session:
            return Text("BETWEEN SESSIONS", style="dim")

        remaining = acc.cooldown_remaining(now)
        if remaining is not None:
            total_secs = int(remaining.total_seconds())
            hours = total_secs // 3600
            mins = (total_secs % 3600) // 60
            if hours > 0:
                return Text(f"COOLDOWN: {hours}h {mins}m", style="bold red")
            else:
                secs = total_secs % 60
                return Text(f"COOLDOWN: {mins}m {secs}s", style="bold red")

        if acc._pending_exit_cooldown:
            return Text("PENDING EXIT", style="bold magenta")

        if acc._trade_active:
            return Text("TRADE ACTIVE", style="bold cyan")

        return Text("SCANNING", style="yellow")

    def _build_trade_history(self) -> Optional[Table]:
        """Build today's trade history table from real-time data."""
        eng = self._engine
        for acc in eng.accounts.values():
            exit_trades = acc.rt_trades_today
            if not exit_trades:
                break

            tbl = Table(
                show_lines=False, expand=True,
                title="Today's Trades (live)",
                title_style="bold cyan",
            )
            tbl.add_column("Time (ET)", ratio=2)
            tbl.add_column("Side", justify="center", ratio=1)
            tbl.add_column("Size", justify="center", ratio=1)
            tbl.add_column("Price", justify="right", ratio=2)
            tbl.add_column("PnL", justify="right", ratio=2)
            tbl.add_column("Fees", justify="right", ratio=1)

            for t in exit_trades:
                ts_raw = t.get("creationTimestamp") or t.get("timestamp", "")
                try:
                    ts = pd.to_datetime(ts_raw)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    time_str = ts.astimezone(ET).strftime("%H:%M:%S")
                except Exception:
                    time_str = "?"

                # RT trade events report the EXIT side -- flip to show entry side
                side_str = "SELL" if t.get("side") == 0 else "BUY"
                side_style = "green" if t.get("side") == 1 else "red"
                price_str = f"{float(t.get('price', 0)):,.2f}"
                pnl = float(t.get("profitAndLoss", 0))
                fees = float(t.get("fees", 0))
                net = pnl - fees
                net_style = "green" if net >= 0 else "red"
                tbl.add_row(
                    time_str,
                    f"[{side_style}]{side_str}[/]",
                    str(t.get("size", "")),
                    price_str,
                    f"[{net_style}]${pnl:+,.2f}[/]",
                    f"${fees:.2f}",
                )

            total_pnl = sum(float(t.get("profitAndLoss", 0)) for t in exit_trades)
            total_fees = sum(float(t.get("fees", 0)) for t in exit_trades)
            net_total = total_pnl - total_fees
            ns = "green" if net_total >= 0 else "red"
            tbl.add_row(
                "[bold]TOTAL[/]", "", "", "",
                f"[bold {ns}]${net_total:+,.2f}[/]",
                f"${total_fees:.2f}",
            )
            return tbl
        return None

    def _build_activity_log(self) -> Table:
        eng = self._engine
        tbl = Table(
            show_lines=False, expand=True,
            title="Activity Log",
            title_style="bold cyan",
            show_header=False,
        )
        tbl.add_column("Time", width=10, style="dim")
        tbl.add_column("Event", ratio=1)
        for ts, msg, style in reversed(eng._activity_log):
            tbl.add_row(ts, f"[{style}]{msg}[/]" if style else msg)
        return tbl


# =========================================================================
# Trading Engine
# =========================================================================

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

        # Real-time state
        self._rt: Optional[RealtimeClient] = None
        self._last_quote: Dict = {}
        self._rt_connected = False

        # Activity log for dashboard (rolling last 20 events)
        self._activity_log: deque = deque(maxlen=20)

        # Dashboard renderer
        self._dashboard = DashboardBuilder(self)

    def _log_activity(self, msg: str, style: str = ""):
        ts = self._now_et().strftime("%H:%M:%S")
        self._activity_log.append((ts, msg, style))

    # -- Telegram Helpers --------------------------------------------------

    def _notify_session_end(self, acc: AccountState, ended_session: str):
        """Send Telegram summary when a trading session ends."""
        if _telegram is None:
            return

        phase_str = acc.phase.value.upper()
        trades = acc.session_trades_log
        session_pnl = sum(float(t.get("profitAndLoss", 0)) - float(t.get("fees", 0)) for t in trades)

        lines = [
            f"<b>Session End: {ended_session}</b>",
            f"Account: {acc.name}",
            f"Phase: {phase_str}",
            f"Session PnL: <b>${session_pnl:+,.2f}</b>",
            f"Trades: {len(trades)}",
        ]

        for i, t in enumerate(trades, 1):
            ts_raw = t.get("creationTimestamp") or t.get("timestamp", "")
            try:
                ts = pd.to_datetime(ts_raw)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                time_str = ts.astimezone(ET).strftime("%I:%M %p")
            except Exception:
                time_str = "?"
            pnl = float(t.get("profitAndLoss", 0))
            fees = float(t.get("fees", 0))
            net = pnl - fees
            side_str = "SELL" if t.get("side") == 0 else "BUY"
            lines.append(f"  {i}. {time_str} {side_str} → ${net:+,.2f}")

        lines.append(f"\nDay PnL: ${acc.daily_pnl:+,.2f}")
        lines.append(f"Equity: ${acc.equity:+,.2f}")

        _telegram.send("\n".join(lines))

    def _notify_eval_passed(self, acc: AccountState):
        """Send Telegram alert when an eval account passes."""
        if _telegram is None:
            return
        _telegram.send(
            f"🎉 <b>EVAL PASSED!</b>\n"
            f"Account: {acc.name}\n"
            f"Equity: ${acc.equity:+,.2f}\n"
            f"Trading Days: {acc.trading_days}\n\n"
            f"Account is now ready to transition to FUNDED!"
        )

    def _notify_payout_ready(self, acc: AccountState):
        """Send Telegram alert when a funded account is eligible for payout."""
        if _telegram is None:
            return
        gross_payout = min(acc.profit_since_payout * PAYOUT_BAL_PCT, PAYOUT_MAX)
        net_payout = gross_payout * 0.90  # 90/10 split
        _telegram.send(
            f"💰 <b>PAYOUT READY!</b> (Consistency Path)\n"
            f"Account: {acc.name}\n"
            f"Winning Days: {acc.winning_days}/{WINNING_DAYS_REQUIRED}\n"
            f"Profit Since Payout: ${acc.profit_since_payout:+,.2f}\n"
            f"Max Day: ${acc.max_day_pnl_since_payout:,.2f} "
            f"(limit: ${acc.profit_since_payout * CONSISTENCY_PCT:,.2f})\n"
            f"Request Payout: ${gross_payout:,.2f}\n"
            f"You Receive: ${net_payout:,.2f} (90/10 split)"
        )

    # -- Setup -------------------------------------------------------------

    def setup(self):
        """Initialize: auth, find accounts, find NQ contract."""
        log.info("Setting up trading engine...")

        if not self.api.authenticate():
            raise ConnectionError("Failed to authenticate with TopstepX")

        nq = self.api.get_nq_contract()
        if not nq:
            raise RuntimeError("Could not find NQ contract")
        self.contract_id = nq.get("id", "")
        self.contract_name = nq.get("name", "")
        log.info(f"NQ contract: {self.contract_id} ({self.contract_name})")

        raw_accounts = self.api.get_accounts()
        if not raw_accounts:
            raise RuntimeError("No trading accounts found")

        for acc in raw_accounts:
            acc_id = acc["id"]
            name = acc.get("name", "")
            can_trade = acc.get("canTrade", True)

            name_lower = name.lower()

            # Skip practice/sim accounts
            if "prac" in name_lower or "sim" in name_lower:
                log.info(f"Skipping practice account: [{acc_id}] {name}")
                continue

            is_funded = any(
                kw in name_lower for kw in ["funded", "live", "express", "payout"]
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

        self._load_state()
        log.info(f"Engine ready: {len(self.accounts)} account(s), contract={self.contract_id}")

        self._startup_cooldown_check()
        self._start_realtime()

    def _startup_cooldown_check(self):
        """On startup, check the most recent closing trade.

        If it was a loss, still within the same session, and less than
        LOSS_COOLDOWN_MINUTES ago, apply the remaining cooldown.
        """
        try:
            now = self._now_et()
            current_session = get_current_session(now)
            today_start = datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT00:00:00Z"
            )

            for acc in self.accounts.values():
                # Set current session so the first loop tick doesn't
                # treat it as a session change and wipe our cooldown
                acc.current_session = current_session

                trades = self.api.get_trades(
                    acc.account_id, today_start
                )
                if not trades:
                    continue

                # Find the most recent closing trade (has PnL)
                closing = [
                    t for t in trades
                    if t.get("profitAndLoss") is not None
                ]
                closing.sort(
                    key=self._parse_trade_timestamp, reverse=True
                )

                if closing:
                    last = closing[0]
                    pnl = float(last["profitAndLoss"])
                    fees = float(last.get("fees", 0))
                    net_pnl = pnl - fees
                    last_ts = self._parse_trade_timestamp(last)
                    last_et = last_ts.astimezone(ET)
                    trade_session = get_current_session(last_et)
                    mins_ago = (
                        datetime.now(timezone.utc) - last_ts
                    ).total_seconds() / 60

                    if (net_pnl < 0
                            and mins_ago < LOSS_COOLDOWN_MINUTES
                            and trade_session == current_session
                            and current_session is not None):
                        remaining = LOSS_COOLDOWN_MINUTES - mins_ago
                        acc.cooldown_until = now + timedelta(
                            minutes=remaining
                        )
                        acc.last_exit_time = now
                        log.info(
                            f"Startup cooldown: acct {acc.account_id} "
                            f"| last trade ${net_pnl:+.2f} "
                            f"({mins_ago:.0f}min ago, {trade_session}) "
                            f"| cooldown until "
                            f"{acc.cooldown_until.strftime('%H:%M')}"
                        )
                    elif net_pnl < 0:
                        log.info(
                            f"Startup: acct {acc.account_id} "
                            f"| last loss ${net_pnl:+.2f} "
                            f"({mins_ago:.0f}min ago, {trade_session}) "
                            f"| session={'same' if trade_session == current_session else 'changed'} "
                            f"| no cooldown"
                        )
                    else:
                        log.info(
                            f"Startup: acct {acc.account_id} "
                            f"| last trade ${net_pnl:+.2f} "
                            f"| no cooldown"
                        )

                # Populate rt_trades_today for dashboard
                acc.rt_trades_today = [
                    t for t in trades
                    if t.get("profitAndLoss") is not None
                ]
                acc.trades_today = len(acc.rt_trades_today)
        except Exception as e:
            log.warning(
                f"Startup cooldown check failed: {e}", exc_info=True
            )

    @staticmethod
    def _parse_trade_timestamp(trade: dict) -> datetime:
        """Parse a trade's timestamp into a timezone-aware datetime."""
        ts = trade.get("creationTimestamp") or trade.get("timestamp", "")
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    def _start_realtime(self):
        """Connect SignalR WebSocket for real-time updates."""
        try:
            self.api.ensure_auth()
            self._rt = RealtimeClient(token_factory=lambda: self.api.token or "")
            self._rt.on_account_update = self._on_rt_account
            self._rt.on_position_update = self._on_rt_position
            self._rt.on_trade_update = self._on_rt_trade
            self._rt.on_order_update = self._on_rt_order
            self._rt.on_quote_update = self._on_rt_quote

            self._rt.subscribe(
                account_ids=list(self.accounts.keys()),
                contract_ids=[self.contract_id],
            )
            self._rt.start()
            self._rt_connected = True
            log.info("Real-time WebSocket started")
        except Exception as e:
            log.error(f"Failed to start real-time WebSocket: {e}", exc_info=True)
            self._rt_connected = False

    # -- Real-Time Event Handlers ------------------------------------------

    def _on_rt_account(self, data: dict):
        """Handle real-time account balance update."""
        acc_id = data.get("accountId") or data.get("id")
        if not acc_id or acc_id not in self.accounts:
            return
        acc = self.accounts[acc_id]

        balance = data.get("balance")
        if balance is not None:
            bal = float(balance)
            acc.equity = bal - START_BALANCE
            acc.peak = max(acc.peak, acc.equity)
            log.debug(f"RT balance update: acct {acc_id} = ${bal:,.2f} (equity ${acc.equity:,.2f})")

        can_trade = data.get("canTrade")
        if can_trade is not None:
            acc.can_trade = bool(can_trade)

    def _on_rt_position(self, data: dict):
        """Handle real-time position change."""
        acc_id = data.get("accountId")
        if not acc_id or acc_id not in self.accounts:
            return
        acc = self.accounts[acc_id]
        if data.get("contractId") != self.contract_id:
            return

        new_size = int(data.get("size", 0))
        avg_price = data.get("averagePrice")

        if new_size != acc.position:
            log.info(f"RT position change: acct {acc_id} | {acc.position} -> {new_size}")

        was_in_position = acc.position != 0
        acc.position = new_size
        if avg_price is not None:
            acc.entry_price = float(avg_price)

        if new_size == 0:
            # Position went flat on exchange -- block entries until RT trade confirms PnL
            if was_in_position:
                acc._pending_exit_cooldown = True
                acc.last_exit_time = self._now_et()
                self._cancel_stale_orders(acc)
            acc.clear_position_state()

    def _on_rt_trade(self, data: dict):
        """Handle real-time trade execution (fills with PnL)."""
        acc_id = data.get("accountId")
        if not acc_id or acc_id not in self.accounts:
            return
        acc = self.accounts[acc_id]

        pnl = data.get("profitAndLoss")
        fees = float(data.get("fees", 0))

        if pnl is None:
            return

        # Exit trade with realized PnL
        net = float(pnl) - fees
        acc.daily_pnl += net
        acc.profit_since_payout += net
        # Track largest single-day PnL for 40% consistency rule
        if acc.daily_pnl > acc.max_day_pnl_since_payout:
            acc.max_day_pnl_since_payout = acc.daily_pnl
        acc.rt_trades_today.append(data)
        acc.session_trades_log.append(data)
        acc._pending_exit_cooldown = False
        acc._trade_active = False
        log.info(
            f"RT trade: acct {acc_id} | "
            f"PnL=${float(pnl):+,.2f} fees=${fees:.2f} net=${net:+,.2f} | "
            f"profit_since_payout=${acc.profit_since_payout:+,.2f}"
        )
        pnl_style = "green" if net >= 0 else "red"
        self._log_activity(f"Trade closed: ${net:+,.2f} (day: ${acc.daily_pnl:+,.2f})", pnl_style)

        if net < 0:
            acc.apply_loss_cooldown(self._now_et())
            log.info(f"Loss cooldown: acct {acc_id} | no entries until {acc.cooldown_until.strftime('%H:%M')}")
            self._log_activity(f"Cooldown until {acc.cooldown_until.strftime('%H:%M')} ({LOSS_COOLDOWN_MINUTES}min)", "yellow")

        # Update trade count
        acc.trades_today = len([
            t for t in acc.rt_trades_today if t.get("profitAndLoss") is not None
        ]) + (1 if acc.position != 0 else 0)

        # Check trailing DD bust (after payout, floor = $0)
        drawdown = acc.peak - acc.equity
        max_dd = EVAL_TRAILING_DD if acc.phase == Phase.EVAL else FUNDED_TRAILING_DD
        busted = drawdown >= max_dd
        if acc.phase == Phase.FUNDED and acc.has_taken_payout and acc.equity <= 0:
            busted = True
        if busted:
            acc.active = False
            log.warning(f"RT BUST detected: acct {acc_id} | DD ${drawdown:,.2f}")

    def _on_rt_order(self, data: dict):
        """Handle real-time order status update. Capture stop order ID for trailing."""
        acc_id = data.get("accountId")
        if not acc_id or acc_id not in self.accounts:
            return
        acc = self.accounts[acc_id]
        order_type = data.get("type", 0)
        order_id = data.get("id") or data.get("orderId")
        log.debug(f"RT order: acct {acc_id} | id={order_id} type={order_type} status={data.get('status', '')}")

        if order_type == OrderType.STOP and order_id and acc.position != 0:
            acc.stop_order_id = int(order_id)
            log.info(f"Captured stop order ID: {order_id} for acct {acc_id}")

    def _resolve_pending_exit(self, acc: AccountState):
        """Fallback: resolve PENDING_EXIT via REST API when RT trade event was missed."""
        now = self._now_et()
        acc_id = acc.account_id
        log.warning(f"Pending exit timeout ({PENDING_EXIT_TIMEOUT_SECS}s) — resolving via REST for acct {acc_id}")

        # Fetch recent trades to find the closing trade's PnL
        start_ts = (now - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S")
        try:
            trades = self.api.get_trades(acc_id, start_ts)
        except Exception as e:
            log.error(f"REST trade fetch failed for acct {acc_id}: {e}")
            trades = []

        # Skip trades already processed via RT (match by trade id)
        seen_ids = {t.get("id") for t in acc.rt_trades_today if t.get("id")}

        # Find the most recent closing trade not already processed
        closing_trades = [
            t for t in (trades or [])
            if t.get("profitAndLoss") is not None
            and t.get("id") not in seen_ids
        ]
        closing_trades.sort(
            key=lambda t: t.get("creationTimestamp", ""), reverse=True
        )

        net = None
        if closing_trades:
            t = closing_trades[0]
            pnl = t.get("profitAndLoss")
            fees = float(t.get("fees", 0))
            net = float(pnl) - fees
            acc.daily_pnl += net
            acc.profit_since_payout += net
            acc.max_day_pnl_since_payout = max(
                acc.max_day_pnl_since_payout, acc.daily_pnl
            )
            acc.rt_trades_today.append(t)
            acc.session_trades_log.append(t)
            log.info(
                f"REST resolved: acct {acc_id} | "
                f"net=${net:+,.2f} | day=${acc.daily_pnl:+,.2f}"
            )
            pnl_style = "green" if net >= 0 else "red"
            self._log_activity(
                f"Trade closed (REST): ${net:+,.2f} "
                f"(day: ${acc.daily_pnl:+,.2f})", pnl_style
            )

        acc._pending_exit_cooldown = False
        acc._trade_active = False

        if net is not None and net < 0:
            acc.apply_loss_cooldown(now)
            log.info(f"Loss cooldown: acct {acc_id} | no entries until {acc.cooldown_until.strftime('%H:%M')}")
            self._log_activity(f"Cooldown until {acc.cooldown_until.strftime('%H:%M')} ({LOSS_COOLDOWN_MINUTES}min)", "yellow")
        elif net is None:
            # Couldn't find PnL — apply cooldown conservatively
            log.warning(f"Could not resolve PnL for acct {acc_id} — applying conservative cooldown")
            acc.apply_loss_cooldown(now)
            self._log_activity(f"Cooldown (unknown PnL) until {acc.cooldown_until.strftime('%H:%M')}", "yellow")

    def _cancel_stale_orders(self, acc: AccountState):
        """Cancel any remaining bracket orders after position goes flat."""
        try:
            orders = self.api.get_open_orders(acc.account_id)
            for o in orders:
                oid = int(o.get("id", o.get("orderId", 0)))
                if oid:
                    self.api.cancel_order(oid)
                    log.info(f"Cancelled stale order {oid} (type={o.get('type')}) on acct {acc.account_id}")
        except Exception as e:
            log.warning(f"Failed to cancel stale orders on acct {acc.account_id}: {e}")

    def _on_rt_quote(self, data: dict):
        """Handle real-time market quote from GatewayQuote."""
        self._last_quote = data
        last_price = data.get("lastPrice")
        if last_price is not None:
            self._last_price = float(last_price)

    # -- State Persistence -------------------------------------------------

    def _load_state(self):
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file) as f:
                saved = json.load(f)
            for acc_id_str, data in saved.get("accounts", {}).items():
                acc_id = int(acc_id_str)
                if acc_id not in self.accounts:
                    continue
                acc = self.accounts[acc_id]
                acc.equity = data.get("equity", 0)
                acc.peak = data.get("peak", 0)
                acc.trading_days = data.get("trading_days", 0)
                acc.winning_days = data.get("winning_days", 0)
                acc.profit_since_payout = data.get("profit_since_payout", 0)
                acc.lifetime_extracted = data.get("lifetime_extracted", 0)
                acc.total_payouts = data.get("total_payouts", 0)
                acc.total_extracted = data.get("total_extracted", 0)
                acc.has_taken_payout = data.get("has_taken_payout", acc.total_payouts > 0)
                acc.max_day_pnl_since_payout = data.get("max_day_pnl_since_payout", 0)
                acc.today = data.get("today", "")
                acc.daily_pnl = data.get("daily_pnl", 0)
                acc.trades_today = data.get("trades_today", 0)
                acc.phase = Phase(data.get("phase", "eval"))
            log.info("Loaded saved state")
        except Exception as e:
            log.warning(f"Could not load state: {e}", exc_info=True)

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
                "has_taken_payout": acc.has_taken_payout,
                "max_day_pnl_since_payout": acc.max_day_pnl_since_payout,
                "today": acc.today,
                "daily_pnl": acc.daily_pnl,
                "trades_today": acc.trades_today,
            }
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)

    # -- Time Helpers ------------------------------------------------------

    @staticmethod
    def _now_et() -> datetime:
        """Current time in US/Eastern."""
        return datetime.now(ET)

    @staticmethod
    def _is_market_hours(dt_et: datetime) -> bool:
        """NQ futures: Sun 6pm - Fri 5pm ET, with daily halt 5pm-6pm ET."""
        wd = dt_et.weekday()
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
        """True if we're in any trading session."""
        return get_current_session(dt_et) is not None

# -- API Sync (fallback when WebSocket is down) ------------------------

    def _sync_from_api(self, acc: AccountState):
        """Full REST sync -- used as fallback when WebSocket is unavailable."""
        try:
            self._sync_position_from_api(acc)
            self._sync_equity_from_api(acc)
            self._sync_daily_pnl_from_api(acc)
        except Exception as e:
            log.debug(f"API sync failed for acct {acc.account_id}: {e}", exc_info=True)

    def _sync_position_from_api(self, acc: AccountState):
        """Sync position state from REST API."""
        positions = self.api.get_positions(acc.account_id)
        nq_pos = None
        for p in (positions or []):
            if p.get("contractId") == acc.contract_id:
                nq_pos = p
                break

        if nq_pos:
            size = int(nq_pos.get("size", 0))
            if acc.position == 0 and size != 0:
                log.warning(
                    f"API sync: detected position {size} "
                    f"but bot thinks flat -- recovering bracket orders"
                )
                self._recover_stop_order(acc, nq_pos)
                self._log_activity(f"Recovered position: {size}x @ {acc.entry_price:.2f}", "yellow")
            acc.position = size
        else:
            if acc.position != 0:
                log.warning("API sync: no NQ position, syncing to flat")
                acc._pending_exit_cooldown = True
                acc.last_exit_time = self._now_et()
            acc.position = 0
            acc.entry_price = 0.0

    def _recover_stop_order(self, acc: AccountState, nq_pos: dict):
        """Try to recover the stop order ID and entry price after a reconnect."""
        try:
            orders = self.api.get_open_orders(acc.account_id)
            for o in orders:
                if o.get("type") in (OrderType.STOP, OrderType.TRAILING_STOP):
                    acc.stop_order_id = int(o.get("id", o.get("orderId", 0)))
                    stop_px = o.get("stopPrice") or o.get("price")
                    if stop_px:
                        acc.stop_price = float(stop_px)
                    log.info(f"Recovered stop order {acc.stop_order_id} @ {acc.stop_price}")
                    break
            avg_px = nq_pos.get("averagePrice")
            if avg_px:
                acc.entry_price = float(avg_px)
                acc.highest_price_since_entry = acc.entry_price
        except Exception as e:
            log.error(f"Failed to recover bracket orders: {e}", exc_info=True)

    def _sync_equity_from_api(self, acc: AccountState):
        """Sync equity from account balance.
        Detects payout processing: if equity drops while payout was pending,
        treat it as payout taken — reset peak, mark has_taken_payout.
        """
        accounts = self.api.get_accounts()
        for a in accounts:
            if a.get("id") == acc.account_id:
                api_balance = a.get("balance")
                if api_balance is not None:
                    bal = float(api_balance)
                    new_equity = bal - START_BALANCE
                    # Detect payout: equity dropped while payout was pending
                    if (acc._payout_notified
                            and acc.phase == Phase.FUNDED
                            and new_equity < acc.equity - 100):  # >$100 drop = payout taken
                        payout_amt = acc.equity - new_equity
                        acc.has_taken_payout = True
                        acc.total_payouts += 1
                        acc.total_extracted += payout_amt
                        acc.winning_days = 0
                        acc.profit_since_payout = 0.0
                        acc.max_day_pnl_since_payout = 0.0
                        # DD resets after payout — peak resets to new equity
                        acc.peak = new_equity
                        acc._payout_notified = False
                        log.info(f"PAYOUT DETECTED: acct {acc.account_id} | "
                                 f"${payout_amt:,.2f} withdrawn | "
                                 f"new equity=${new_equity:,.2f} | "
                                 f"total payouts={acc.total_payouts}")
                        self._log_activity(f"Payout #{acc.total_payouts}: ${payout_amt:,.2f} withdrawn", "green")
                    acc.equity = new_equity
                    acc.peak = max(acc.peak, acc.equity)
                break

    def _sync_daily_pnl_from_api(self, acc: AccountState):
        """Sync daily PnL from trades (source of truth)."""
        today_start = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z")
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
        acc.trades_today = trade_count + (1 if acc.position != 0 else 0)
        acc.rt_trades_today = [t for t in trades if t.get("profitAndLoss") is not None]

    # -- Daily Reset -------------------------------------------------------

    def _check_new_day(self):
        """Reset daily state when the date rolls over."""
        today_str = str(self._now_et().date())
        for acc in self.accounts.values():
            if acc.today == today_str:
                continue
            if acc.today:  # not first run
                if acc.daily_pnl >= WINNING_DAY_THRESHOLD:
                    acc.winning_days += 1
                acc.trading_days += 1

            acc.today = today_str
            acc.daily_pnl = 0.0
            acc.trades_today = 0
            acc.trades_this_session = 0
            acc.current_session = None
            acc.rt_trades_today = []
            acc.cooldown_until = None
            acc.last_exit_time = None
            if hasattr(acc, '_day_start_balance'):
                del acc._day_start_balance
            log.info(f"Account {acc.account_id}: new day {today_str}")

    # -- Build Daily Bars for Previous Day Close ----------------------------

    @staticmethod
    def _futures_session_date(dt: datetime) -> date:
        """Map a bar timestamp to the correct futures trading session date.

        NQ session: 6pm ET (prev day) -> 5pm ET (current day).
        Bars from 6pm-midnight belong to the NEXT calendar date's session.
        Bars from midnight-5pm belong to the current calendar date's session.
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        et_dt = dt.astimezone(ET)
        if et_dt.hour >= 17:
            return (et_dt + timedelta(days=1)).date()
        return et_dt.date()

    def _refresh_daily_bars(self):
        """Fetch historical 5m bars and aggregate to daily OHLCV for prev-day close."""
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
            log.error(f"Failed to refresh daily bars: {e}", exc_info=True)

    # -- Fetch Today's 5m Bars ---------------------------------------------

    def _get_today_bars(self, now_et: datetime) -> List[Dict]:
        """Fetch today's 5m bars for VWAP calculation.

        Uses UTC calendar date to match backtest (lm.py trading_day = dt.date).
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
            session_bars.sort(key=lambda b: b.get("t", ""))
            return session_bars
        except Exception as e:
            log.error(f"Failed to get today's bars: {e}", exc_info=True)
            return []

    # -- Prev Day Close (shared between strategy + dashboard) ---------------

    def _get_prev_day_close(self) -> Optional[float]:
        if self._daily_bars is not None and len(self._daily_bars) >= 2:
            today_utc = datetime.now(timezone.utc).date()
            prev_days = self._daily_bars[self._daily_bars["date"] < today_utc]
            if not prev_days.empty:
                return float(prev_days.iloc[-1]["close"])
        return None

    # -- Core Trading Logic ------------------------------------------------

    def _trade_eval(self, acc: AccountState):
        """VWAP Micro MR strategy for eval accounts."""
        now = self._now_et()

        # Always fetch bars so dashboard has data even outside session
        bars = self._get_today_bars(now)
        if bars:
            acc.bars_5m = bars
            acc.vwap = compute_vwap(bars)

        # Use real-time price if available, fallback to bar close
        current_price = getattr(self, '_last_price', None) or (_bar_close(bars[-1]) if bars else None)

        # Trail the stop on every cycle when in a position (regardless of other guards)
        if acc.position != 0 and current_price is not None:
            self._trail_stop(acc, current_price)
            return

        # Session tracking: reset trades + cooldown on session change
        session_name = get_current_session(now)
        if session_name != acc.current_session:
            prev_session = acc.current_session
            if prev_session is not None and acc.session_trades_log:
                self._notify_session_end(acc, prev_session)
            if session_name is not None:
                log.info(
                    f"Account {acc.account_id}: session changed "
                    f"to {session_name} (was {prev_session}), "
                    f"resetting session trade count + cooldown"
                )
            acc.current_session = session_name
            acc.trades_this_session = 0
            acc.cooldown_until = None
            acc.session_trades_log = []

        # Close position outside session window
        if session_name is None:
            if acc.position != 0:
                self._close_position(acc, "session_end")
            return

        # Daily loss limit -- stop trading but don't bust
        if acc.daily_pnl <= -DAILY_LOSS_LIMIT:
            if acc.position != 0:
                self._close_position(acc, "daily_loss_cap")
            return

        # Daily profit cap
        max_daily = EVAL_CONFIG.get("max_daily_pnl")
        if max_daily and acc.daily_pnl >= max_daily:
            return

        if acc.trades_this_session >= EVAL_CONFIG["max_trades_per_session"]:
            return

        # Consolidated entry guard check
        guard = acc.entry_blocked(now)

        if guard == EntryGuard.TRADE_ACTIVE:
            # Safety release: if position is flat and stuck for too long,
            # assume WebSocket missed the trade event -- apply conservative cooldown
            if acc.position == 0 and acc.last_exit_time is not None:
                stuck_seconds = (now - acc.last_exit_time).total_seconds()
                if stuck_seconds > STUCK_TRADE_TIMEOUT_SECS:
                    log.warning(
                        f"_trade_active stuck for {stuck_seconds:.0f}s -- "
                        f"force-clearing with {LOSS_COOLDOWN_MINUTES}min cooldown"
                    )
                    acc._trade_active = False
                    acc._pending_exit_cooldown = False
                    acc.apply_loss_cooldown(now)
                    guard = acc.entry_blocked(now)

        if guard == EntryGuard.PENDING_EXIT:
            if acc.last_exit_time is not None:
                pending_secs = (now - acc.last_exit_time).total_seconds()
                if pending_secs > PENDING_EXIT_TIMEOUT_SECS:
                    self._resolve_pending_exit(acc)
                    guard = acc.entry_blocked(now)

        if guard != EntryGuard.CLEAR:
            return

        if not bars:
            return

        prev_day_close = self._get_prev_day_close()

        # Check for new entry
        stop_ticks = pts_to_ticks(EVAL_CONFIG["stop_pts"])
        target_ticks = pts_to_ticks(EVAL_CONFIG["target_pts"])
        qty = EVAL_CONFIG["contracts"]

        if check_vwap_mr_entry(bars, EVAL_CONFIG, prev_day_close=prev_day_close):
            self._open_position(
                acc,
                side=OrderSide.BID,
                contracts=qty,
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
                contracts=qty,
                entry_price=current_price,
                stop_pts=EVAL_CONFIG["stop_pts"],
                target_pts=EVAL_CONFIG["target_pts"],
                stop_ticks=stop_ticks,
                target_ticks=-target_ticks,
            )

    def _trail_stop(self, acc: AccountState, current_price: float):
        """Ratchet the trailing stop and modify the exchange stop order.

        Called on every poll cycle when in a position. Entry places a fixed 5pt stop.
        Once price moves enough, we tighten it by modifying the stop on the exchange.
        """
        trail_pts = EVAL_CONFIG.get("trailing_stop_pts")
        new_stop = None

        if acc.position > 0:
            acc.highest_price_since_entry = max(acc.highest_price_since_entry, current_price)
            if trail_pts is not None:
                trail_stop = round(round((acc.highest_price_since_entry - trail_pts) / TICK_SIZE) * TICK_SIZE, 2)
                if trail_stop > acc.stop_price:
                    new_stop = trail_stop
                    acc.stop_price = trail_stop
        else:
            acc.highest_price_since_entry = min(acc.highest_price_since_entry, current_price)
            if trail_pts is not None:
                trail_stop = round(round((acc.highest_price_since_entry + trail_pts) / TICK_SIZE) * TICK_SIZE, 2)
                if trail_stop < acc.stop_price:
                    new_stop = trail_stop
                    acc.stop_price = trail_stop

        if new_stop is None:
            return

        if acc.stop_order_id is not None:
            self._modify_stop_on_exchange(acc, new_stop)
        else:
            self._find_and_modify_stop(acc, new_stop)

    def _modify_stop_on_exchange(self, acc: AccountState, new_stop: float):
        """Modify a known stop order to a new price."""
        try:
            self.api.modify_order(
                account_id=acc.account_id,
                order_id=acc.stop_order_id,
                stop_price=new_stop,
            )
            log.info(
                f"Trail stop moved on exchange: order {acc.stop_order_id} "
                f"-> {new_stop:.2f} (high={acc.highest_price_since_entry:.2f})"
            )
            self._log_activity(f"Stop moved -> {new_stop:.2f} (best: {acc.highest_price_since_entry:.2f})", "cyan")
        except Exception as e:
            log.error(f"Failed to modify stop order: {e}", exc_info=True)

    def _find_and_modify_stop(self, acc: AccountState, new_stop: float):
        """Find the stop order ID from open orders and modify it."""
        try:
            orders = self.api.get_open_orders(acc.account_id)
            for o in orders:
                if o.get("type") == OrderType.STOP:
                    acc.stop_order_id = int(o.get("id", o.get("orderId", 0)))
                    self.api.modify_order(
                        account_id=acc.account_id,
                        order_id=acc.stop_order_id,
                        stop_price=new_stop,
                    )
                    log.info(f"Found & moved stop order {acc.stop_order_id} -> {new_stop:.2f}")
                    break
        except Exception as e:
            log.error(f"Failed to find/modify stop order: {e}", exc_info=True)

    def _trade_funded(self, acc: AccountState):
        """VWAP Micro MR strategy for funded accounts."""
        now = self._now_et()
        cfg = FUNDED_CONFIG

        # Payout ready -- stop trading until payout is processed
        if acc._payout_notified:
            return

        # Always fetch bars so dashboard has data even outside session
        bars = self._get_today_bars(now)
        if bars:
            acc.bars_5m = bars
            acc.vwap = compute_vwap(bars)

        current_price = getattr(self, '_last_price', None) or (_bar_close(bars[-1]) if bars else None)

        # Trail the stop on every cycle when in a position
        if acc.position != 0 and current_price is not None:
            self._trail_stop(acc, current_price)
            return

        # Session tracking: reset trades + cooldown on session change
        session_name = get_current_session(now)
        if session_name != acc.current_session:
            prev_session = acc.current_session
            if prev_session is not None and acc.session_trades_log:
                self._notify_session_end(acc, prev_session)
            if session_name is not None:
                log.info(
                    f"Account {acc.account_id}: session changed "
                    f"to {session_name} (was {prev_session}), "
                    f"resetting session trade count + cooldown"
                )
            acc.current_session = session_name
            acc.trades_this_session = 0
            acc.cooldown_until = None
            acc.session_trades_log = []

        # Close position outside session window
        if session_name is None:
            if acc.position != 0:
                self._close_position(acc, "session_end")
            return

        # Daily loss limit
        if acc.daily_pnl <= -DAILY_LOSS_LIMIT:
            if acc.position != 0:
                self._close_position(acc, "daily_loss_cap")
            return

        # Daily profit cap
        max_daily = cfg.get("max_daily_pnl")
        if max_daily and acc.daily_pnl >= max_daily:
            return

        if acc.trades_this_session >= cfg["max_trades_per_session"]:
            return

        # Consolidated entry guard check
        guard = acc.entry_blocked(now)

        if guard == EntryGuard.TRADE_ACTIVE:
            if acc.position == 0 and acc.last_exit_time is not None:
                stuck_seconds = (now - acc.last_exit_time).total_seconds()
                if stuck_seconds > STUCK_TRADE_TIMEOUT_SECS:
                    log.warning(
                        f"_trade_active stuck for {stuck_seconds:.0f}s -- "
                        f"force-clearing with {LOSS_COOLDOWN_MINUTES}min cooldown"
                    )
                    acc._trade_active = False
                    acc._pending_exit_cooldown = False
                    acc.apply_loss_cooldown(now)
                    guard = acc.entry_blocked(now)

        if guard == EntryGuard.PENDING_EXIT:
            if acc.last_exit_time is not None:
                pending_secs = (now - acc.last_exit_time).total_seconds()
                if pending_secs > PENDING_EXIT_TIMEOUT_SECS:
                    self._resolve_pending_exit(acc)
                    guard = acc.entry_blocked(now)

        if guard != EntryGuard.CLEAR:
            return

        if not bars:
            return

        prev_day_close = self._get_prev_day_close()

        # Check for new entry
        stop_ticks = pts_to_ticks(cfg["stop_pts"])
        target_ticks = pts_to_ticks(cfg["target_pts"])
        qty = cfg["contracts"]

        if check_vwap_mr_entry(bars, cfg, prev_day_close=prev_day_close):
            self._open_position(
                acc,
                side=OrderSide.BID,
                contracts=qty,
                entry_price=current_price,
                stop_pts=cfg["stop_pts"],
                target_pts=cfg["target_pts"],
                stop_ticks=-stop_ticks,
                target_ticks=target_ticks,
            )
        elif (not cfg.get("long_only", True)
              and check_vwap_mr_short_entry(bars, cfg, prev_day_close=prev_day_close)):
            self._open_position(
                acc,
                side=OrderSide.ASK,
                contracts=qty,
                entry_price=current_price,
                stop_pts=cfg["stop_pts"],
                target_pts=cfg["target_pts"],
                stop_ticks=stop_ticks,
                target_ticks=-target_ticks,
            )

    # -- Order Execution ---------------------------------------------------

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
        trailing_stop: bool = False,
    ):
        """Open a position via market order with optional brackets."""
        try:
            self.api.place_market_order(
                account_id=acc.account_id,
                contract_id=acc.contract_id,
                side=side,
                size=contracts,
                stop_loss_ticks=stop_ticks,
                take_profit_ticks=target_ticks,
                trailing_stop=trailing_stop,
            )

            acc.position = contracts if side == OrderSide.BID else -contracts
            acc._trade_active = True
            acc.entry_price = entry_price
            if side == OrderSide.BID:
                acc.stop_price = entry_price - stop_pts if stop_pts else 0.0
                acc.target_price = entry_price + target_pts if target_pts else 999999.0
            else:
                acc.stop_price = entry_price + stop_pts if stop_pts else 999999.0
                acc.target_price = entry_price - target_pts if target_pts else 0.0
            acc.highest_price_since_entry = entry_price
            acc.trades_today += 1
            acc.trades_this_session += 1

            # Retry up to 3 times to capture stop order ID for trailing stop
            for attempt in range(1, 4):
                try:
                    time.sleep(0.5 * attempt)  # 0.5s, 1s, 1.5s
                    orders = self.api.get_open_orders(acc.account_id)
                    for o in orders:
                        if o.get("type") == OrderType.STOP:
                            acc.stop_order_id = int(o.get("id", o.get("orderId", 0)))
                            log.info(f"Captured stop order ID on attempt {attempt}: {acc.stop_order_id}")
                            break
                    if acc.stop_order_id is not None:
                        break
                except Exception as e:
                    log.warning(f"Stop order ID fetch attempt {attempt}/3 failed: {e}")
            if acc.stop_order_id is None:
                log.warning(f"Could not capture stop order ID after 3 attempts -- trailing will search each cycle")

            trail_str = " [TRAILING]" if trailing_stop else ""
            side_str = "BUY" if side == OrderSide.BID else "SELL"
            log.info(
                f"OPENED: Acct {acc.account_id} | {side_str} {contracts}x NQ @ ~{entry_price:.2f} | "
                f"SL={acc.stop_price:.2f} TP={acc.target_price:.2f}{trail_str}"
            )
            self._log_activity(
                f"ENTRY: {side_str} {contracts}x @ {entry_price:.2f} | SL={acc.stop_price:.2f} TP={acc.target_price:.2f}",
                "green" if side == OrderSide.BID else "red"
            )
        except Exception as e:
            log.error(f"Order failed on acct {acc.account_id}: {e}", exc_info=True)
            self._log_activity(f"Order FAILED: {e}", "red")
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

            log.info(f"CLOSED: Acct {acc.account_id} | {reason} | {side_str} {contracts}x NQ")
            self._log_activity(f"Position closed: {reason} ({side_str} {contracts}x)", "yellow")

            acc.position = 0
            acc.clear_position_state()
            acc.last_exit_time = self._now_et()

            # Check trailing DD bust (after payout, floor = $0)
            drawdown = acc.peak - acc.equity
            max_dd = EVAL_TRAILING_DD if acc.phase == Phase.EVAL else FUNDED_TRAILING_DD
            busted = drawdown >= max_dd
            if acc.phase == Phase.FUNDED and acc.has_taken_payout and acc.equity <= 0:
                busted = True
            if busted:
                acc.active = False
                log.warning(f"BUST: Acct {acc.account_id} | DD ${drawdown:,.2f} >= ${max_dd:,.2f}")

            # Check eval pass
            if (acc.phase == Phase.EVAL
                    and acc.equity >= EVAL_PROFIT_TARGET
                    and acc.trading_days >= 2):
                log.info(f"EVAL PASSED: Acct {acc.account_id} | Equity=${acc.equity:,.2f}")
                self._notify_eval_passed(acc)

            # Check payout ready (funded, Consistency Path) -- notify once
            consistency_ok = (acc.profit_since_payout > 0
                              and acc.max_day_pnl_since_payout
                                  <= acc.profit_since_payout * CONSISTENCY_PCT)
            if (acc.phase == Phase.FUNDED
                    and acc.winning_days >= WINNING_DAYS_REQUIRED
                    and consistency_ok
                    and not acc._payout_notified):
                acc._payout_notified = True
                self._notify_payout_ready(acc)

        except Exception as e:
            log.error(f"Close failed on acct {acc.account_id}: {e}", exc_info=True)
            acc.last_error = str(e)

    # -- Main Loop ---------------------------------------------------------

    def run(self):
        """Main trading loop -- runs 24/7."""
        self.running = True
        log.info("Bot started -- running 24/7")

        self._refresh_daily_bars()
        last_daily_refresh = time.time()
        last_rest_sync = 0.0
        last_dashboard_update = 0.0
        REST_SYNC_INTERVAL = 60

        try:
            with Live(self._dashboard.build(), console=console, refresh_per_second=0.5, vertical_overflow="crop") as live:
                while self.running:
                    try:
                        now = self._now_et()
                        market_open = self._is_market_hours(now)
                        self._check_new_day()

                        if time.time() - last_daily_refresh > 1800:
                            self._refresh_daily_bars()
                            last_daily_refresh = time.time()

                        self.api.ensure_auth()

                        if market_open:
                            for acc in self.accounts.values():
                                if not acc.active or not acc.can_trade:
                                    continue
                                try:
                                    if acc.phase == Phase.EVAL:
                                        self._trade_eval(acc)
                                    else:
                                        self._trade_funded(acc)
                                except Exception as e:
                                    log.error(f"Trade error on acct {acc.account_id}: {e}", exc_info=True)
                                    acc.last_error = str(e)

                        if time.time() - last_rest_sync > REST_SYNC_INTERVAL:
                            for acc in self.accounts.values():
                                self._sync_from_api(acc)
                            last_rest_sync = time.time()

                        dash_interval = self.POLL_INTERVAL if market_open else 30
                        if time.time() - last_dashboard_update >= dash_interval:
                            live.update(self._dashboard.build())
                            last_dashboard_update = time.time()

                        self._save_state()
                        time.sleep(self.POLL_INTERVAL)

                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        log.error(f"Loop error: {e}\n{traceback.format_exc()}")
                        time.sleep(60)

        except KeyboardInterrupt:
            log.info("Shutting down...")
            if self._rt:
                self._rt.stop()
            for acc in self.accounts.values():
                if acc.position != 0:
                    has_brackets = False
                    try:
                        orders = self.api.get_open_orders(acc.account_id)
                        stop_orders = [o for o in orders if o.get("type") in (OrderType.STOP, OrderType.TRAILING_STOP)]
                        has_brackets = len(stop_orders) > 0
                    except Exception:
                        pass
                    if has_brackets:
                        log.info(f"Acct {acc.account_id}: position open with bracket orders on exchange -- leaving as-is")
                    else:
                        log.info(f"Acct {acc.account_id}: no brackets found -- closing position")
                        self._close_position(acc, "shutdown")
            self._save_state()
            log.info("Bot stopped.")


# =========================================================================
# Entry Point
# =========================================================================

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

    global _telegram
    _telegram = _build_telegram_notifier()

    api = ProjectXAPI(api_key, username)
    engine = TradingEngine(api)
    engine.POLL_INTERVAL = args.poll

    tg_str = "ON" if _telegram else "OFF"
    console.print(Panel(
        "[bold]TopstepX Automated Trading Bot[/]\n\n"
        "Strategy: VWAP Micro Mean-Reversion\n"
        "Sessions: Asia (6PM-2AM) -> London (3AM-9:30AM) "
        "-> NY (9:30AM-3:45PM ET)\n"
        f"Trade cap: {EVAL_CONFIG['max_trades_per_session']}"
        f"/session (resets each session), "
        f"cd={LOSS_COOLDOWN_MINUTES}min\n"
        "Funded: Consistency Path payout rules\n"
        f"Telegram: {tg_str}\n\n"
        "Data feed: SignalR WebSocket (real-time) + REST "
        "fallback\n"
        f"Poll interval: {args.poll}s\n"
        "Press Ctrl+C to stop (positions closed safely)",
        title="Starting",
        border_style="green",
    ))

    try:
        engine.setup()
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
