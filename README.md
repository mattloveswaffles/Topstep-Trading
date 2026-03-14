# Topstep NQ Futures Trading Bot

Automated VWAP Mean-Reversion strategy for passing Topstep 50K Combine evaluations and extracting payouts on NQ (Nasdaq-100 E-mini) futures.

## Strategy: VWAP Micro Mean-Reversion

Trades price dips below (longs) and spikes above (shorts) the intraday VWAP on 5-minute bars.

**Entry conditions (long):**
- Price closes below previous bar's close
- Price is >= 1.5 pts below VWAP
- Bearish bar (close < open) with >= 0.5 pt drop
- RSI filter disabled (set to 50/50)

**Short entry** mirrors the long conditions above VWAP.

**Exit:** 5pt initial stop, 2pt trailing stop, 20pt target. Bracket orders managed by the exchange.

**Risk Management:**
- 180-minute cooldown after any losing trade (resets at session open)
- $1,600 daily profit cap (stops trading for the day once reached)
- $1,000 daily loss limit
- Coast mode on funded accounts: reduces to 1 contract + $200 daily cap after $1,500 profit since last payout

**Sessions (all traded, per-session trade reset):**
| Session | Hours (ET) | Trades |
|---------|-----------|--------|
| Asia    | 6:00 PM - 2:00 AM | 5/session |
| London  | 3:00 AM - 9:30 AM | 5/session |
| NY      | 9:30 AM - 3:45 PM | 5/session |

Trade count and cooldown reset at the start of each session. 2 contracts per trade (1 in coast mode).

## Architecture

- **topstep_bot.py** — Live trading bot with Rich TUI dashboard
- **backtest_suite.py** — Strategy backtesting + Monte Carlo simulation
- **engine/backtest.py** — Core backtest engine
- **engine/alpha.py** — Signal generation from historical data
- **test_pipeline.py** — Practice account integration test

**Data feed:** SignalR WebSocket (real-time) with REST API fallback. 5-second poll interval.

**Notifications:** Optional Telegram integration for session summaries, eval pass alerts, and payout-ready alerts. Runs on a background thread to never block trading.

## Setup

1. **Install dependencies:**
```bash
pip install requests pandas numpy numba rich python-dotenv signalrcore
```

2. **Configure:**
```bash
cp .env.example .env
```
```
API_KEY=your_topstepx_api_key
TOPSTEPX_USERNAME=your_email
TELEGRAM_BOT_TOKEN=your_bot_token    # optional
TELEGRAM_CHAT_ID=your_chat_id        # optional, auto-discovered if omitted
```
Get API key from: TopstepX Dashboard -> Settings -> API Access

3. **Run the bot:**
```bash
python topstep_bot.py
```

For persistent server deployment:
```bash
tmux new -s bot
python topstep_bot.py
# Ctrl+B then D to detach
```

4. **Run backtests:**
```bash
python backtest_suite.py --combo --eval-strategy vwap_mr_all_sessions
python backtest_suite.py --coast --strategy vwap_mr_all_sessions
```

## Topstep Rules

### Eval (50K Combine)

- **Profit target:** $3,000
- **Trailing drawdown:** $2,000
- **Daily loss limit:** $1,000
- **Min trading days:** 2

### Funded Account (Express Funded)

- **Trailing drawdown:** $2,000
- **Daily loss limit:** $1,000
- **Activation fee:** $150 (one-time)

### Payout Rules

- **Consistency rule:** 5 winning days ($150+ PnL each) before requesting payout. Counter resets after each payout.
- **Max payout:** $5,000 or 50% of profit balance (whichever is less)
- **Auto-pause:** Bot stops trading on accounts eligible for payout until processed
- **Processing:** Payouts requested through TopstepX dashboard

## Telegram Notifications

When configured, the bot sends:
- **Session-end summary:** PnL per account, each trade with timestamp and result, phase (EVAL/FUNDED)
- **Eval passed:** Alert when an account clears the $3K profit target
- **Payout ready:** Alert when a funded account has 5 winning days + positive profit, with estimated payout amount
