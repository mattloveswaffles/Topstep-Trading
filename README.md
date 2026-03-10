# Topstep NQ Futures Trading Bot

Automated VWAP Mean-Reversion strategy for passing Topstep 50K Combine evaluations on NQ (Nasdaq-100 E-mini) futures.

## Strategy: VWAP Micro Mean-Reversion

Trades price dips below (longs) and spikes above (shorts) the intraday VWAP on 5-minute bars.

**Entry conditions (long):**
- Price closes below previous bar's close
- Price is >= 1.5 pts below VWAP
- Bearish bar (close < open) with >= 0.5 pt drop
- RSI filter disabled (set to 50/50)

**Short entry** mirrors the long conditions above VWAP.

**Exit:** 5pt initial stop, 2pt trailing stop, 20pt target. Bracket orders managed by the exchange.

**Session:** 9:30 AM - 3:45 PM ET, max 5 trades/day, 2 contracts.

**Backtest results (pessimistic):**
- 1,230 trades over 410 days (2021-2026)
- 37% win rate, $143 avg PnL/trade, 2.14 profit factor
- 90.8% eval pass rate, ~10 calendar days avg

## Setup

1. **Clone and install:**
```bash
git clone <repo-url> && cd <repo>
python -m venv venv
source venv/bin/activate
pip install requests pandas numpy numba rich python-dotenv
```

2. **Configure API key:**
```bash
cp .env.example .env
# Edit .env with your TopstepX API key and username
# Get API key from: TopstepX Dashboard -> Settings -> API Access
```

3. **Run the bot:**
```bash
python topstep_bot.py
```

For persistent server deployment:
```bash
tmux new -s bot
source venv/bin/activate
python topstep_bot.py
# Ctrl+B then D to detach
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

- **Consistency rule:** Must have 5 winning days ($150+ PnL each) before requesting a payout. Days do not need to be consecutive. Counter resets after each payout.
- **Max payout:** $5,000 or 50% of profit balance (whichever is less)
- **Processing:** Payouts requested through TopstepX dashboard
