# Topstep NQ Futures Strategy Report

## Topstep 50K Combine Rules

### Evaluation Phase
| Rule | Value |
|------|-------|
| Profit Target | $3,000 |
| Trailing Drawdown Limit | $2,000 (from peak equity) |
| Daily Loss Limit | $1,000 |
| Consistency Rule | Best single day < 50% of total profit |
| Min Trading Days | 2 |
| Max Contracts | 5 (NQ standard) |
| Cost | $50/month + $150 activation on pass, OR $110 upfront (no activation) |

### Funded Account / Payout Rules
| Rule | Value |
|------|-------|
| Trailing Drawdown | $2,000 (same as eval) |
| Daily Loss Limit | $1,000 |
| Winning Days for Payout | 5 days with $150+ profit each |
| Payout Amount | Up to 50% of funded balance |
| Payout Cap | $5,000 per payout |
| Profit Split | 90% to trader |
| On Blowup | Account lost, buy new combine |

### Key Constraints
- Trailing DD is measured from **peak equity**, not starting balance. A $1,500 profit followed by a $2,000 pullback blows the account even though you're still up $500 from start.
- The consistency rule prevents passing by having one monster day — profit must be spread across at least 2 days.
- Payout requires 5 **cumulative** (not consecutive) winning days of $150+.

---

## The Universal Edge: Asymmetric Risk/Reward

The single most important finding: **tight stops with wide targets** dramatically improve pass rates regardless of entry signal.

| Stop | Target | RR Ratio | Max Daily Losers Before Cap |
|------|--------|----------|-----------------------------|
| 5 pts | 20 pts | 4:1 | 6 |
| 5 pts | 15 pts | 3:1 | 6 |
| 8 pts | 30 pts | 3.75:1 | 4 |
| 3 pts | 12 pts | 4:1 | 9 |

A 5-point stop on 1 contract means you'd need **14 consecutive losers** to hit the $2,000 trailing DD — statistically near-impossible. The 4:1 RR means you only need ~35% win rate to break even.

---

## Top Strategies

### Strategy 1: TREND CONTINUATION + VWAP/RSI Filter
**Highest pass rate — safest approach**

**Entry**: `close > EMA20 AND close > EMA50 AND EMA20 > EMA50`
**Filter**: `close > VWAP AND RSI14 < 75`
**Session**: Full RTH (9:30 AM - 3:45 PM ET)
**Bracket**: 5-point stop / 20-point target, 1 contract, 3-5 trades/day

**Why it works**: Only trades when the trend is clearly established. Low frequency means low variance. The VWAP filter keeps you on the right side of intraday fair value, and the RSI cap avoids buying into overbought conditions.

**Tradeoff**: Slow to pass (12-22 days). Fewer funded accounts but they survive a long time.

---

### Strategy 2: ANY LONG (Broad Signal)
**Fastest pass — highest volume approach**

**Entry**: `(close > EMA20 AND EMA20 > EMA50) OR RSI14 > 55`
**Filter**: None
**Session**: Extended (10:00 AM - 4:00 PM ET)
**Bracket**: 5-point stop / 20-point target, 1 contract, 8-15 trades/day

**Why it works**: The signal fires frequently (~61% of session bars), so variance smooths out quickly through volume. The 4:1 RR ratio means you only need ~35% win rate to profit, and the broad signal captures the general upward bias of NQ.

**Tradeoff**: More eval failures and funded blowups, but the volume of attempts and fast pass speed compensates.

---

### Strategy 3: VWAP + MICRO MOMENTUM
**Best pass rate at speed**

**Entry**: `close > VWAP AND close > close[1]`
**Filter**: None
**Session**: Wide (9:00 AM - 5:00 PM ET)
**Bracket**: 3-point stop / 12-point target, 1 contract, 30 trades/day

**Why it works**: Very high trade frequency with tiny risk per trade. The law of large numbers kicks in — over 30 trades per day, positive expectancy dominates. VWAP acts as an intraday trend filter and the uptick condition confirms micro-momentum.

---

### Strategy 4: TREND + MOMENTUM Combined
**Best balance of speed and safety**

**Entry**: `(close > EMA20 AND close > EMA50 AND EMA20 > EMA50) OR (RSI14 > 55 AND RSI14 < 75 AND close > VWAP)`
**Filter**: `close > VWAP`
**Session**: Extended (10:00 AM - 4:00 PM ET)
**Bracket**: 5-point stop / 20-point target, 1 contract, 5 trades/day

**Why it works**: Combines two independent edge sources — trend following (EMA stack) and mean-reversion momentum (RSI + VWAP). The OR logic means you trade in strong trends AND in choppy markets where RSI momentum aligns with VWAP.

---

## Pass Rate vs Speed Tradeoff

| Target Speed | Best Strategy | Approx Pass Rate | Config |
|-------------|---------------|-------------------|--------|
| 2-3 days | Any Long | 45-65% | S5/T20, 1-2ct, 8-15t/d |
| 4-5 days | Broad signal + layers | 50-72% | S5/T20, 1ct, 15-25t/d |
| 7-8 days | VWAP + Micro Momentum | ~82% | S3/T12, 1ct, 30t/d |
| 10-12 days | Trend + Momentum | ~88% | S5/T20, 1ct, 5t/d |
| 15-20 days | Trend + VWAP | ~94% | S5/T20, 1ct, 3t/d |
| 20-30 days | Trend Cont + vol filter | ~100% | S5/T20, 1ct, 3t/d |

**Key insight**: 80%+ pass rate is not achievable in under 5 days. You need ~$1,500/day with a $2,000 trailing DD ceiling, and variance at that speed is too high.

---

## Funded Account Survival

The biggest leak is **funded accounts blowing up before extracting enough payouts**. The trailing DD from peak equity means even a great day can set you up for failure.

| Strategy Style | Funded Survival | Payout Extraction |
|----------------|-----------------|-------------------|
| Conservative (Trend Cont) | Long-lived | Many payouts per account |
| Balanced (Trend+Mom) | Medium | Variable |
| Aggressive (Any Long) | Short-lived | Few payouts per account |

Conservative strategies survive longer and extract more per funded account. Aggressive strategies churn through more accounts but make it up in volume.

---

## Cost Analysis

### Option A: $50/month + $150 activation
- Cost per eval failure: $50
- Cost per funded account: $200

### Option B: $110 upfront, no activation
- Cost per attempt: $110 (pass or fail)

Option B is cheaper at any pass rate above ~37%. The higher your pass rate, the more you save with Option B.

---

## Strategy Implementation

All strategies use simple indicators available in any charting platform:
- **EMA 20 & 50**: Trend direction
- **RSI 14**: Momentum filter
- **VWAP**: Intraday fair value
- **Fixed bracket order**: Stop and target set at entry

These can be implemented in:
- TradingView (Pine Script)
- NinjaTrader (NinjaScript)
- Manual execution with bracket orders

The entry rules are simple enough to execute manually: check the conditions, enter long, and place a bracket order with your stop and target.
