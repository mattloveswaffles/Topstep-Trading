# Topstep 50K NQ — Strategy for 80%+ Pass (and 5-Day Option)

## Important tradeoff

**The report states: *80%+ pass rate is not achievable in under 5 days.***  
You need ~$1,500/day with a $2,000 trailing drawdown ceiling; variance at that speed is too high.

So you have two realistic choices:

| Goal | Approach | Expected pass rate | Typical days |
|------|----------|---------------------|---------------|
| **80%+ pass** | VWAP + micro momentum, 3 pt stop / 12 pt target, ~30 trades/day | **~82%** | **7–8 days** |
| **Pass in ~5 days** | Broad signal, 5 pt stop / 20 pt target, 15–25 trades/day | **50–72%** | **4–5 days** |

Recommendation: **If you want 80%+ pass rate, use the VWAP + micro strategy and accept 7–8 days.** If you need to pass in ~5 days, use the broad strategy and accept ~50–72% pass rate (Option B $110 upfront is cheaper above ~37% pass).

---

## Combine rules (quick reference)

- **Profit target:** $3,000  
- **Trailing drawdown:** $2,000 from **peak equity**  
- **Daily loss limit:** $1,000  
- **Consistency:** Best single day must be **< 50%** of total profit  
- **Min trading days:** 2  
- **Max contracts:** 5 (NQ)

---

## Strategy 1: 80%+ pass — VWAP + micro momentum (7–8 days)

- **Entry:** `close > VWAP` and `close > close[1]`  
- **Bracket:** **3 pt stop / 12 pt target** (4:1 RR), 1 contract  
- **Session:** Wide, e.g. 9:00–17:00 (more bars → more trades)  
- **Volume:** ~30 trades/day so law of large numbers smooths variance  

**Run:**

```bash
python lm.py your_data.parquet --strategy topstep_vwap_micro --stop-pts 3 --target-pts 12 --session-start 09:00 --session-end 17:00 --simulations 500
```

---

## Strategy 2: Best for ~5 days — Broad signal (50–72% pass)

- **Entry:** `(close > EMA20 and EMA20 > EMA50)` **OR** `RSI14 > 55`  
- **Bracket:** **5 pt stop / 20 pt target** (4:1 RR), 1 contract  
- **Session:** Extended 10:00–16:00  
- **Trades/day:** ~8–15 (optional cap with `--max-trades-per-day 15`)  

**Run:**

```bash
python lm.py your_data.parquet --strategy topstep_broad --stop-pts 5 --target-pts 20 --session-start 10:00 --session-end 16:00 --simulations 500
```

---

## Strategy 3: Balance — Trend + momentum

- **Entry:** (EMA stack trend) **OR** (RSI 55–75 and close > VWAP), with filter `close > VWAP`  
- **Bracket:** 5 pt stop / 20 pt target  
- **Session:** 10:00–16:00, ~5 trades/day  

**Run:**

```bash
python lm.py your_data.parquet --strategy topstep_trend_momentum --stop-pts 5 --target-pts 20 --session-start 10:00 --session-end 16:00 --max-trades-per-day 5 --simulations 500
```

---

## Why fixed-point brackets help

- **5 pt stop** on 1 contract = $100 risk per trade → need **14 consecutive losers** to hit $2,000 trailing DD (very unlikely).  
- **4:1 RR** → only ~**35% win rate** needed to break even; positive expectancy with moderate win rate.  
- **3 pt / 12 pt** → smaller risk per trade, more trades per day, lower variance per day.

---

## Data and volume

- **VWAP** is computed per day from typical price `(H+L+C)/3` and **volume**.  
- Your parquet should have `open`, `high`, `low`, `close`, `datetime`, and ideally **volume** (required for VWAP; if missing, a constant volume is used and VWAP behaves like a running average of typical price).  
- Session times (`--session-start`, `--session-end`) are interpreted in the **same timezone as your `datetime`** (e.g. if data is ET, use 09:30–15:45 for RTH).

---

## Suggested “80% in ~5 days” compromise

If you insist on both **high pass rate** and **~5 days**:

1. Use **topstep_vwap_micro** with **3 pt / 12 pt**.  
2. Widen session to **9:00–17:00** to maximize trades per day.  
3. Run **Monte Carlo** (e.g. `--simulations 500`) on your data; if “Average Days to Pass” is around 6–7, you are close to the report’s 7–8 day / ~82% result.  
4. Accept that reaching **80%+ in strictly ≤5 days** is, per the report, not realistic; the above gets you as close as possible (e.g. 6–7 days at ~82% pass).

---

## Summary table

| Strategy                  | Pass rate (report) | Typical days | Command focus                          |
|---------------------------|--------------------|--------------|----------------------------------------|
| `topstep_vwap_micro`     | ~82%               | 7–8         | `--stop-pts 3 --target-pts 12`          |
| `topstep_broad`          | 50–72%             | 4–5         | `--stop-pts 5 --target-pts 20`         |
| `topstep_trend_momentum` | ~88% (slower)      | 10–12       | `--stop-pts 5 --target-pts 20 --max-trades-per-day 5` |

Use **topstep_vwap_micro** for 80%+ pass; use **topstep_broad** for fastest pass in ~5 days at lower pass rate.

---

## Mean reversion (buy the dip) — often 20–25% pass on backtest data

If the **trend** rule (close > VWAP, close > prev close) has negative expectancy on your data, try **mean reversion**: go long when price is **below** VWAP and **declining** (dip buy), same fixed brackets.

**Run:**

```bash
python run_bs_topstep_from_backtest.py nq_bs_backtest.csv --strategy vwap_micro --mean-reversion --stop-pts 5 --target-pts 20 --simulations 200
```

- **Pass rate:** ~20–25% in testing (vs 0% trend version on same data).
- **Win rate:** ~26–27%; avg PnL/trade positive.
- **Consistency:** Some paths fail the "best day < 50% of total profit" rule; 5/20 pts helps spread profits.

### Highest-probability only: `--high-prob-only`

Take only the best setups and cap at **3 trades/day**: deeper dip (≥2.5 pts below VWAP), clear down bar (open − close ≥ 1 pt), RSI < 38, max 3 t/day.

```bash
python run_bs_topstep_from_backtest.py nq_bs_backtest.csv --strategy vwap_micro --mean-reversion --high-prob-only --stop-pts 5 --target-pts 20 --simulations 200
```

- **Pass rate:** ~**97%**. **Win rate:** ~31–32%; avg PnL/trade ~\$56. Fewer trades (~800–900); ~21 days to pass.

---

## Strategy 4: VWAP + Micro Momentum (Full Topstep spec)

Disciplined intraday strategy matching the VWAP + Micro Momentum prompt: VWAP slope filter, z-score momentum crosses, ATR-based stops/targets, momentum exit.

- **VWAP:** Intraday cumulative (typical price × volume) / volume, reset per day.
- **VWAP slope:** Trade only in trend direction — long when VWAP_t > VWAP_{t-1}, short when VWAP_t < VWAP_{t-1}.
- **Micro Momentum:** SMA_fast(typical price) − SMA_slow(typical price) with fast=3, slow=7; normalized to z-score over last 20 bars.
- **Long entry:** Price above VWAP, VWAP slope up, z crosses from negative to positive (z > 0 and z_prev ≤ 0).
- **Short entry (optional):** Price below VWAP, VWAP slope down, z crosses positive to negative. Set `long_only=True` (default) for long-only.
- **Stop / target:** ATR-based — stop = k_stop × ATR14 (default 0.4), target = k_target × ATR14 (default 0.75).
- **Momentum exit:** Close long when z crosses below zero; close short when z crosses above zero.
- **Cap:** 5–10 trades per day (default 10) for Topstep consistency.

**Run (lm.py):**

```bash
python lm.py your_data.parquet --strategy topstep_vwap_micro_momentum --session-start 09:30 --session-end 15:45 --max-trades-per-day 10 --simulations 500
```

**Run (run_bs_topstep_from_backtest.py):**

```bash
python run_bs_topstep_from_backtest.py nq_bs_backtest.csv --strategy vwap_micro_momentum --simulations 500
```

Default strategy in `run_bs_topstep_from_backtest.py` is now **vwap_micro_momentum**. Use `--strategy vwap_micro` for the fixed 3/12 pt variant.
