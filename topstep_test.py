import pandas as pd
import numpy as np

from lm import monte_carlo_topstep


def main() -> None:
    rng = np.random.default_rng(0)

    n_days = 20
    rows = []
    base_time = pd.Timestamp("2024-01-01")

    for d in range(n_days):
        trading_day = f"day_{d + 1}"
        day_times = (
            base_time + pd.to_timedelta(d, "D") + pd.to_timedelta(np.arange(5), "m")
        )
        pnls = rng.normal(loc=50.0, scale=150.0, size=5)
        cum = 0.0
        for t, pnl in zip(day_times, pnls):
            cum += pnl
            rows.append(
                {
                    "timestamp": t,
                    "instrument": "NQ",
                    "entry_price": 100.0,
                    "exit_price": 100.0,
                    "contracts": 1,
                    "pnl": float(pnl),
                    "cumulative_pnl": cum,
                    "trading_day": trading_day,
                }
            )

    trades = pd.DataFrame(rows)

    summary, attempts = monte_carlo_topstep(
        trades, n_simulations=100, combine_cost=49.0
    )

    print(summary.to_string(index=False))
    print("\nAttempts head:")
    print(attempts.head().to_string(index=False))


if __name__ == "__main__":
    main()

