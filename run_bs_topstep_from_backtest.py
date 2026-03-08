import argparse
import pandas as pd
import numpy as np

from lm import (
    monte_carlo_topstep,
    run_topstep_nq,
    run_vwap_micro_momentum_topstep,
    START_EQUITY,
    COMBINE_COST,
)

DISPLAY_COMBINE_FEE = 50.0
DISPLAY_ACTIVATION_FEE = 150.0


def print_equity_sparkline(equity: pd.Series, max_points: int = 120) -> None:
    """Print a simple ASCII sparkline of the equity curve."""
    # Use plain ASCII characters to avoid Windows console encoding issues
    blocks = " .:-=+*#%@"

    values = equity.astype(float).to_numpy()
    if values.size == 0:
        print("No equity data to plot.")
        return

    # Downsample to at most max_points to keep output readable
    n_points = min(max_points, values.size)
    idx = np.linspace(0, values.size - 1, n_points).astype(int)
    sampled = values[idx]

    vmin = sampled.min()
    vmax = sampled.max()
    if vmax == vmin:
        print("Equity sparkline:")
        print(blocks[0] * n_points)
        print(f"Equity constant at: {vmin:.2f}")
        return

    scaled = (sampled - vmin) / (vmax - vmin) * (len(blocks) - 1)
    chars = "".join(blocks[int(round(x))] for x in scaled)

    print("Equity sparkline ({} points):".format(n_points))
    print(chars)
    print(f"Min equity: {vmin:.2f}, Max equity: {vmax:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Topstep backtest runner on OHLC data. "
            "Defaults are tuned to the higher-pass-rate setup."
        )
    )
    parser.add_argument(
        "data_path",
        nargs="?",
        default="nq_bs_backtest.csv",
        help="Path to OHLC data (CSV or parquet) with datetime, open, high, low, close, and ideally volume.",
    )
    parser.add_argument(
        "--strategy",
        choices=["vwap_micro", "vwap_micro_momentum"],
        default="vwap_micro",
        help=(
            "vwap_micro: fixed brackets (default runner mode, tuned for higher pass rate). "
            "vwap_micro_momentum: z-score + ATR."
        ),
    )
    parser.add_argument(
        "--objective",
        choices=["max_net_per_day", "high_pass", "around_80"],
        default="high_pass",
        help=(
            "max_net_per_day: optimize dollars per trading day (faster pass, lower pass rate). "
            "high_pass: optimize pass rate (slower to pass). "
            "around_80: target roughly 80% pass rate."
        ),
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=100,
        help="Number of Topstep Combine Monte Carlo simulations.",
    )
    parser.add_argument(
        "--stop-pts",
        type=float,
        default=5.0,
        help="Stop in points (default runner setup: 5).",
    )
    parser.add_argument(
        "--target-pts",
        type=float,
        default=20.0,
        help="Target in points (default runner setup: 20).",
    )
    parser.add_argument(
        "--adaptive-target",
        action="store_true",
        help="Use 16-point target in lower-volatility conditions and 20-point target in stronger-volatility conditions.",
    )
    parser.add_argument(
        "--low-vol-target-pts",
        type=float,
        default=14.0,
        help="Target used when adaptive-target classifies the setup as lower volatility.",
    )
    parser.add_argument(
        "--high-vol-target-pts",
        type=float,
        default=20.0,
        help="Target used when adaptive-target classifies the setup as higher volatility.",
    )
    parser.add_argument(
        "--contracts",
        type=int,
        default=2,
        help="Contracts per trade. Default is 2 for the updated 76%% pass configuration.",
    )
    parser.add_argument(
        "--max-trades-per-day",
        type=int,
        default=None,
        help="Optional cap on trades per day. If omitted, strategy defaults are used.",
    )
    parser.add_argument(
        "--daily-stop-loss",
        type=float,
        default=None,
        help="Optional tighter daily stop in dollars for the strategy layer.",
    )
    parser.add_argument(
        "--a-plus-only",
        action="store_true",
        help="Only take A+ setups: deeper dip, stronger red candle, and more oversold RSI; defaults to fewer trades/day.",
    )
    parser.add_argument(
        "--hybrid-sizing",
        action="store_true",
        help="Use 1 contract on regular high-prob setups and full --contracts only on A+ setups.",
    )
    parser.add_argument(
        "--partial-scale-out",
        action="store_true",
        help="Take a partial profit before the full target when position size is 2+ contracts.",
    )
    parser.add_argument(
        "--partial-exit-pts",
        type=float,
        default=None,
        help="Partial scale-out target in points. Default is half the full target.",
    )
    parser.add_argument(
        "--breakeven-after-partial",
        action="store_true",
        help="Move stop to breakeven after the partial scale-out fills.",
    )
    parser.add_argument(
        "--rolling-stop-distance-pts",
        type=float,
        default=None,
        help="Optional rolling stop distance in points after a partial scale-out.",
    )
    parser.add_argument(
        "--session-start",
        default="09:30",
        help="Session start HH:MM. Use with --session-tz for RTH (e.g. 09:30 ET).",
    )
    parser.add_argument(
        "--session-end",
        default="15:45",
        help="Session end HH:MM (e.g. 15:45 ET = RTH close).",
    )
    parser.add_argument(
        "--session-tz",
        default="America/New_York",
        help="Timezone for session times (e.g. America/New_York). If data is UTC, bars are converted so 09:30-15:45 = RTH.",
    )
    parser.add_argument(
        "--mean-reversion",
        action="store_true",
        help="For vwap_micro: buy when price below VWAP and declining (dip buy).",
    )
    parser.add_argument(
        "--trend-following",
        action="store_true",
        help="Override the default buy-the-dip mode and use the old trend-following vwap_micro setup.",
    )
    parser.add_argument(
        "--session-end-am",
        action="store_true",
        dest="session_end_am",
        help="Use morning only: session end 11:30 (fewer, potentially higher-quality bars).",
    )
    parser.add_argument(
        "--high-prob-only",
        action="store_true",
        dest="high_prob_only",
        help="Only take highest-probability setups: deeper dip + down bar + RSI (MR), or extended z (momentum); max 3 t/day.",
    )
    parser.add_argument(
        "--all-signals",
        action="store_true",
        help="Override the default high-probability filter and take all qualifying signals.",
    )
    parser.add_argument(
        "--confirmation-entry",
        action="store_true",
        help="Require the next bar to confirm before entering (green confirmation bar).",
    )
    parser.add_argument(
        "--confirmation-reclaim-prior-high",
        action="store_true",
        help="For confirmation-entry, also require the confirmation bar to reclaim the prior bar high.",
    )
    parser.add_argument(
        "--higher-tf-bias",
        action="store_true",
        help="Require higher-timeframe bias alignment (HTF close above EMA and EMA rising).",
    )
    parser.add_argument(
        "--higher-tf-minutes",
        type=int,
        default=15,
        help="Higher timeframe in minutes for the bias filter.",
    )
    parser.add_argument(
        "--day-bias-filter",
        action="store_true",
        help="Only trade days with a positive first window move and a minimum positive session gap.",
    )
    parser.add_argument(
        "--day-bias-first-window-minutes",
        type=int,
        default=15,
        help="Minutes from the session open used to define the early day bias.",
    )
    parser.add_argument(
        "--day-bias-min-first-window-move-pts",
        type=float,
        default=0.0,
        help="Minimum points the first window must gain from session open.",
    )
    parser.add_argument(
        "--day-bias-min-gap-pts",
        type=float,
        default=3.0,
        help="Minimum points the session open must be above the prior day close.",
    )
    args = parser.parse_args()

    # Default to the day-bias filter because it materially improved win rate,
    # pass rate, days-to-pass, and net/day on this dataset.
    if not getattr(args, "day_bias_filter", False):
        args.day_bias_filter = True

    # Load OHLC data (CSV or parquet)
    path = args.data_path
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["datetime"])
    # If datetimes are naive, assume UTC so session_tz (e.g. America/New_York) can convert to RTH
    if "datetime" in df.columns and pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        try:
            if df["datetime"].dt.tz is None:
                df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        except Exception:
            pass
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        if "datetime" not in df.columns and df.index.name:
            df = df.rename(columns={df.index.name: "datetime"})
    df = df.sort_values("datetime")
    required = ["datetime", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Data must have columns: {required}. Missing: {missing}. "
            "Use OHLC data (e.g. from lm.py backtest or raw NQ bars)."
        )
    volume_col = "volume" if "volume" in df.columns else None

    # Session in minutes from midnight
    def parse_time(s: str) -> int:
        parts = s.strip().split(":")
        if len(parts) != 2:
            raise ValueError(f"Expected HH:MM, got {s}")
        return int(parts[0]) * 60 + int(parts[1])

    session_start_min = parse_time(args.session_start)
    session_end_min = parse_time(args.session_end)
    use_morning_only = bool(getattr(args, "session_end_am", False) or args.objective == "high_pass")
    if use_morning_only:
        session_end_min = 11 * 60 + 30
    session_tz = args.session_tz or None

    objective = args.objective

    # Default objective is high_pass so a plain run returns to the original
    # high-pass-rate configuration. around_80 uses a looser filtered
    # mean-reversion variant that tested near 83% pass on this dataset.
    mr = bool(getattr(args, "mean_reversion", False) or not getattr(args, "trend_following", False))
    if objective == "around_80":
        mr = True

    if getattr(args, "high_prob_only", False):
        high_prob = True
    elif getattr(args, "all_signals", False):
        high_prob = False
    else:
        high_prob = objective in {"high_pass", "around_80"}

    if args.strategy == "vwap_micro_momentum":
        # VWAP slope + Micro Momentum z-score, ATR-based stop/target, momentum exit (Topstep spec)
        print("Strategy: VWAP + Micro Momentum (VWAP slope, z-score cross, ATR stops)" + (" [high-prob only, max 3 t/day]" if high_prob else ""))
        bar_df, trades_df, final_pnl = run_vwap_micro_momentum_topstep(
            df,
            timestamp_col="datetime",
            open_col="open",
            high_col="high",
            low_col="low",
            close_col="close",
            volume_col=volume_col,
            contracts=args.contracts,
            session_start_min=session_start_min,
            session_end_min=session_end_min,
            session_tz=session_tz,
            max_trades_per_day=args.max_trades_per_day or 5,
            high_prob_only=high_prob,
        )
    else:
        # Fixed-point VWAP micro runner. Defaults use the filtered mean-reversion setup.
        stop_pts = args.stop_pts
        target_pts = args.target_pts
        min_dip_pts = 2.5
        min_down_bar_pts = 1.0
        rsi_oversold_max = 45.0

        if objective == "around_80":
            stop_pts = 4.0
            target_pts = 16.0
            min_dip_pts = 1.0
            min_down_bar_pts = 0.0
            rsi_oversold_max = None

        max_trades_per_day = args.max_trades_per_day
        if max_trades_per_day is None:
            if args.a_plus_only:
                max_trades_per_day = 2
            elif high_prob:
                max_trades_per_day = 3

        print(
            "Strategy: Topstep VWAP+micro — "
            + str(stop_pts)
            + " pt stop / "
            + str(target_pts)
            + " pt target"
            + (" [mean-reversion]" if mr else "")
            + (" [high-prob only, max 3 t/day]" if high_prob else "")
            + (" [A+ only]" if args.a_plus_only else "")
            + (" [hybrid sizing]" if args.hybrid_sizing else "")
            + (" [partial scale-out]" if args.partial_scale_out else "")
            + (" [breakeven after partial]" if args.breakeven_after_partial else "")
            + (" [adaptive target]" if args.adaptive_target else "")
            + (" [confirmation entry]" if args.confirmation_entry else "")
            + (" [reclaim prior high]" if args.confirmation_reclaim_prior_high else "")
            + (f" [HTF bias {args.higher_tf_minutes}m]" if args.higher_tf_bias else "")
            + (
                f" [day bias {args.day_bias_first_window_minutes}m move>={args.day_bias_min_first_window_move_pts:g} gap>={args.day_bias_min_gap_pts:g}]"
                if args.day_bias_filter else ""
            )
            + (" [around_80 preset]" if objective == "around_80" else "")
            + (f" [RSI<{rsi_oversold_max:g}]" if high_prob and rsi_oversold_max is not None else "")
        )
        bar_df, trades_df, final_pnl = run_topstep_nq(
            df,
            strategy="vwap_micro",
            timestamp_col="datetime",
            open_col="open",
            high_col="high",
            low_col="low",
            close_col="close",
            volume_col=volume_col,
            stop_pts=stop_pts,
            target_pts=target_pts,
            adaptive_target=args.adaptive_target,
            low_vol_target_pts=args.low_vol_target_pts,
            high_vol_target_pts=args.high_vol_target_pts,
            contracts=args.contracts,
            session_start_min=session_start_min,
            session_end_min=session_end_min,
            session_tz=session_tz,
            max_trades_per_day=max_trades_per_day,
            vwap_micro_mean_reversion=mr,
            high_prob_only=high_prob,
            min_dip_pts=min_dip_pts,
            min_down_bar_pts=min_down_bar_pts,
            rsi_oversold_max=rsi_oversold_max,
            a_plus_only=args.a_plus_only,
            hybrid_sizing=args.hybrid_sizing,
            base_contracts=1,
            partial_scale_out=args.partial_scale_out,
            partial_exit_pts=args.partial_exit_pts,
            breakeven_after_partial=args.breakeven_after_partial,
            rolling_stop_distance_pts=args.rolling_stop_distance_pts,
            confirmation_entry=args.confirmation_entry,
            confirmation_reclaim_prior_high=args.confirmation_reclaim_prior_high,
            higher_tf_bias=args.higher_tf_bias,
            higher_tf_minutes=args.higher_tf_minutes,
            day_bias_filter=args.day_bias_filter,
            day_bias_first_window_minutes=args.day_bias_first_window_minutes,
            day_bias_min_first_window_move_pts=args.day_bias_min_first_window_move_pts,
            day_bias_min_gap_pts=args.day_bias_min_gap_pts,
            daily_stop_loss=args.daily_stop_loss,
        )

    # Account equity from 50k starting value (from strategy bar-level output)
    equity = START_EQUITY + bar_df["cum_pnl"].astype(float)
    print("Final equity: {:.2f} (PnL: {:.2f})".format(equity.iloc[-1], equity.iloc[-1] - START_EQUITY))
    print_equity_sparkline(equity)

    if trades_df.empty:
        print("\nNo trades generated. Check session times (--session-tz America/New_York and 09:30-15:45 for RTH) and data timezone.")
        return

    # Diagnostic: why 0%% pass rate often = wrong session or negative expectancy
    n_trades = len(trades_df)
    wins = trades_df["pnl"] > 0
    win_rate = wins.sum() / n_trades if n_trades else 0
    avg_pnl = trades_df["pnl"].mean()
    avg_win = trades_df.loc[wins, "pnl"].mean() if wins.any() else 0
    avg_loss = trades_df.loc[~wins, "pnl"].mean() if (~wins).any() else 0
    n_days = trades_df["trading_day"].nunique()
    print("\nTrade diagnostic (in-sample):")
    print("  Trades: {}, Days: {}, Win rate: {:.1%}, Avg PnL/trade: ${:.2f}".format(n_trades, n_days, win_rate, avg_pnl))
    print("  Avg winner: ${:.2f}, Avg loser: ${:.2f}".format(avg_win, avg_loss))
    if avg_pnl <= 0:
        print("  -> Negative expectancy: try --mean-reversion (buy dip below VWAP) or fix session/strategy.")
    elif n_trades > 500 and not high_prob:
        print("  -> For fewer, higher-probability setups and often much higher pass rate: add --high-prob-only.")

    print("\nRunning Topstep Monte Carlo on {} strategy trades ({} simulations)...\n".format(args.strategy, args.simulations))
    summary, attempts = monte_carlo_topstep(
        trades_df, n_simulations=args.simulations, combine_cost=COMBINE_COST
    )

    print(summary.to_string(index=False))

    # Economic rule: losses from failed/blown accounts are UNREALISED; only the evaluation
    # fee per attempt is the real cost of a fail. Realised PnL = passed accounts only.
    total_accounts = len(attempts)
    passed_accounts = (attempts["Result"] == "PASS").sum()
    passed_profit_total = attempts.loc[attempts["Result"] == "PASS", "Profit"].sum()

    total_fees_all_attempts = total_accounts * COMBINE_COST
    total_fees_passed_only = passed_accounts * COMBINE_COST
    total_days = int(attempts["Days Traded"].sum())

    net_all_attempts = passed_profit_total - total_fees_all_attempts
    net_passed_only = passed_profit_total - total_fees_passed_only
    net_with_activation = (
        passed_profit_total
        - (total_accounts * DISPLAY_COMBINE_FEE)
        - (passed_accounts * DISPLAY_ACTIVATION_FEE)
    )
    net_with_activation_per_day = (
        net_with_activation / total_days if total_days > 0 else 0.0
    )

    print("\nNet across all simulations (failed-account losses unrealised):")
    print(f"  Total accounts: {total_accounts}, Passed: {passed_accounts}")
    print(f"  Total simulated trading days: {total_days}")
    print(f"  Total realised profit (passed accounts only): {passed_profit_total:.2f}")
    print(f"  Net after paying combine cost for all attempts ({COMBINE_COST:.2f} each): {net_all_attempts:.2f}")
    print(f"  Net after paying combine cost only for passed attempts ({COMBINE_COST:.2f} each): {net_passed_only:.2f}")
    print(
        "  Net using $50 combine fee on all attempts + $150 activation on passed accounts: "
        f"{net_with_activation:.2f}"
    )
    print(
        "  Net per simulated trading day using $50 + $150 fee model: "
        f"{net_with_activation_per_day:.2f}"
    )

    print("\nAttempts (first 10):")
    print(attempts.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

