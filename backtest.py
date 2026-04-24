"""
Backtesting utilities for Jarvis_Betting.

Current backtest path:
 - Moneyline backtest against completed games
 - Uses stored projections + latest available odds snapshots
 - NFL-first, but works for all supported leagues with stored outcomes
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import DEFAULT_EDGE_THRESHOLD_PCT, MAX_BACKTEST_BET_SIZE_UNITS
from database import Game, OddsHistory, Projection, get_session
from utils import LOGGER


@dataclass
class BacktestSummary:
    total_bets: int
    wins: int
    losses: int
    units_risked: float
    units_profit: float
    roi: float


class Backtester:
    def run_moneyline_backtest(self, min_edge: float = DEFAULT_EDGE_THRESHOLD_PCT) -> tuple[pd.DataFrame, BacktestSummary]:
        with get_session() as session:
            games_df = pd.read_sql(session.query(Game).statement, session.bind)
            projections_df = pd.read_sql(session.query(Projection).statement, session.bind)
            odds_df = pd.read_sql(session.query(OddsHistory).statement, session.bind)

        if games_df.empty or projections_df.empty or odds_df.empty:
            summary = BacktestSummary(0, 0, 0, 0.0, 0.0, 0.0)
            return pd.DataFrame(), summary

        odds_df["pulled_at"] = pd.to_datetime(odds_df["pulled_at"], utc=True, errors="coerce")
        latest_h2h = odds_df[odds_df["market"] == "h2h"].sort_values("pulled_at").groupby(["game_id", "selection_name"], as_index=False).tail(1)

        merged = projections_df.merge(
            games_df[["id", "home_team", "away_team", "home_score", "away_score", "completed", "league", "sport"]],
            left_on="game_id",
            right_on="id",
            suffixes=("", "_game"),
        )
        merged = merged[merged["completed"] == 1].copy()
        merged = merged[merged["edge_pct"].abs() >= min_edge].copy()

        if merged.empty:
            summary = BacktestSummary(0, 0, 0, 0.0, 0.0, 0.0)
            return merged, summary

        bets: list[dict[str, object]] = []
        for _, row in merged.iterrows():
            selection = row["home_team"] if row["recommended_bet"] == f"Moneyline: {row['home_team']}" else row["away_team"]
            line = latest_h2h[(latest_h2h["game_id"] == row["game_id"]) & (latest_h2h["selection_name"] == selection)]
            if line.empty:
                continue
            price = float(line.iloc[0]["price_american"])
            home_won = row["home_score"] > row["away_score"]
            won = bool(home_won) if selection == row["home_team"] else bool(not home_won)
            stake = MAX_BACKTEST_BET_SIZE_UNITS
            profit = self._profit_from_american_odds(price, stake) if won else -stake
            bets.append(
                {
                    "game_id": row["game_id"],
                    "league": row["league"],
                    "selection": selection,
                    "price_american": price,
                    "won": won,
                    "profit_units": profit,
                    "edge_pct": row["edge_pct"],
                }
            )

        results = pd.DataFrame(bets)
        if results.empty:
            summary = BacktestSummary(0, 0, 0, 0.0, 0.0, 0.0)
            return results, summary

        wins = int(results["won"].sum())
        losses = int((~results["won"]).sum())
        units_risked = float(len(results) * MAX_BACKTEST_BET_SIZE_UNITS)
        units_profit = float(results["profit_units"].sum())
        roi = (units_profit / units_risked) if units_risked else 0.0
        summary = BacktestSummary(
            total_bets=len(results),
            wins=wins,
            losses=losses,
            units_risked=units_risked,
            units_profit=units_profit,
            roi=roi,
        )
        LOGGER.info("Backtest completed: %s", summary)
        return results, summary

    @staticmethod
    def _profit_from_american_odds(price_american: float, stake: float) -> float:
        if price_american > 0:
            return stake * (price_american / 100.0)
        return stake * (100.0 / abs(price_american))


if __name__ == "__main__":
    results, summary = Backtester().run_moneyline_backtest()
    print(summary)
    print(results.head())
    print("Test this now:")
    print("python backtest.py")
