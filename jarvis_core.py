"""
Core orchestration engine for Jarvis_Betting.

Responsibilities:
 - initialize the database
 - ingest live and scheduled data
 - build features
 - train/update models
 - persist projections
 - optionally schedule recurring refresh jobs

NFL is prioritized as the first fully wired path.
"""

from __future__ import annotations

from typing import Any

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config import MODEL_RETRAIN_INTERVAL_HOURS, NFL_REFRESH_INTERVAL_MINUTES
from data_ingestion import SportsDataIngestionService
from database import Projection, get_session, init_db
from models import FeatureBuilder, JarvisPredictor
from utils import LOGGER


class JarvisBettingSystem:
    def __init__(self) -> None:
        self.ingestion = SportsDataIngestionService()
        self.feature_builder = FeatureBuilder()
        self.predictor = JarvisPredictor()
        self.scheduler = BlockingScheduler(timezone="America/New_York")

    def refresh_data(self) -> dict[str, int]:
        LOGGER.info("Starting refresh_data cycle")
        return self.ingestion.ingest_all_primary_sports()

    def train_and_project(self) -> int:
        LOGGER.info("Starting train_and_project cycle")
        games_df, odds_df, injuries_df = self.ingestion.build_feature_frames()
        feature_frame = self.feature_builder.build(games_df, odds_df, injuries_df)
        self.predictor.train(feature_frame)
        bundle = self.predictor.predict(feature_frame[feature_frame["completed"] == 0].copy())

        if bundle.frame.empty:
            LOGGER.warning("No pending games available for projection.")
            return 0

        persisted = 0
        with get_session() as session:
            for _, row in bundle.frame.iterrows():
                projection = Projection(
                    game_id=int(row["id"]),
                    sport=row["sport"],
                    league=row["league"],
                    model_name=bundle.model_name,
                    win_prob_home=float(row["win_prob_home"]) if row["win_prob_home"] is not None else None,
                    projected_home_score=float(row["projected_home_score"]) if row["projected_home_score"] is not None else None,
                    projected_away_score=float(row["projected_away_score"]) if row["projected_away_score"] is not None else None,
                    edge_pct=float(row["edge_pct"]) if row["edge_pct"] is not None else None,
                    recommended_bet=row["recommended_bet"],
                    confidence=float(row["confidence"]) if row["confidence"] is not None else None,
                    feature_snapshot={feature: row.get(feature) for feature in bundle.features_used},
                )
                session.add(projection)
                persisted += 1

        self.predictor.save()
        LOGGER.info("Projection cycle complete: %s rows persisted", persisted)
        return persisted

    def run_once(self) -> dict[str, Any]:
        init_db()
        refresh_results = self.refresh_data()
        projection_count = self.train_and_project()
        return {"refresh_results": refresh_results, "projection_count": projection_count}

    def schedule(self) -> None:
        """
        Start a blocking scheduler for autonomous updates.
        This is suitable for a dedicated process or server runtime.
        """
        init_db()
        self.scheduler.add_job(self.refresh_data, IntervalTrigger(minutes=NFL_REFRESH_INTERVAL_MINUTES), id="refresh_data", replace_existing=True)
        self.scheduler.add_job(self.train_and_project, IntervalTrigger(hours=MODEL_RETRAIN_INTERVAL_HOURS), id="train_and_project", replace_existing=True)
        LOGGER.info("Scheduler started with refresh=%sm retrain=%sh", NFL_REFRESH_INTERVAL_MINUTES, MODEL_RETRAIN_INTERVAL_HOURS)
        self.scheduler.start()


if __name__ == "__main__":
    result = JarvisBettingSystem().run_once()
    print(result)
    print("Test this now:")
    print("python jarvis_core.py")
