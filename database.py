"""
Database layer for Jarvis_Betting.

This module defines the SQLAlchemy ORM schema and database helpers.
The requested core tables are implemented here:
  - games
  - players
  - injuries
  - odds_history
  - projections
  - line_movement

Primary focus is NFL, but the schema is multi-sport by design so we can expand to
NBA, MLB, and Soccer without changing the core storage model.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from config import DATABASE_PATH


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base declarative class for SQLAlchemy ORM models."""


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
        nullable=False,
    )


class Game(Base, TimestampMixin):
    __tablename__ = "games"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    external_game_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    sport: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    league: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    season: Mapped[str | None] = mapped_column(String(32), nullable=True)
    week: Mapped[str | None] = mapped_column(String(32), nullable=True)
    commence_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(64), default="scheduled", nullable=False, index=True)
    home_team: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    away_team: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    home_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    venue: Mapped[str | None] = mapped_column(String(256), nullable=True)
    source: Mapped[str] = mapped_column(String(64), default="espn", nullable=False)
    raw_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    players: Mapped[list["Player"]] = relationship(back_populates="game", cascade="all, delete-orphan")
    injuries: Mapped[list["Injury"]] = relationship(back_populates="game", cascade="all, delete-orphan")
    odds_history: Mapped[list["OddsHistory"]] = relationship(back_populates="game", cascade="all, delete-orphan")
    projections: Mapped[list["Projection"]] = relationship(back_populates="game", cascade="all, delete-orphan")
    line_movements: Mapped[list["LineMovement"]] = relationship(back_populates="game", cascade="all, delete-orphan")


class Player(Base, TimestampMixin):
    __tablename__ = "players"
    __table_args__ = (
        UniqueConstraint("sport", "league", "external_player_id", name="uq_player_external_per_league"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    external_player_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    sport: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    league: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    game_id: Mapped[int | None] = mapped_column(ForeignKey("games.id"), nullable=True, index=True)
    team: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    full_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    position: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source: Mapped[str] = mapped_column(String(64), default="espn", nullable=False)
    raw_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    game: Mapped[Game | None] = relationship(back_populates="players")
    injuries: Mapped[list["Injury"]] = relationship(back_populates="player", cascade="all, delete-orphan")


class Injury(Base, TimestampMixin):
    __tablename__ = "injuries"
    __table_args__ = (
        Index("ix_injuries_lookup", "sport", "league", "reported_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    external_injury_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    sport: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    league: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    game_id: Mapped[int | None] = mapped_column(ForeignKey("games.id"), nullable=True, index=True)
    player_id: Mapped[int | None] = mapped_column(ForeignKey("players.id"), nullable=True, index=True)
    team: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    player_name: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    injury_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    reported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    source: Mapped[str] = mapped_column(String(64), default="api-sports", nullable=False)
    raw_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    game: Mapped[Game | None] = relationship(back_populates="injuries")
    player: Mapped[Player | None] = relationship(back_populates="injuries")


class OddsHistory(Base, TimestampMixin):
    __tablename__ = "odds_history"
    __table_args__ = (
        Index("ix_odds_lookup", "sport", "league", "market", "pulled_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), nullable=False, index=True)
    sport: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    league: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(64), default="the-odds-api", nullable=False)
    bookmaker: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    selection_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    price_american: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_decimal: Mapped[float | None] = mapped_column(Float, nullable=True)
    point: Mapped[float | None] = mapped_column(Float, nullable=True)
    implied_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    pulled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    raw_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    game: Mapped[Game] = relationship(back_populates="odds_history")


class Projection(Base, TimestampMixin):
    __tablename__ = "projections"
    __table_args__ = (
        Index("ix_projection_lookup", "sport", "league", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), nullable=False, index=True)
    sport: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    league: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    win_prob_home: Mapped[float | None] = mapped_column(Float, nullable=True)
    projected_home_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    projected_away_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    edge_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    recommended_bet: Mapped[str | None] = mapped_column(String(256), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    feature_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    game: Mapped[Game] = relationship(back_populates="projections")


class LineMovement(Base, TimestampMixin):
    __tablename__ = "line_movement"
    __table_args__ = (
        Index("ix_line_move_lookup", "sport", "league", "detected_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), nullable=False, index=True)
    sport: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    league: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    sportsbook: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    selection_name: Mapped[str] = mapped_column(String(128), nullable=False)
    opening_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    opening_point: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_point: Mapped[float | None] = mapped_column(Float, nullable=True)
    movement_abs: Mapped[float | None] = mapped_column(Float, nullable=True)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    source: Mapped[str] = mapped_column(String(64), default="the-odds-api", nullable=False)
    raw_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    game: Mapped[Game] = relationship(back_populates="line_movements")


engine = create_engine(f"sqlite:///{DATABASE_PATH}", future=True, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, class_=Session)


@contextmanager
def get_session() -> Iterator[Session]:
    """Yield a managed SQLAlchemy session with rollback safety."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()
    print(f"Initialized SQLite database at: {DATABASE_PATH}")
    print("Test this now:")
    print("python database.py")
