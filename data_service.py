"""Normalized read-only data service for the Jarvis Betting dashboard.

The service only exposes data already stored from configured providers. It never
fabricates odds, stats, injuries, weather, hit rates, or projections. Missing
fields are normalized to explicit empty values so the dashboard can render clean
"Unavailable" / "No data" states on first launch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from cachetools import TTLCache, cached

from data_ingestion import (
    GAME_COLUMNS,
    INJURY_COLUMNS,
    LINE_MOVE_COLUMNS,
    ODDS_COLUMNS,
    PROJECTION_COLUMNS,
    SOCIAL_TREND_COLUMNS,
    empty_games_frame,
    empty_injuries_frame,
    empty_line_moves_frame,
    empty_odds_frame,
    empty_projections_frame,
    empty_social_trends_frame,
    ensure_frame_schema,
)
from database import Game, Injury, LineMovement, OddsHistory, Player, Projection, get_session, init_db
from models import PROP_COLUMNS, build_player_props_frame, empty_props_frame, prop_type_for_market, strong_props
from utils import convert_series_to_est, display_bookmaker, format_est, is_sharp_book, safe_float, safe_json_loads

LEAGUE_META: dict[str, dict[str, str]] = {
    "nba": {"label": "NBA", "sport": "basketball", "odds_key": "basketball_nba"},
    "nhl": {"label": "NHL", "sport": "hockey", "odds_key": "icehockey_nhl"},
    "mlb": {"label": "MLB", "sport": "baseball", "odds_key": "baseball_mlb"},
    "nfl": {"label": "NFL", "sport": "football", "odds_key": "americanfootball_nfl"},
}

BOOK_BADGES: dict[str, str] = {
    "fanduel": "FD",
    "draftkings": "DK",
    "betmgm": "MGM",
    "caesars": "CZR",
    "espn bet": "ESPN",
    "pinnacle": "PIN",
    "circa": "CIRCA",
    "betrivers": "BR",
    "bet365": "365",
    "fanatics": "FAN",
}

TEAM_ABBR: dict[str, str] = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CWS", "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM", "New York Yankees": "NYY", "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD", "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
}

GAME_EXTRA_COLUMNS = [
    *GAME_COLUMNS,
    "commence_time_est",
    "start_est",
    "matchup",
    "home_abbr",
    "away_abbr",
    "home_logo",
    "away_logo",
    "weather_context",
]

PLAYER_COLUMNS = ["id", "external_player_id", "sport", "league", "game_id", "team", "full_name", "position", "status", "source", "raw_json"]

PROP_VIEW_COLUMNS = [
    "favorite", "pf_score", "team", "position", "player_name", "headshot", "prop_type", "market_display",
    "prop_side", "point", "projected_line", "bookmaker", "book_badge", "price_american", "hit_rate_l10",
    "hit_rate_l5", "streak", "season_matchup", "previous_season_hit_rate", "current_season_hit_rate",
    "display_edge_pct", "confidence", "realism_score", "why", "game_id", "league", "matchup",
]

_stats_cache: TTLCache[str, pd.DataFrame] = TTLCache(maxsize=8, ttl=300)


def _safe_read(model: Any, columns: list[str], empty_factory: Any) -> pd.DataFrame:
    try:
        with get_session() as session:
            frame = pd.read_sql(session.query(model).statement, session.bind)
        return ensure_frame_schema(frame, columns)
    except Exception:
        return empty_factory()


def _extract_logo(game_raw: Any, side: str) -> str | None:
    raw = safe_json_loads(game_raw)
    for competition in raw.get("competitions") or []:
        for competitor in competition.get("competitors") or []:
            if competitor.get("homeAway") == side:
                logos = (competitor.get("team") or {}).get("logos") or []
                if logos:
                    return logos[0].get("href")
    return None


def _extract_weather(game_raw: Any) -> dict[str, Any]:
    raw = safe_json_loads(game_raw)
    context = raw.get("weather_context") or raw.get("weather") or {}
    return context if isinstance(context, dict) else {}


def team_abbr(name: Any) -> str:
    text = str(name or "TBD")
    if text in TEAM_ABBR:
        return TEAM_ABBR[text]
    parts = [part[0] for part in text.replace("-", " ").split() if part]
    return "".join(parts[:3]).upper() or "TBD"


def book_badge(bookmaker: Any) -> str:
    key = str(bookmaker or "").lower().replace("_", " ")
    for needle, badge in BOOK_BADGES.items():
        if needle in key:
            return badge
    clean = display_bookmaker(bookmaker)
    return "".join(part[:1] for part in clean.split()[:3]).upper() or "BOOK"


def normalize_games(games: pd.DataFrame) -> pd.DataFrame:
    games = ensure_frame_schema(games, GAME_COLUMNS)
    if games.empty:
        return pd.DataFrame(columns=GAME_EXTRA_COLUMNS)
    frame = games.copy()
    frame["commence_time_est"] = convert_series_to_est(frame["commence_time"])
    frame["start_est"] = frame["commence_time_est"].apply(lambda value: format_est(value, "%a %b %d, %I:%M %p %Z"))
    frame["matchup"] = frame["away_team"].fillna("TBD") + " @ " + frame["home_team"].fillna("TBD")
    frame["home_abbr"] = frame["home_team"].apply(team_abbr)
    frame["away_abbr"] = frame["away_team"].apply(team_abbr)
    frame["home_logo"] = frame["raw_json"].apply(lambda raw: _extract_logo(raw, "home"))
    frame["away_logo"] = frame["raw_json"].apply(lambda raw: _extract_logo(raw, "away"))
    frame["weather_context"] = frame["raw_json"].apply(_extract_weather)
    return ensure_frame_schema(frame, GAME_EXTRA_COLUMNS)


def normalize_odds(odds: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    odds = ensure_frame_schema(odds, ODDS_COLUMNS)
    if odds.empty:
        return empty_odds_frame()
    frame = odds.copy()
    frame["pulled_at_est"] = convert_series_to_est(frame["pulled_at"])
    frame["bookmaker_display"] = frame["bookmaker"].apply(display_bookmaker)
    frame["book_badge"] = frame["bookmaker"].apply(book_badge)
    frame["sharp_book"] = frame["bookmaker"].apply(is_sharp_book)
    frame["prop_type"] = frame["market"].apply(prop_type_for_market)
    games_small = ensure_frame_schema(games, ["id", "matchup", "commence_time_est", "home_team", "away_team"])
    if not games_small.empty:
        frame = frame.merge(games_small, left_on="game_id", right_on="id", how="left", suffixes=("", "_game"))
    return frame


def normalize_players(players: pd.DataFrame) -> pd.DataFrame:
    players = ensure_frame_schema(players, PLAYER_COLUMNS)
    if players.empty:
        return players
    frame = players.copy()
    frame["headshot"] = frame["raw_json"].apply(lambda raw: (safe_json_loads(raw).get("headshot") or {}).get("href") if isinstance(safe_json_loads(raw).get("headshot"), dict) else None)
    return frame


def normalize_props(props: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    props = ensure_frame_schema(props, PROP_COLUMNS)
    if props.empty:
        return pd.DataFrame(columns=PROP_VIEW_COLUMNS)
    frame = props.copy()
    frame["favorite"] = "☆"
    frame["bookmaker"] = frame["bookmaker"].apply(display_bookmaker)
    frame["book_badge"] = frame["bookmaker"].apply(book_badge)
    frame["team"] = frame.get("team", pd.Series(["Unavailable"] * len(frame))).fillna("Unavailable")
    frame["position"] = frame.get("position", pd.Series(["Unavailable"] * len(frame))).fillna("Unavailable")
    frame["headshot"] = frame.get("headshot", pd.Series([None] * len(frame)))
    frame["streak"] = frame.apply(lambda row: _streak_label(row.get("hit_rate_l5"), row.get("hit_rate_l10")), axis=1)
    frame["season_matchup"] = "Unavailable"
    frame["previous_season_hit_rate"] = pd.NA
    frame["current_season_hit_rate"] = frame["hit_rate_l10"]
    if not players.empty and "full_name" in players:
        player_map = players.drop_duplicates("full_name").set_index("full_name")
        if "team" in player_map:
            frame["team"] = frame["team"].where(frame["team"].ne("Unavailable"), frame["player_name"].map(player_map["team"]))
        if "position" in player_map:
            frame["position"] = frame["position"].where(frame["position"].ne("Unavailable"), frame["player_name"].map(player_map["position"]))
        if "headshot" in player_map:
            frame["headshot"] = frame["headshot"].fillna(frame["player_name"].map(player_map["headshot"]))
    return ensure_frame_schema(frame, PROP_VIEW_COLUMNS)


def _streak_label(l5: Any, l10: Any) -> str:
    l5v = safe_float(l5)
    l10v = safe_float(l10)
    if l5v is None and l10v is None:
        return "No data"
    best = max(value for value in [l5v, l10v] if value is not None)
    if best >= 0.70:
        return "Hot"
    if best <= 0.35:
        return "Cold"
    return "Neutral"


@dataclass(frozen=True)
class DashboardData:
    games: pd.DataFrame
    odds: pd.DataFrame
    injuries: pd.DataFrame
    moves: pd.DataFrame
    projections: pd.DataFrame
    players: pd.DataFrame
    props: pd.DataFrame
    strong_props: pd.DataFrame


@cached(_stats_cache)
def load_dashboard_data() -> DashboardData:
    init_db()
    games_raw = _safe_read(Game, GAME_COLUMNS, empty_games_frame)
    odds_raw = _safe_read(OddsHistory, ODDS_COLUMNS, empty_odds_frame)
    injuries = ensure_frame_schema(_safe_read(Injury, INJURY_COLUMNS, empty_injuries_frame), INJURY_COLUMNS)
    moves = ensure_frame_schema(_safe_read(LineMovement, LINE_MOVE_COLUMNS, empty_line_moves_frame), LINE_MOVE_COLUMNS)
    projections = ensure_frame_schema(_safe_read(Projection, PROJECTION_COLUMNS, empty_projections_frame), PROJECTION_COLUMNS)
    players = normalize_players(_safe_read(Player, PLAYER_COLUMNS, lambda: pd.DataFrame(columns=PLAYER_COLUMNS)))

    # Keep raw model input separate from UI-normalized frames. The prop model
    # performs its own game merge and can fail if odds already contain UI merge
    # columns such as id_game/matchup/start_est.
    props_raw = build_player_props_frame(odds_raw, games_raw, injuries_df=injuries, social_df=empty_social_trends_frame())

    games = normalize_games(games_raw)
    odds = normalize_odds(odds_raw, games)
    props = normalize_props(props_raw, players)
    strong = strong_props(props_raw) if not props_raw.empty else empty_props_frame()
    strong = normalize_props(strong, players)
    return DashboardData(games, odds, injuries, moves, projections, players, props, strong)


def clear_service_cache() -> None:
    _stats_cache.clear()


def league_games(data: DashboardData, league: str, selected_date: Any | None = None) -> pd.DataFrame:
    frame = data.games[data.games["league"].astype(str).eq(league)].copy() if not data.games.empty else pd.DataFrame(columns=GAME_EXTRA_COLUMNS)
    if selected_date is not None and not frame.empty:
        selected = pd.Timestamp(selected_date).date()
        same_day = frame[frame["commence_time_est"].dt.date.eq(selected)].copy()
        if not same_day.empty:
            return same_day.sort_values("commence_time_est")
    return frame.sort_values("commence_time_est") if not frame.empty else frame


def game_odds(data: DashboardData, game_id: Any, markets: list[str] | None = None) -> pd.DataFrame:
    markets = markets or ["h2h", "spreads", "totals"]
    frame = data.odds[(data.odds["game_id"] == game_id) & (data.odds["market"].astype(str).isin(markets))].copy() if not data.odds.empty else empty_odds_frame()
    if frame.empty:
        return frame
    frame["best_price"] = frame.groupby(["market", "selection_name"], dropna=False)["price_american"].transform("max") == frame["price_american"]
    return frame.sort_values(["market", "selection_name", "best_price"], ascending=[True, True, False])


def league_props(data: DashboardData, league: str) -> pd.DataFrame:
    return data.props[data.props["league"].astype(str).eq(league)].copy() if not data.props.empty else pd.DataFrame(columns=PROP_VIEW_COLUMNS)


def league_strong_props(data: DashboardData, league: str) -> pd.DataFrame:
    return data.strong_props[data.strong_props["league"].astype(str).eq(league)].copy() if not data.strong_props.empty else pd.DataFrame(columns=PROP_VIEW_COLUMNS)


def game_props(data: DashboardData, game_id: Any) -> pd.DataFrame:
    return data.props[data.props["game_id"].eq(game_id)].copy() if not data.props.empty else pd.DataFrame(columns=PROP_VIEW_COLUMNS)


def game_weather(game: pd.Series) -> dict[str, Any]:
    context = game.get("weather_context")
    return context if isinstance(context, dict) else {}


def weather_risk(weather: dict[str, Any]) -> str:
    if not weather or not weather.get("weather_available", False):
        return "Unavailable"
    precip = safe_float(weather.get("precipitation_probability")) or 0.0
    forecast = str(weather.get("short_forecast") or "").lower()
    if "thunder" in forecast or precip >= 70:
        return "Postponement Likely"
    if precip >= 45:
        return "Delay Likely"
    if precip >= 25 or "rain" in forecast or "showers" in forecast:
        return "Chance for Delay"
    return "Clear"


def projection_cards(data: DashboardData, league: str) -> pd.DataFrame:
    projections = data.projections[data.projections["league"].astype(str).eq(league)].copy() if not data.projections.empty else empty_projections_frame()
    if projections.empty:
        return projections
    games = data.games[["id", "away_team", "home_team", "matchup", "start_est"]].copy() if not data.games.empty else pd.DataFrame(columns=["id", "away_team", "home_team", "matchup", "start_est"])
    return projections.merge(games, left_on="game_id", right_on="id", how="left", suffixes=("", "_game"))


def injury_splits_proxy(data: DashboardData, league: str) -> pd.DataFrame:
    injuries = data.injuries[data.injuries["league"].astype(str).eq(league)].copy() if not data.injuries.empty else empty_injuries_frame()
    if injuries.empty:
        return pd.DataFrame(columns=["Player", "With/Without injured player", "Games", "Line", "USG%", "MIN", "PTS", "REB", "AST", "PRA", "3PM"])
    frame = injuries.rename(columns={"player_name": "Player"}).copy()
    frame["With/Without injured player"] = frame["status"].fillna("Unavailable")
    for column in ["Games", "Line", "USG%", "MIN", "PTS", "REB", "AST", "PRA", "3PM"]:
        frame[column] = "Unavailable"
    return frame[["Player", "With/Without injured player", "Games", "Line", "USG%", "MIN", "PTS", "REB", "AST", "PRA", "3PM"]]
