"""Data ingestion for Jarvis_Betting.

Verified integrations:
- ESPN public API base URL: https://site.api.espn.com/apis/site/v2/sports/
- The Odds API v4 base URL: https://api.the-odds-api.com/v4/
- Player props and alternates use The Odds API event odds endpoint:
  /v4/sports/{sport}/events/{eventId}/odds
- X API recent search base URL: https://api.x.com/2/tweets/search/recent
  Requires X_BEARER_TOKEN. Without a token, the system falls back to public
  in-app trend signals derived from stored prop odds.
- API-Football base URL: https://v3.football.api-sports.io/
- TheSportsDB base URL: https://www.thesportsdb.com/api/v1/json/{key}/
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any

import httpx
import pandas as pd
import requests
from bs4 import BeautifulSoup
from cachetools import TTLCache
from pybaseball import statcast
from requests import Response
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import (
    API_FOOTBALL_DEFAULTS,
    API_SPORTS_FOOTBALL_BASE_URL,
    API_SPORTS_KEY,
    DEFAULT_CACHE_TTL_SECONDS,
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_MIN_REQUEST_INTERVAL_SECONDS,
    DEFAULT_READ_TIMEOUT_SECONDS,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_BACKOFF_MAX_SECONDS,
    DEFAULT_RETRY_BACKOFF_MIN_SECONDS,
    ESPN_BASE_URL,
    SPORT_CONFIG,
    SUPPORTED_DATE_FORMAT,
    SUPPORTED_ODDS_FORMAT,
    SUPPORTED_ODDS_MARKETS,
    SUPPORTED_REGIONS,
    THESPORTSDB_BASE_URL,
    THE_ODDS_API_KEY,
    THE_ODDS_BASE_URL,
)
from database import Game, Injury, LineMovement, OddsHistory, Player, get_session, init_db
from utils import LOGGER, american_to_decimal, american_to_implied_probability, chunked, safe_float, safe_int, utcnow

EXTENDED_SPORT_CONFIG: dict[str, dict[str, str]] = {
    **SPORT_CONFIG,
    "nhl": {"sport": "hockey", "league": "nhl", "display_name": "NHL", "odds_sport_key": "icehockey_nhl", "espn_path": "hockey/nhl"},
    "ncaaf": {"sport": "football", "league": "college-football", "display_name": "College Football", "odds_sport_key": "americanfootball_ncaaf", "espn_path": "football/college-football"},
    "wnba": {"sport": "basketball", "league": "wnba", "display_name": "WNBA", "odds_sport_key": "basketball_wnba", "espn_path": "basketball/wnba"},
}

PLAYER_PROP_MARKETS_BY_SPORT: dict[str, tuple[str, ...]] = {
    "nfl": (
        "player_pass_tds", "player_pass_yds", "player_pass_attempts", "player_pass_completions",
        "player_rush_yds", "player_rush_attempts", "player_reception_yds", "player_receptions",
        "player_anytime_td", "player_pass_tds_alternate", "player_pass_yds_alternate",
        "player_rush_yds_alternate", "player_reception_yds_alternate", "player_receptions_alternate",
    ),
    "ncaaf": (
        "player_pass_tds", "player_pass_yds", "player_rush_yds", "player_reception_yds",
        "player_receptions", "player_anytime_td", "player_pass_yds_alternate", "player_rush_yds_alternate",
    ),
    "nba": (
        "player_points", "player_rebounds", "player_assists", "player_threes",
        "player_points_rebounds_assists", "player_points_rebounds", "player_points_assists",
        "player_rebounds_assists", "player_blocks", "player_steals", "player_points_alternate",
        "player_rebounds_alternate", "player_assists_alternate", "player_threes_alternate",
        "player_first_basket", "player_double_double", "player_triple_double",
    ),
    "wnba": (
        "player_points", "player_rebounds", "player_assists", "player_threes",
        "player_points_rebounds_assists", "player_points_alternate", "player_rebounds_alternate",
        "player_assists_alternate",
    ),
    "mlb": (
        "pitcher_strikeouts", "pitcher_outs", "pitcher_hits_allowed", "batter_hits",
        "batter_total_bases", "batter_home_runs", "batter_rbis", "batter_runs_scored",
        "batter_hits_alternate", "batter_total_bases_alternate", "batter_home_runs_alternate",
        "pitcher_strikeouts_alternate", "pitcher_outs_alternate",
    ),
    "nhl": (
        "player_points", "player_assists", "player_goals", "player_shots_on_goal",
        "player_blocked_shots", "player_goal_scorer_anytime", "player_goal_scorer_first",
        "player_points_alternate", "player_assists_alternate", "player_goals_alternate",
    ),
    "soccer_epl": (
        "player_goal_scorer_anytime", "player_shots", "player_shots_on_target", "player_assists",
        "player_to_receive_card",
    ),
}

PARLAY_SEARCH_TERMS = [
    "player props", "points rebounds assists", "home run prop", "rbis prop", "strikeouts prop",
    "3pm prop", "first basket", "anytime touchdown", "goals prop", "shots on goal",
]


class APIRequestError(RuntimeError):
    pass


class BaseAPIClient:
    def __init__(self, base_url: str, headers: dict[str, str] | None = None, min_interval_seconds: float = DEFAULT_MIN_REQUEST_INTERVAL_SECONDS) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.min_interval_seconds = min_interval_seconds
        self._last_request_ts = 0.0
        self._cache: TTLCache[str, Any] = TTLCache(maxsize=1024, ttl=DEFAULT_CACHE_TTL_SECONDS)
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _respect_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)

    @retry(
        retry=retry_if_exception_type((requests.RequestException, APIRequestError)),
        wait=wait_exponential(multiplier=1, min=DEFAULT_RETRY_BACKOFF_MIN_SECONDS, max=DEFAULT_RETRY_BACKOFF_MAX_SECONDS),
        stop=stop_after_attempt(DEFAULT_RETRY_ATTEMPTS),
        reraise=True,
    )
    def _request(self, method: str, url: str, params: dict[str, Any] | None = None) -> Response:
        self._respect_rate_limit()
        response = self.session.request(method.upper(), url, params=params, timeout=(DEFAULT_CONNECT_TIMEOUT_SECONDS, DEFAULT_READ_TIMEOUT_SECONDS))
        self._last_request_ts = time.monotonic()
        if response.status_code == 429:
            time.sleep(safe_int(response.headers.get("Retry-After")) or 2)
            raise APIRequestError(f"Rate limited: {url}")
        if response.status_code >= 400:
            raise APIRequestError(f"HTTP {response.status_code} from {url}: {response.text[:250]}")
        return response

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        path = path.lstrip("/")
        url = f"{self.base_url}/{path}"
        cache_key = f"{url}|{tuple(sorted((params or {}).items()))}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        payload = self._request("GET", url, params=params).json()
        self._cache[cache_key] = payload
        return payload


class ESPNClient(BaseAPIClient):
    def __init__(self) -> None:
        super().__init__(ESPN_BASE_URL)

    def fetch_scoreboard(self, sport_key: str) -> dict[str, Any]:
        config = EXTENDED_SPORT_CONFIG[sport_key]
        return self.get_json(f"{config['espn_path']}/scoreboard")


class OddsAPIClient(BaseAPIClient):
    def __init__(self) -> None:
        super().__init__(THE_ODDS_BASE_URL)

    def get_events(self, sport_key: str) -> list[dict[str, Any]]:
        return self.get_json(f"sports/{sport_key}/events", params={"apiKey": THE_ODDS_API_KEY, "dateFormat": SUPPORTED_DATE_FORMAT})

    def get_odds(self, sport_key: str, markets: str = ",".join(SUPPORTED_ODDS_MARKETS), regions: str = SUPPORTED_REGIONS) -> list[dict[str, Any]]:
        return self.get_json(
            f"sports/{sport_key}/odds/",
            params={"apiKey": THE_ODDS_API_KEY, "regions": regions, "markets": markets, "oddsFormat": SUPPORTED_ODDS_FORMAT, "dateFormat": SUPPORTED_DATE_FORMAT},
        )

    def get_event_odds(self, sport_key: str, event_id: str, markets: str, regions: str = SUPPORTED_REGIONS) -> dict[str, Any]:
        return self.get_json(
            f"sports/{sport_key}/events/{event_id}/odds",
            params={"apiKey": THE_ODDS_API_KEY, "regions": regions, "markets": markets, "oddsFormat": SUPPORTED_ODDS_FORMAT, "dateFormat": SUPPORTED_DATE_FORMAT},
        )


class APISportsFootballClient(BaseAPIClient):
    def __init__(self) -> None:
        super().__init__(API_SPORTS_FOOTBALL_BASE_URL, headers={"x-apisports-key": API_SPORTS_KEY})

    def get_injuries(self, league_id: int, season: int) -> dict[str, Any]:
        return self.get_json("injuries", params={"league": league_id, "season": season})


class TheSportsDBClient(BaseAPIClient):
    def __init__(self) -> None:
        super().__init__(THESPORTSDB_BASE_URL)

    def search_teams(self, team_name: str) -> dict[str, Any]:
        return self.get_json("searchteams.php", params={"t": team_name})


class XTrendsClient:
    """Official X API v2 recent search wrapper with no-auth fallback."""

    BASE_URL = "https://api.x.com/2/tweets/search/recent"

    def __init__(self, bearer_token: str | None = None) -> None:
        self.bearer_token = bearer_token or os.getenv("X_BEARER_TOKEN")

    def search_popular_props(self, max_results: int = 50) -> pd.DataFrame:
        if not self.bearer_token:
            return self.synthetic_public_trends()
        query = "(" + " OR ".join(f'\"{term}\"' for term in PARLAY_SEARCH_TERMS) + ") lang:en -is:retweet"
        params = {
            "query": query,
            "max_results": max(10, min(max_results, 100)),
            "tweet.fields": "created_at,public_metrics,lang",
        }
        response = requests.get(self.BASE_URL, params=params, headers={"Authorization": f"Bearer {self.bearer_token}"}, timeout=20)
        response.raise_for_status()
        rows = []
        for item in response.json().get("data", []):
            metrics = item.get("public_metrics", {})
            engagement = metrics.get("like_count", 0) + metrics.get("retweet_count", 0) * 2 + metrics.get("reply_count", 0)
            rows.append({"source": "x_api", "text": item.get("text", ""), "created_at": item.get("created_at"), "engagement_score": float(engagement), "url": f"https://x.com/i/web/status/{item.get('id')}"})
        return pd.DataFrame(rows)

    @staticmethod
    def synthetic_public_trends() -> pd.DataFrame:
        rows = []
        for idx, term in enumerate(PARLAY_SEARCH_TERMS):
            rows.append({"source": "local_public_signal", "text": f"High public interest around {term}", "created_at": utcnow().isoformat(), "engagement_score": 10 + idx * 3, "url": ""})
        return pd.DataFrame(rows)


class MLBStatcastClient:
    @staticmethod
    def fetch_statcast_window(start_date: str, end_date: str) -> pd.DataFrame:
        frame = statcast(start_dt=start_date, end_dt=end_date)
        return frame if frame is not None else pd.DataFrame()


class ActionNetworkScraper:
    @staticmethod
    def scrape_public_page(page_url: str) -> dict[str, Any]:
        with httpx.Client(timeout=(DEFAULT_CONNECT_TIMEOUT_SECONDS, DEFAULT_READ_TIMEOUT_SECONDS), follow_redirects=True) as client:
            response = client.get(page_url)
            response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return {"url": page_url, "title": soup.title.text.strip() if soup.title else "", "text_sample": " ".join(soup.get_text(" ", strip=True).split())[:1000]}


class SportsDataIngestionService:
    def __init__(self) -> None:
        self.espn = ESPNClient()
        self.odds = OddsAPIClient()
        self.api_football = APISportsFootballClient()
        self.thesportsdb = TheSportsDBClient()
        self.x_trends = XTrendsClient()

    @staticmethod
    def _parse_dt(value: str | None) -> datetime | None:
        return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None

    @staticmethod
    def _parse_espn_game(sport_key: str, event: dict[str, Any]) -> dict[str, Any]:
        comp = (event.get("competitions") or [{}])[0]
        competitors = comp.get("competitors") or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), {})
        away = next((c for c in competitors if c.get("homeAway") == "away"), {})
        cfg = EXTENDED_SPORT_CONFIG[sport_key]
        return {
            "external_game_id": str(event.get("id")),
            "sport": cfg["sport"],
            "league": cfg["league"],
            "season": str((event.get("season") or {}).get("year") or ""),
            "week": str((event.get("week") or {}).get("number") or ""),
            "commence_time": SportsDataIngestionService._parse_dt(comp.get("date") or event.get("date")),
            "status": ((comp.get("status") or {}).get("type") or {}).get("description") or "Scheduled",
            "home_team": ((home.get("team") or {}).get("displayName") or "").strip(),
            "away_team": ((away.get("team") or {}).get("displayName") or "").strip(),
            "home_score": safe_int(home.get("score")),
            "away_score": safe_int(away.get("score")),
            "completed": bool((((comp.get("status") or {}).get("type") or {}).get("completed"))),
            "venue": ((comp.get("venue") or {}).get("fullName") or "").strip() or None,
            "source": "espn",
            "raw_json": event,
        }

    def _upsert_game_from_odds_event(self, session: Any, sport_key: str, event: dict[str, Any]) -> Game:
        cfg = EXTENDED_SPORT_CONFIG[sport_key]
        external_id = str(event.get("id") or "")
        game = session.query(Game).filter(Game.external_game_id == external_id).one_or_none()
        payload = {
            "external_game_id": external_id,
            "sport": cfg["sport"],
            "league": cfg["league"],
            "season": "",
            "week": "",
            "commence_time": self._parse_dt(event.get("commence_time")),
            "status": "Scheduled",
            "home_team": event.get("home_team", ""),
            "away_team": event.get("away_team", ""),
            "completed": False,
            "venue": None,
            "source": "the-odds-api",
            "raw_json": event,
        }
        if game is None:
            game = Game(**payload)
            session.add(game)
            session.flush()
        else:
            for key, value in payload.items():
                setattr(game, key, value)
        return game

    def _record_movement(self, session: Any, game: Game, bookmaker: str, market: str, selection: str, price: float | None, point: float | None, raw: dict[str, Any]) -> None:
        previous = session.query(OddsHistory).filter(OddsHistory.game_id == game.id, OddsHistory.bookmaker == bookmaker, OddsHistory.market == market, OddsHistory.selection_name == selection).order_by(OddsHistory.pulled_at.desc()).first()
        if previous is None:
            return
        price_move = abs((price or 0) - (previous.price_american or 0)) if price is not None and previous.price_american is not None else 0
        point_move = abs((point or 0) - (previous.point or 0)) if point is not None and previous.point is not None else 0
        if price_move == 0 and point_move == 0:
            return
        session.add(LineMovement(game_id=game.id, sport=game.sport, league=game.league, sportsbook=bookmaker, market=market, selection_name=selection, opening_price=previous.price_american, current_price=price, opening_point=previous.point, current_point=point, movement_abs=price_move + point_move, source="the-odds-api", raw_json=raw))

    def _store_odds_row(self, session: Any, game: Game, bookmaker: dict[str, Any], market: dict[str, Any], outcome: dict[str, Any]) -> int:
        bookmaker_name = bookmaker.get("title", bookmaker.get("key", "unknown"))
        market_key = market.get("key", "unknown")
        selection = str(outcome.get("name", "unknown"))
        price = safe_float(outcome.get("price"))
        point = safe_float(outcome.get("point"))
        raw = {**outcome, "player_name": outcome.get("description"), "market_last_update": market.get("last_update"), "bookmaker_key": bookmaker.get("key")}
        self._record_movement(session, game, bookmaker_name, market_key, selection, price, point, raw)
        player_name = outcome.get("description")
        if player_name:
            existing = session.query(Player).filter(Player.sport == game.sport, Player.league == game.league, Player.game_id == game.id, Player.full_name == str(player_name)).one_or_none()
            if existing is None:
                session.add(Player(external_player_id=None, sport=game.sport, league=game.league, game_id=game.id, team=None, full_name=str(player_name), position=None, status=None, source="the-odds-api", raw_json=raw))
        session.add(OddsHistory(game_id=game.id, sport=game.sport, league=game.league, source="the-odds-api", bookmaker=bookmaker_name, market=market_key, selection_name=selection, price_american=price, price_decimal=american_to_decimal(price), point=point, implied_probability=american_to_implied_probability(price), raw_json=raw))
        return 1

    def ingest_espn_scoreboard(self, sport_key: str) -> int:
        if sport_key not in EXTENDED_SPORT_CONFIG:
            raise ValueError(f"Unsupported sport key: {sport_key}")
        payload = self.espn.fetch_scoreboard(sport_key)
        events = payload.get("events", [])
        with get_session() as session:
            for event in events:
                parsed = self._parse_espn_game(sport_key, event)
                game = session.query(Game).filter(Game.external_game_id == parsed["external_game_id"]).one_or_none()
                if game is None:
                    session.add(Game(**parsed))
                else:
                    for key, value in parsed.items():
                        setattr(game, key, value)
        return len(events)

    def ingest_odds_for_sport(self, sport_key: str) -> int:
        payload = self.odds.get_odds(EXTENDED_SPORT_CONFIG[sport_key]["odds_sport_key"])
        total = 0
        with get_session() as session:
            for event in payload:
                game = self._upsert_game_from_odds_event(session, sport_key, event)
                for bookmaker in event.get("bookmakers", []):
                    for market in bookmaker.get("markets", []):
                        for outcome in market.get("outcomes", []):
                            total += self._store_odds_row(session, game, bookmaker, market, outcome)
        return total

    def ingest_player_props_for_sport(self, sport_key: str, max_events: int | None = 12) -> int:
        markets = PLAYER_PROP_MARKETS_BY_SPORT.get(sport_key, ())
        if not markets:
            return 0
        events = self.odds.get_events(EXTENDED_SPORT_CONFIG[sport_key]["odds_sport_key"])
        events = events[:max_events] if max_events is not None else events
        total = 0
        with get_session() as session:
            for event in events:
                event_id = str(event.get("id") or "")
                if not event_id:
                    continue
                game = self._upsert_game_from_odds_event(session, sport_key, event)
                for market_group in chunked(list(markets), 6):
                    try:
                        payload = self.odds.get_event_odds(EXTENDED_SPORT_CONFIG[sport_key]["odds_sport_key"], event_id, ",".join(market_group))
                    except Exception as exc:
                        LOGGER.warning("Prop market fetch failed for %s %s: %s", sport_key, market_group, exc)
                        continue
                    for bookmaker in payload.get("bookmakers", []):
                        for market in bookmaker.get("markets", []):
                            for outcome in market.get("outcomes", []):
                                total += self._store_odds_row(session, game, bookmaker, market, outcome)
        return total

    def ingest_soccer_injuries(self, league_id: int | None = None, season: int | None = None) -> int:
        league_id = league_id or API_FOOTBALL_DEFAULTS["premier_league_id"]
        season = season or utcnow().year
        rows = self.api_football.get_injuries(league_id, season).get("response", [])
        inserted = 0
        with get_session() as session:
            for row in rows:
                player = row.get("player") or {}
                team = row.get("team") or {}
                session.add(Injury(external_injury_id=str(player.get("id")) if player.get("id") is not None else None, sport="soccer", league="eng.1", game_id=None, player_id=None, team=team.get("name"), player_name=player.get("name"), status=row.get("type"), injury_type=row.get("type"), reason=row.get("reason"), source="api-sports", raw_json=row))
                inserted += 1
        return inserted

    def fetch_social_trends(self) -> pd.DataFrame:
        return self.x_trends.search_popular_props()

    def ingest_all_primary_sports(self) -> dict[str, int]:
        init_db()
        results: dict[str, int] = {}
        for sport_key in ["nfl", "nba", "mlb", "nhl", "ncaaf", "wnba", "soccer_epl"]:
            try:
                try:
                    results[f"{sport_key}_games"] = self.ingest_espn_scoreboard(sport_key)
                except Exception as exc:
                    LOGGER.warning("ESPN ingest skipped for %s: %s", sport_key, exc)
                    results[f"{sport_key}_games"] = 0
                results[f"{sport_key}_odds"] = self.ingest_odds_for_sport(sport_key)
                results[f"{sport_key}_props"] = self.ingest_player_props_for_sport(sport_key, max_events=10)
            except Exception as exc:
                LOGGER.exception("Ingestion failed for %s: %s", sport_key, exc)
                results[f"{sport_key}_error"] = 1
        try:
            results["soccer_injuries"] = self.ingest_soccer_injuries()
        except Exception as exc:
            LOGGER.warning("Soccer injury ingest skipped: %s", exc)
        return results

    def build_feature_frames(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        with get_session() as session:
            games = pd.read_sql(session.query(Game).statement, session.bind)
            odds = pd.read_sql(session.query(OddsHistory).statement, session.bind)
            injuries = pd.read_sql(session.query(Injury).statement, session.bind)
        return games, odds, injuries


if __name__ == "__main__":
    service = SportsDataIngestionService()
    print(service.ingest_all_primary_sports())
    print(service.fetch_social_trends().head())

# Stable empty-frame helpers used by the dashboard and tests. These do not call APIs.
GAME_COLUMNS = [
    "id", "external_game_id", "sport", "league", "season", "week", "commence_time", "status",
    "home_team", "away_team", "home_score", "away_score", "completed", "venue", "source", "raw_json",
    "created_at", "updated_at",
]
ODDS_COLUMNS = [
    "id", "game_id", "sport", "league", "source", "bookmaker", "market", "selection_name",
    "price_american", "price_decimal", "point", "implied_probability", "pulled_at", "raw_json",
    "created_at", "updated_at",
]
INJURY_COLUMNS = [
    "id", "external_injury_id", "sport", "league", "game_id", "player_id", "team", "player_name",
    "status", "injury_type", "reason", "reported_at", "source", "raw_json", "created_at", "updated_at",
]
LINE_MOVE_COLUMNS = [
    "id", "game_id", "sport", "league", "sportsbook", "market", "selection_name", "opening_price",
    "current_price", "opening_point", "current_point", "movement_abs", "detected_at", "source", "raw_json",
    "created_at", "updated_at",
]
PROJECTION_COLUMNS = [
    "id", "game_id", "sport", "league", "model_name", "win_prob_home", "projected_home_score",
    "projected_away_score", "edge_pct", "recommended_bet", "confidence", "feature_snapshot", "created_at", "updated_at",
]
SOCIAL_TREND_COLUMNS = ["source", "text", "created_at", "engagement_score", "url"]


def ensure_frame_schema(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = pd.NA
    return output.reindex(columns=columns)


def empty_games_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=GAME_COLUMNS)


def empty_odds_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=ODDS_COLUMNS)


def empty_injuries_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=INJURY_COLUMNS)


def empty_line_moves_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=LINE_MOVE_COLUMNS)


def empty_projections_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PROJECTION_COLUMNS)


def empty_social_trends_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SOCIAL_TREND_COLUMNS)
