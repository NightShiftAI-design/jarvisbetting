"""
Central configuration for Jarvis_Betting.

This file intentionally hardcodes the exact API keys provided by the user.
In a production system you would normally inject secrets via environment variables,
but this project explicitly requires hardcoded values in this file.

Verified data sources used by this project:

1. ESPN public API
   Base URL: https://site.api.espn.com/apis/site/v2/sports/
   Verification note: opened directly on 2026-04-23 and confirmed JSON responses for:
     - https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard
     - https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard
     - https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard
     - https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard

2. The Odds API
   Base URL: https://api.the-odds-api.com/v4/
   Verification note: official docs confirm host https://api.the-odds-api.com and endpoints
   under /v4/sports/ as of 2026-04-23.

3. API-Football / API-Sports
   Base URL: https://v3.football.api-sports.io/
   Verification note: official API-Football docs/tutorials confirm v3 base URL and
   x-apisports-key header auth as of 2026-04-23.

4. TheSportsDB
   Base URL: https://www.thesportsdb.com/api/v1/json/3/
   Verification note: official docs confirm free V1 API at /api/v1/json/{key}/ as of 2026-04-23.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final


APP_NAME: Final[str] = "Jarvis_Betting"
APP_VERSION: Final[str] = "0.1.0"

# Exact API keys provided by the user.
THE_ODDS_API_KEY: Final[str] = "ef39d572412af88ebe3449c236226fc8"
API_SPORTS_KEY: Final[str] = "2b0b725ec3c107fe57f6d77002bf4267"
THESPORTSDB_KEY: Final[str] = "3"

# Verified base URLs.
ESPN_BASE_URL: Final[str] = "https://site.api.espn.com/apis/site/v2/sports"
THE_ODDS_BASE_URL: Final[str] = "https://api.the-odds-api.com/v4"
API_SPORTS_FOOTBALL_BASE_URL: Final[str] = "https://v3.football.api-sports.io"
THESPORTSDB_BASE_URL: Final[str] = f"https://www.thesportsdb.com/api/v1/json/{THESPORTSDB_KEY}"

# Storage paths.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
LOG_DIR: Final[Path] = PROJECT_ROOT / "logs"
MODEL_DIR: Final[Path] = PROJECT_ROOT / "artifacts"
DATABASE_PATH: Final[Path] = DATA_DIR / "jarvis_betting.db"

# Logging.
LOG_LEVEL: Final[str] = "INFO"
LOG_FILE: Final[Path] = LOG_DIR / "jarvis_betting.log"

# Networking behavior.
DEFAULT_TIMEOUT_SECONDS: Final[float] = 20.0
DEFAULT_CONNECT_TIMEOUT_SECONDS: Final[float] = 10.0
DEFAULT_READ_TIMEOUT_SECONDS: Final[float] = 20.0
DEFAULT_RETRY_ATTEMPTS: Final[int] = 4
DEFAULT_RETRY_BACKOFF_MIN_SECONDS: Final[int] = 1
DEFAULT_RETRY_BACKOFF_MAX_SECONDS: Final[int] = 8
DEFAULT_CACHE_TTL_SECONDS: Final[int] = 300
DEFAULT_MIN_REQUEST_INTERVAL_SECONDS: Final[float] = 0.6

# Scheduler behavior.
NFL_REFRESH_INTERVAL_MINUTES: Final[int] = 20
NON_NFL_REFRESH_INTERVAL_MINUTES: Final[int] = 45
MODEL_RETRAIN_INTERVAL_HOURS: Final[int] = 6

# Betting thresholds.
DEFAULT_EDGE_THRESHOLD_PCT: Final[float] = 0.03
DEFAULT_CONFIDENCE_FLOOR: Final[float] = 0.52
MAX_BACKTEST_BET_SIZE_UNITS: Final[float] = 1.0

# Supported leagues. NFL is the primary first-class workflow.
SPORT_CONFIG: Final[dict[str, dict[str, str]]] = {
    "nfl": {
        "sport": "football",
        "league": "nfl",
        "display_name": "NFL",
        "odds_sport_key": "americanfootball_nfl",
        "espn_path": "football/nfl",
    },
    "nba": {
        "sport": "basketball",
        "league": "nba",
        "display_name": "NBA",
        "odds_sport_key": "basketball_nba",
        "espn_path": "basketball/nba",
    },
    "mlb": {
        "sport": "baseball",
        "league": "mlb",
        "display_name": "MLB",
        "odds_sport_key": "baseball_mlb",
        "espn_path": "baseball/mlb",
    },
    "soccer_epl": {
        "sport": "soccer",
        "league": "eng.1",
        "display_name": "Soccer - EPL",
        "odds_sport_key": "soccer_epl",
        "espn_path": "soccer/eng.1",
    },
}

# Soccer defaults for API-Football examples. These can be expanded later.
API_FOOTBALL_DEFAULTS: Final[dict[str, int]] = {
    "premier_league_id": 39,
    "la_liga_id": 140,
    "champions_league_id": 2,
}

# Supported odds markets for The Odds API.
SUPPORTED_ODDS_MARKETS: Final[tuple[str, ...]] = ("h2h", "spreads", "totals")
SUPPORTED_REGIONS: Final[str] = "us"
SUPPORTED_ODDS_FORMAT: Final[str] = "american"
SUPPORTED_DATE_FORMAT: Final[str] = "iso"


def ensure_directories() -> None:
    """Create local runtime directories if they do not exist."""
    for path in (DATA_DIR, LOG_DIR, MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)


# Make runtime directories available on import.
ensure_directories()


if __name__ == "__main__":
    print(f"{APP_NAME} {APP_VERSION}")
    print(f"Database path: {DATABASE_PATH}")
    print("Supported leagues:", ", ".join(SPORT_CONFIG))
    print("Test this now:")
    print("python config.py")
