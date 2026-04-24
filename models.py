"""Model, prop scoring, and realistic parlay math helpers for Jarvis_Betting."""

from __future__ import annotations

import itertools
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DEFAULT_CONFIDENCE_FLOOR, DEFAULT_EDGE_THRESHOLD_PCT, MODEL_DIR
from utils import LOGGER, american_to_decimal, clamp, decimal_to_american, is_sharp_book, safe_float, safe_json_loads

STRONG_PF_MIN = 72
STRONG_EDGE_MIN = 0.012
STRONG_CONFIDENCE_MIN = 0.50
MAX_DISPLAY_EDGE = 0.18
MAX_LINE_DELTA = 12.0
MAIN_LINE_MAX_DELTA = 8.0
REASONABLE_ALT_MAX_DELTA = 8.0
MIN_REALISM_SCORE = 62
LONGSHOT_CONFIDENCE_MIN = 0.72
LONGSHOT_EDGE_MIN = 0.08
WEATHER_SENSITIVE_PROP_TYPES = {
    "Homeruns",
    "Hits",
    "Bases / Total Bases",
    "RBIs",
    "Passing",
    "Receiving",
    "Rushing",
    "Goals",
    "Shots / Saves",
}

PROP_TYPE_ORDER = [
    "All",
    "Points",
    "Rebounds",
    "Assists",
    "Threes / 3PM",
    "Homeruns",
    "Hits",
    "Bases / Total Bases",
    "RBIs",
    "Goals",
    "Shots / Saves",
    "Passing",
    "Rushing",
    "Receiving",
    "Strikeouts",
    "Touchdowns",
    "Other",
]

PROP_COLUMNS = [
    "id", "game_id", "sport", "league", "home_team", "away_team", "commence_time", "status", "completed",
    "matchup", "bookmaker", "market", "market_display", "prop_type", "selection_name", "prop_side",
    "player_name", "point", "projected_line", "line_delta", "price_american", "price_decimal",
    "implied_probability", "market_probability", "projected_probability", "edge_pct", "display_edge_pct",
    "confidence", "pf_score", "realism_score", "variance_flag", "hit_rate_l5", "hit_rate_l10", "book_count",
    "sharp_prob", "sharp_point", "sharp_books", "social_score", "social_mentions", "injury_flag",
    "injury_note", "weather_impact", "weather_note", "trend_note", "sharp_edge_vs_pinnacle",
    "sharp_confirmed", "line_movement_signal", "best_available", "positive_ev", "reasonable_line",
    "strong_pick", "why", "raw_json", "pulled_at",
]


@dataclass
class PredictionBundle:
    model_name: str
    features_used: list[str]
    frame: pd.DataFrame


PROP_MARKET_SCALES: dict[str, float] = {
    "player_pass_tds": 0.30,
    "player_pass_yds": 14.0,
    "player_rush_yds": 8.0,
    "player_reception_yds": 8.0,
    "player_receptions": 0.7,
    "player_anytime_td": 0.07,
    "player_points": 2.1,
    "player_rebounds": 1.0,
    "player_assists": 0.8,
    "player_threes": 0.45,
    "player_points_rebounds_assists": 3.0,
    "pitcher_strikeouts": 0.65,
    "pitcher_outs": 0.9,
    "batter_hits": 0.25,
    "batter_total_bases": 0.45,
    "batter_home_runs": 0.035,
    "batter_rbis": 0.22,
    "player_shots_on_goal": 0.45,
    "player_total_saves": 2.5,
    "player_goals": 0.06,
}


def ensure_columns(frame: pd.DataFrame, columns: list[str], defaults: dict[str, Any] | None = None) -> pd.DataFrame:
    defaults = defaults or {}
    output = frame.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = defaults.get(column, np.nan)
    return output.reindex(columns=list(dict.fromkeys([*output.columns, *columns])))


def empty_props_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PROP_COLUMNS)


def market_display_name(market: str) -> str:
    return str(market or "unknown").replace("_", " ").title()


def prop_type_for_market(market: Any) -> str:
    key = str(market or "").lower()
    if "points_rebounds_assists" in key or key == "player_points" or "player_points_alternate" in key:
        return "Points"
    if "rebounds" in key:
        return "Rebounds"
    if "assists" in key and "goal" not in key:
        return "Assists"
    if "threes" in key or "3pm" in key:
        return "Threes / 3PM"
    if "home_runs" in key or "home_run" in key:
        return "Homeruns"
    if "batter_hits" in key:
        return "Hits"
    if "total_bases" in key:
        return "Bases / Total Bases"
    if "rbis" in key:
        return "RBIs"
    if "goals" in key or "goal_scorer" in key:
        return "Goals"
    if "shots" in key or "saves" in key:
        return "Shots / Saves"
    if "pass" in key:
        return "Passing"
    if "rush" in key:
        return "Rushing"
    if "reception" in key or "receiving" in key:
        return "Receiving"
    if "strikeouts" in key:
        return "Strikeouts"
    if "touchdown" in key or key.endswith("_td") or "anytime_td" in key:
        return "Touchdowns"
    return "Other"


def get_available_prop_types(props_df: pd.DataFrame) -> list[str]:
    props_df = ensure_columns(props_df, ["prop_type"])
    available = set(props_df["prop_type"].dropna().astype(str)) if not props_df.empty else set()
    ordered = [prop_type for prop_type in PROP_TYPE_ORDER if prop_type == "All" or prop_type in available]
    return ordered or ["All"]


def get_props_by_type(props_df: pd.DataFrame, prop_type: str | list[str] | None) -> pd.DataFrame:
    props_df = ensure_columns(props_df, PROP_COLUMNS)
    if props_df.empty or prop_type is None:
        return props_df
    selected = [prop_type] if isinstance(prop_type, str) else list(prop_type)
    if not selected or "All" in selected:
        return props_df
    return props_df[props_df["prop_type"].astype(str).isin(selected)].copy()


def market_scale(market: str) -> float:
    market = str(market or "")
    base = market.replace("_alternate", "")
    scale = PROP_MARKET_SCALES.get(base, 1.0)
    return scale * 0.70 if market.endswith("_alternate") else scale


def infer_player_name(row: pd.Series) -> str:
    raw = safe_json_loads(row.get("raw_json"))
    player = raw.get("player_name") or raw.get("description")
    selection = str(row.get("selection_name") or "").strip()
    if player:
        return str(player)
    if selection not in {"Over", "Under", "Yes", "No"}:
        return selection
    return "Unknown Player"


def clean_side(value: Any) -> str:
    side = str(value or "").title()
    if side in {"Over", "Under", "Yes", "No"}:
        return side
    return side or "Over"


def is_longshot_alt(row: pd.Series) -> bool:
    market = str(row.get("market") or "")
    price = safe_float(row.get("price_american"))
    probability = safe_float(row.get("market_probability")) or 0.5
    line_delta = abs(safe_float(row.get("line_delta")) or 0.0)
    return market.endswith("_alternate") and ((price is not None and price >= 250) or probability <= 0.30 or line_delta > REASONABLE_ALT_MAX_DELTA)


def calculate_realism_score(row: pd.Series) -> int:
    line_delta = abs(safe_float(row.get("line_delta")) or 0.0)
    confidence = safe_float(row.get("confidence")) or 0.0
    market_probability = safe_float(row.get("market_probability")) or 0.50
    edge = safe_float(row.get("edge_pct")) or 0.0
    market = str(row.get("market") or "")
    injury_penalty = 18 if bool(row.get("injury_flag")) else 0
    alt_penalty = 16 if market.endswith("_alternate") else 0
    longshot_penalty = 20 if is_longshot_alt(row) else 0
    distance_penalty = min(38, line_delta * 3.0)
    probability_penalty = max(0.0, 0.42 - market_probability) * 60
    edge_penalty = max(0.0, edge - MAX_DISPLAY_EDGE) * 80
    confidence_bonus = confidence * 18
    score = 100 - distance_penalty - probability_penalty - alt_penalty - longshot_penalty - injury_penalty - edge_penalty + confidence_bonus
    return int(round(clamp(score, 0, 100)))


def _context_from_game_raw(row: pd.Series) -> dict[str, Any]:
    raw = safe_json_loads(row.get("raw_json_game"))
    if not raw:
        raw = safe_json_loads(row.get("game_raw_json"))
    return raw if isinstance(raw, dict) else {}


def _weather_context(row: pd.Series) -> dict[str, Any]:
    raw = _context_from_game_raw(row)
    context = raw.get("weather_context") or raw.get("weather") or {}
    return context if isinstance(context, dict) else {}


def weather_adjustment(row: pd.Series) -> tuple[float, str]:
    prop_type = str(row.get("prop_type") or "Other")
    if prop_type not in WEATHER_SENSITIVE_PROP_TYPES:
        return 0.0, "Weather neutral for this prop type"
    weather = _weather_context(row)
    if not weather:
        return 0.0, "Weather unavailable"
    wind_mph = safe_float(weather.get("wind_mph")) or 0.0
    temperature = safe_float(weather.get("temperature_f"))
    precipitation = safe_float(weather.get("precipitation_probability")) or 0.0
    wind_direction = str(weather.get("wind_direction") or "").lower()
    short_forecast = str(weather.get("short_forecast") or "").lower()
    impact = 0.0
    notes: list[str] = []
    if prop_type in {"Homeruns", "Hits", "Bases / Total Bases", "RBIs"}:
        if wind_mph >= 8 and any(token in wind_direction for token in ["out", "s", "sw", "w"]):
            impact += 0.012
            notes.append(f"{wind_mph:.0f}mph wind helps carry")
        elif wind_mph >= 8:
            impact -= 0.014
            notes.append(f"{wind_mph:.0f}mph wind suppresses carry")
        if temperature is not None and temperature >= 78:
            impact += 0.006
            notes.append("warm hitting weather")
        if precipitation >= 35 or "rain" in short_forecast:
            impact -= 0.010
            notes.append("rain risk")
    elif prop_type in {"Passing", "Receiving", "Rushing", "Goals", "Shots / Saves"}:
        if wind_mph >= 15:
            impact -= 0.010
            notes.append(f"{wind_mph:.0f}mph wind adds variance")
        if precipitation >= 35 or "rain" in short_forecast or "snow" in short_forecast:
            impact -= 0.008
            notes.append("precipitation risk")
    return clamp(impact, -0.035, 0.025), "; ".join(notes) if notes else "Weather checked, no major adjustment"


def trend_adjustment(row: pd.Series) -> tuple[float, str]:
    social_score = safe_float(row.get("social_score")) or 0.0
    book_count = safe_float(row.get("book_count")) or 0.0
    hit_rate_l10 = safe_float(row.get("hit_rate_l10")) or 0.50
    impact = 0.0
    notes: list[str] = []
    if book_count >= 5:
        impact += 0.003
        notes.append("broad book coverage")
    if social_score >= 35:
        impact += 0.003
        notes.append("public trend signal")
    if hit_rate_l10 >= 0.58:
        impact += 0.004
        notes.append("recent form supports")
    elif hit_rate_l10 <= 0.45:
        impact -= 0.006
        notes.append("recent form weak")
    return clamp(impact, -0.015, 0.012), "; ".join(notes) if notes else "No strong recent trend signal"


def line_movement_adjustment(row: pd.Series) -> tuple[float, str]:
    raw = safe_json_loads(row.get("raw_json"))
    movement = safe_float(raw.get("line_movement_signal")) or safe_float(row.get("line_movement_signal")) or 0.0
    if movement >= 20:
        return 0.010, "strong positive line movement"
    if movement >= 8:
        return 0.004, "modest line movement"
    return 0.0, "No strong line movement"


class FeatureBuilder:
    @staticmethod
    def _latest_moneyline_snapshot(odds_df: pd.DataFrame) -> pd.DataFrame:
        odds_df = ensure_columns(odds_df, ["market", "pulled_at", "game_id", "selection_name", "implied_probability"])
        if odds_df.empty:
            return pd.DataFrame()
        moneyline = odds_df[odds_df["market"] == "h2h"].copy()
        if moneyline.empty:
            return pd.DataFrame()
        moneyline["pulled_at"] = pd.to_datetime(moneyline["pulled_at"], utc=True, errors="coerce")
        moneyline = moneyline.sort_values("pulled_at").groupby(["game_id", "selection_name"], as_index=False).tail(1)
        pivot = moneyline.pivot_table(index="game_id", columns="selection_name", values="implied_probability", aggfunc="last")
        pivot.columns = [f"market_prob_{str(col).strip().lower().replace(' ', '_')}" for col in pivot.columns]
        return pivot.reset_index()

    @staticmethod
    def _injury_counts(injuries_df: pd.DataFrame) -> pd.DataFrame:
        injuries_df = ensure_columns(injuries_df, ["game_id"])
        if injuries_df.empty:
            return pd.DataFrame(columns=["game_id", "injury_count"])
        return injuries_df.groupby("game_id", dropna=True).size().reset_index(name="injury_count")

    def build(self, games_df: pd.DataFrame, odds_df: pd.DataFrame, injuries_df: pd.DataFrame) -> pd.DataFrame:
        games_df = ensure_columns(games_df, ["id", "commence_time", "home_score", "away_score", "completed", "league", "sport", "home_team"])
        if games_df.empty:
            return pd.DataFrame()
        frame = games_df.copy()
        frame["commence_time"] = pd.to_datetime(frame["commence_time"], utc=True, errors="coerce")
        frame["home_score"] = pd.to_numeric(frame["home_score"], errors="coerce")
        frame["away_score"] = pd.to_numeric(frame["away_score"], errors="coerce")
        frame["home_win"] = np.where(frame["completed"].fillna(False), (frame["home_score"] > frame["away_score"]).astype(int), np.nan)
        frame["score_diff"] = frame["home_score"] - frame["away_score"]
        frame["total_points"] = frame["home_score"] + frame["away_score"]
        frame["is_nfl"] = frame["league"].eq("nfl").astype(int)
        frame["is_nba"] = frame["league"].eq("nba").astype(int)
        frame["is_mlb"] = frame["league"].eq("mlb").astype(int)
        frame["is_soccer"] = frame["sport"].eq("soccer").astype(int)
        frame["is_nhl"] = frame["league"].eq("nhl").astype(int)
        moneyline = self._latest_moneyline_snapshot(odds_df)
        if not moneyline.empty:
            frame = frame.merge(moneyline, how="left", left_on="id", right_on="game_id")
        injuries = self._injury_counts(injuries_df)
        if not injuries.empty:
            frame = frame.merge(injuries, how="left", left_on="id", right_on="game_id")
        frame["injury_count"] = frame["injury_count"].fillna(0) if "injury_count" in frame else 0
        frame["market_home_probability"] = 0.50
        for idx, row in frame.iterrows():
            key = f"market_prob_{str(row['home_team']).strip().lower().replace(' ', '_')}"
            if key in frame.columns and pd.notna(row.get(key)):
                frame.loc[idx, "market_home_probability"] = row.get(key)
        frame["days_to_game"] = (frame["commence_time"] - pd.Timestamp.now(tz="UTC")).dt.total_seconds().fillna(0) / 86400.0
        return frame


class JarvisPredictor:
    def __init__(self) -> None:
        self.features = ["market_home_probability", "injury_count", "days_to_game", "is_nfl", "is_nba", "is_mlb", "is_soccer", "is_nhl"]
        self.win_model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])
        self.home_score_model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LinearRegression())])
        self.away_score_model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LinearRegression())])
        self.is_trained = False

    def train(self, feature_frame: pd.DataFrame) -> None:
        feature_frame = ensure_columns(feature_frame, [*self.features, "home_win", "home_score", "away_score"])
        trainable = feature_frame.dropna(subset=["home_win", "home_score", "away_score"]).copy()
        if len(trainable) < 10:
            LOGGER.warning("Insufficient historical rows to fully train; using market-informed fallback.")
            self.is_trained = False
            return
        x = trainable[self.features]
        self.win_model.fit(x, trainable["home_win"].astype(int))
        self.home_score_model.fit(x, trainable["home_score"])
        self.away_score_model.fit(x, trainable["away_score"])
        self.is_trained = True

    def predict(self, feature_frame: pd.DataFrame) -> PredictionBundle:
        feature_frame = ensure_columns(feature_frame, self.features)
        if feature_frame.empty:
            return PredictionBundle("jarvis_v3_conservative", self.features, pd.DataFrame())
        scoring = feature_frame.copy()
        x = scoring[self.features]
        if self.is_trained:
            scoring["win_prob_home"] = self.win_model.predict_proba(x)[:, 1]
            scoring["projected_home_score"] = self.home_score_model.predict(x)
            scoring["projected_away_score"] = self.away_score_model.predict(x)
        else:
            scoring["win_prob_home"] = scoring["market_home_probability"].fillna(0.50)
            scoring["projected_home_score"] = np.where(scoring.get("league", "").isin(["nfl", "nba"]), 24.0, 3.2)
            scoring["projected_away_score"] = np.where(scoring.get("league", "").isin(["nfl", "nba"]), 21.0, 2.8)
        scoring["edge_pct"] = (scoring["win_prob_home"] - scoring["market_home_probability"]).clip(-MAX_DISPLAY_EDGE, MAX_DISPLAY_EDGE)
        scoring["confidence"] = scoring["win_prob_home"].apply(lambda value: abs(value - 0.50) * 2.0 if pd.notna(value) else 0)
        scoring["recommended_bet"] = scoring.apply(self._recommend_bet, axis=1)
        return PredictionBundle("jarvis_v3_conservative", self.features, scoring)

    @staticmethod
    def _recommend_bet(row: pd.Series) -> str | None:
        edge = safe_float(row.get("edge_pct")) or 0.0
        confidence = safe_float(row.get("confidence")) or 0.0
        if edge >= DEFAULT_EDGE_THRESHOLD_PCT and confidence >= DEFAULT_CONFIDENCE_FLOOR:
            return f"Moneyline: {row.get('home_team')}"
        if edge <= -DEFAULT_EDGE_THRESHOLD_PCT and confidence >= DEFAULT_CONFIDENCE_FLOOR:
            return f"Moneyline: {row.get('away_team')}"
        return None

    def save(self, filename: str = "jarvis_predictor.pkl") -> Path:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / filename
        with path.open("wb") as handle:
            pickle.dump(self, handle)
        return path

    @staticmethod
    def load(path: str | Path) -> "JarvisPredictor":
        with Path(path).open("rb") as handle:
            return pickle.load(handle)


def build_player_props_frame(odds_df: pd.DataFrame, games_df: pd.DataFrame, injuries_df: pd.DataFrame | None = None, social_df: pd.DataFrame | None = None) -> pd.DataFrame:
    odds_df = ensure_columns(odds_df, ["id", "game_id", "sport", "league", "bookmaker", "market", "selection_name", "price_american", "price_decimal", "point", "implied_probability", "pulled_at", "raw_json"])
    games_df = ensure_columns(games_df, ["id", "sport", "league", "home_team", "away_team", "commence_time", "status", "completed"])
    if odds_df.empty or games_df.empty:
        return empty_props_frame()
    props = odds_df[odds_df["market"].astype(str).str.startswith(("player_", "batter_", "pitcher_"), na=False)].copy()
    if props.empty:
        return empty_props_frame()
    props["pulled_at"] = pd.to_datetime(props["pulled_at"], utc=True, errors="coerce")
    props["player_name"] = props.apply(infer_player_name, axis=1)
    props["prop_side"] = props["selection_name"].apply(clean_side)
    props["point"] = pd.to_numeric(props["point"], errors="coerce")
    props["price_american"] = pd.to_numeric(props["price_american"], errors="coerce")
    props["implied_probability"] = pd.to_numeric(props["implied_probability"], errors="coerce").fillna(0.50)
    props["is_sharp_book"] = props["bookmaker"].apply(is_sharp_book)
    latest = props.sort_values("pulled_at").groupby(["game_id", "bookmaker", "market", "player_name", "prop_side", "point"], dropna=False).tail(1)
    latest = latest.merge(games_df[["id", "sport", "league", "home_team", "away_team", "commence_time", "status", "completed", "raw_json"]], left_on="game_id", right_on="id", how="left", suffixes=("", "_game"))
    if "sport_game" in latest:
        latest["sport"] = latest["sport"].fillna(latest["sport_game"])
    if "league_game" in latest:
        latest["league"] = latest["league"].fillna(latest["league_game"])
    latest["matchup"] = latest["away_team"].fillna("TBD") + " @ " + latest["home_team"].fillna("TBD")
    latest["market_display"] = latest["market"].apply(market_display_name)
    latest["prop_type"] = latest["market"].apply(prop_type_for_market)
    consensus = latest.groupby(["game_id", "market", "player_name", "prop_side"], dropna=False).agg(
        consensus_prob=("implied_probability", "mean"),
        consensus_point=("point", "mean"),
        book_count=("bookmaker", "nunique"),
        avg_price=("price_american", "mean"),
    ).reset_index()
    sharp = latest[latest["is_sharp_book"]].groupby(["game_id", "market", "player_name", "prop_side"], dropna=False).agg(
        sharp_prob=("implied_probability", "mean"),
        sharp_point=("point", "mean"),
        sharp_books=("bookmaker", lambda values: ", ".join(sorted(set(map(str, values))))),
    ).reset_index()
    latest = latest.merge(consensus, on=["game_id", "market", "player_name", "prop_side"], how="left")
    latest = latest.merge(sharp, on=["game_id", "market", "player_name", "prop_side"], how="left")
    social_strength = _social_strength(latest, social_df)
    latest = latest.merge(social_strength, on=["player_name", "market"], how="left")
    latest["social_score"] = latest["social_score"].fillna(0.0)
    latest["social_mentions"] = latest["social_mentions"].fillna(0)
    latest["injury_flag"] = False
    if injuries_df is not None and not injuries_df.empty:
        injuries_df = ensure_columns(injuries_df, ["player_name"])
        injured = set(injuries_df["player_name"].dropna().astype(str).str.lower())
        latest["injury_flag"] = latest["player_name"].astype(str).str.lower().isin(injured)

    base_bias = np.where(latest["book_count"] >= 5, 0.010, np.where(latest["book_count"] >= 3, 0.007, 0.003))
    sharp_delta = (latest["sharp_prob"].fillna(latest["consensus_prob"]) - latest["consensus_prob"]).fillna(0)
    social_boost = latest["social_score"].clip(0, 100) / 5000.0
    side_sign = np.where(latest["prop_side"].isin(["Over", "Yes"]), 1, -1)
    latest["projected_probability"] = latest["consensus_prob"].fillna(0.50) + base_bias * side_sign + sharp_delta * 0.18 + social_boost
    weather_results = latest.apply(weather_adjustment, axis=1)
    trend_results = latest.apply(trend_adjustment, axis=1)
    movement_results = latest.apply(line_movement_adjustment, axis=1)
    latest["weather_impact"] = [item[0] for item in weather_results]
    latest["weather_note"] = [item[1] for item in weather_results]
    latest["trend_note"] = [item[1] for item in trend_results]
    latest["line_movement_signal"] = [item[1] for item in movement_results]
    latest["injury_note"] = np.where(latest["injury_flag"], "Player appears on stored injury report", "No stored injury flag")
    injury_penalty = np.where(latest["injury_flag"], -0.060, 0.0)
    latest["projected_probability"] = latest["projected_probability"] + latest["weather_impact"] + [item[0] for item in trend_results] + [item[0] for item in movement_results] + injury_penalty
    latest["projected_probability"] = latest["projected_probability"].apply(lambda value: clamp(float(value), 0.05, 0.92))
    latest["market_probability"] = latest["implied_probability"].fillna(0.50)
    latest["sharp_edge_vs_pinnacle"] = latest["projected_probability"] - latest["sharp_prob"]
    latest["sharp_confirmed"] = latest["sharp_prob"].isna() | (latest["sharp_edge_vs_pinnacle"] > 0)
    raw_edge = latest["projected_probability"] - latest["market_probability"]
    latest["edge_pct"] = raw_edge.clip(-MAX_DISPLAY_EDGE, MAX_DISPLAY_EDGE)
    latest["display_edge_pct"] = latest["edge_pct"]
    latest["projected_line"] = latest["point"] + ((latest["projected_probability"] - 0.50) * 2.0 * latest["market"].apply(market_scale))
    latest["line_delta"] = (latest["projected_line"] - latest["point"]).abs()
    latest["hit_rate_l5"] = (0.50 + latest["edge_pct"] * 1.25 + latest["social_score"].clip(0, 100) / 1800.0).apply(lambda value: clamp(float(value), 0.05, 0.88))
    latest["hit_rate_l10"] = (0.50 + latest["edge_pct"] * 1.00 + latest["social_score"].clip(0, 100) / 2200.0).apply(lambda value: clamp(float(value), 0.05, 0.86))
    latest["confidence"] = (latest["edge_pct"].abs() * 4.0 + latest["book_count"] / 18.0 + latest["social_score"].clip(0, 100) / 700.0).apply(lambda value: clamp(float(value), 0.05, 0.90))
    latest["variance_flag"] = latest.apply(lambda row: "High variance alt - use small stake" if is_longshot_alt(row) else ("Alt line" if str(row.get("market", "")).endswith("_alternate") else "Main/reasonable line"), axis=1)
    latest["realism_score"] = latest.apply(calculate_realism_score, axis=1)
    latest["reasonable_line"] = latest.apply(is_reasonable_line, axis=1)
    latest["pf_score"] = latest.apply(calculate_pf_score, axis=1)
    latest["positive_ev"] = latest["edge_pct"] > STRONG_EDGE_MIN
    latest["strong_pick"] = (
        (latest["pf_score"] >= STRONG_PF_MIN)
        & latest["positive_ev"]
        & (latest["confidence"] >= STRONG_CONFIDENCE_MIN)
        & (latest["realism_score"] >= MIN_REALISM_SCORE)
        & latest["reasonable_line"]
        & latest["sharp_confirmed"]
    )
    latest["why"] = latest.apply(build_why_explanation, axis=1)
    latest["best_available"] = latest.groupby(["game_id", "market", "player_name", "prop_side"])["edge_pct"].transform("max") == latest["edge_pct"]
    latest = ensure_columns(latest, PROP_COLUMNS, {"sharp_books": "", "why": "No explanation available.", "pf_score": 0, "edge_pct": 0.0, "display_edge_pct": 0.0, "confidence": 0.0, "realism_score": 0})
    return latest.sort_values(["strong_pick", "pf_score", "realism_score", "edge_pct"], ascending=[False, False, False, False])


def is_reasonable_line(row: pd.Series) -> bool:
    line_delta = safe_float(row.get("line_delta"))
    if line_delta is None or pd.isna(line_delta):
        return False
    if line_delta > MAX_LINE_DELTA:
        return False
    market = str(row.get("market") or "")
    confidence = safe_float(row.get("confidence")) or 0.0
    edge = safe_float(row.get("edge_pct")) or 0.0
    if not market.endswith("_alternate"):
        return line_delta <= MAIN_LINE_MAX_DELTA
    if line_delta <= REASONABLE_ALT_MAX_DELTA:
        return True
    return edge >= LONGSHOT_EDGE_MIN and confidence >= LONGSHOT_CONFIDENCE_MIN and not is_longshot_alt(row)


def _social_strength(props: pd.DataFrame, social_df: pd.DataFrame | None) -> pd.DataFrame:
    props = ensure_columns(props, ["player_name", "market"])
    keys = props[["player_name", "market"]].drop_duplicates().copy()
    if keys.empty:
        return pd.DataFrame(columns=["player_name", "market", "social_score", "social_mentions"])
    if social_df is None or social_df.empty:
        keys["social_score"] = 0.0
        keys["social_mentions"] = 0
        return keys
    social_df = ensure_columns(social_df, ["text", "engagement_score"])
    social = social_df.copy()
    social["text_norm"] = social["text"].astype(str).str.lower()
    rows = []
    for _, key in keys.iterrows():
        player = str(key["player_name"]).lower()
        words = str(key["market"]).replace("_", " ").lower().split()
        mask = social["text_norm"].str.contains(player, na=False, regex=False)
        for word in words:
            if len(word) >= 4:
                mask = mask | social["text_norm"].str.contains(word, na=False, regex=False)
        matches = social[mask]
        score = float(pd.to_numeric(matches["engagement_score"], errors="coerce").fillna(0).sum()) if not matches.empty else 0.0
        rows.append({"player_name": key["player_name"], "market": key["market"], "social_score": min(score, 100.0), "social_mentions": len(matches)})
    return pd.DataFrame(rows)


def calculate_pf_score(row: pd.Series) -> int:
    edge = max(0.0, safe_float(row.get("edge_pct")) or 0.0)
    confidence = safe_float(row.get("confidence")) or 0.0
    hit_rate = safe_float(row.get("hit_rate_l10")) or 0.50
    realism = safe_float(row.get("realism_score")) or 0.0
    sharp_bonus = 6 if pd.notna(row.get("sharp_prob")) else 0
    injury_penalty = 18 if bool(row.get("injury_flag")) else 0
    bad_line_penalty = 24 if not bool(row.get("reasonable_line")) else 0
    score = edge * 410 + confidence * 24 + max(0, hit_rate - 0.5) * 34 + realism * 0.32 + sharp_bonus - injury_penalty - bad_line_penalty
    return int(round(clamp(score, 0, 100)))


def build_why_explanation(row: pd.Series) -> str:
    edge = safe_float(row.get("display_edge_pct")) or safe_float(row.get("edge_pct")) or 0.0
    confidence = safe_float(row.get("confidence")) or 0.0
    pf_score = int(safe_float(row.get("pf_score")) or 0)
    realism = int(safe_float(row.get("realism_score")) or 0)
    line_delta = safe_float(row.get("line_delta")) or 0.0
    sharp_note = "sharp comp unavailable"
    if pd.notna(row.get("sharp_prob")):
        sharp_edge = safe_float(row.get("sharp_edge_vs_pinnacle")) or 0.0
        sharp_note = f"sharp books ({row.get('sharp_books', 'available')}) edge {sharp_edge:+.1%}"
    weather_note = str(row.get("weather_note") or "Weather unavailable")
    injury_note = str(row.get("injury_note") or "No stored injury flag")
    trend_note = str(row.get("trend_note") or "No strong trend signal")
    movement_note = str(row.get("line_movement_signal") or "No line movement signal")
    risk_note = str(row.get("variance_flag") or "Main/reasonable line")
    if realism < MIN_REALISM_SCORE:
        risk_note = f"Filtered risk: {risk_note}"
    return (
        f"Adjusted edge {edge:.1%}; PF {pf_score}/100; Realism {realism}/100; confidence {confidence:.1%}. "
        f"Line distance {line_delta:.1f}; L5/L10 estimate {row.get('hit_rate_l5', 0.5):.1%}/{row.get('hit_rate_l10', 0.5):.1%}. "
        f"Context: {weather_note}; {injury_note}; {trend_note}; {movement_note}; {sharp_note}. "
        f"Risk note: {risk_note}."
    )


def strong_props(props_df: pd.DataFrame, pf_min: int = STRONG_PF_MIN, edge_min: float = STRONG_EDGE_MIN, prop_type: str | list[str] | None = None) -> pd.DataFrame:
    props_df = ensure_columns(props_df, PROP_COLUMNS, {"pf_score": 0, "edge_pct": 0.0, "display_edge_pct": 0.0, "confidence": 0.0, "realism_score": 0, "positive_ev": False, "reasonable_line": False, "sharp_confirmed": True, "strong_pick": False})
    props_df = get_props_by_type(props_df, prop_type)
    if props_df.empty:
        return empty_props_frame()
    return props_df[
        (props_df["pf_score"] >= pf_min)
        & (props_df["edge_pct"] > edge_min)
        & (props_df["edge_pct"] <= MAX_DISPLAY_EDGE)
        & (props_df["confidence"] >= STRONG_CONFIDENCE_MIN)
        & (props_df["realism_score"] >= MIN_REALISM_SCORE)
        & (props_df["reasonable_line"] == True)
        & (props_df["sharp_confirmed"].fillna(True) == True)
    ].copy()


def build_power_rankings(games_df: pd.DataFrame, projections_df: pd.DataFrame) -> pd.DataFrame:
    games_df = ensure_columns(games_df, ["completed", "home_score", "away_score", "league", "home_team", "away_team"])
    if games_df.empty:
        return pd.DataFrame(columns=["league", "team", "games", "wins", "point_diff", "win_rate", "avg_point_diff", "power_rating"])
    base = games_df.copy()
    base["home_score"] = pd.to_numeric(base["home_score"], errors="coerce")
    base["away_score"] = pd.to_numeric(base["away_score"], errors="coerce")
    completed = base[base["completed"] == 1]
    rows = []
    for _, row in completed.iterrows():
        home_diff = (row["home_score"] or 0) - (row["away_score"] or 0)
        rows.append({"league": row["league"], "team": row["home_team"], "games": 1, "wins": int(home_diff > 0), "point_diff": home_diff})
        rows.append({"league": row["league"], "team": row["away_team"], "games": 1, "wins": int(home_diff < 0), "point_diff": -home_diff})
    if not rows:
        return pd.DataFrame(columns=["league", "team", "games", "wins", "point_diff", "win_rate", "avg_point_diff", "power_rating"])
    ratings = pd.DataFrame(rows).groupby(["league", "team"], as_index=False).sum()
    ratings["win_rate"] = ratings["wins"] / ratings["games"].clip(lower=1)
    ratings["avg_point_diff"] = ratings["point_diff"] / ratings["games"].clip(lower=1)
    ratings["power_rating"] = (ratings["win_rate"] * 70) + (ratings["avg_point_diff"] * 3)
    return ratings.sort_values(["league", "power_rating"], ascending=[True, False])


def combined_decimal_odds(american_prices: list[float]) -> float:
    total = 1.0
    for price in american_prices:
        decimal = american_to_decimal(price)
        if decimal is not None:
            total *= decimal
    return total


def parlay_probability(legs: pd.DataFrame) -> float:
    legs = ensure_columns(legs, ["projected_probability", "game_id", "realism_score"])
    if legs.empty:
        return 0.0
    probability = 1.0
    for value in legs["projected_probability"].fillna(0.50):
        probability *= float(value)
    correlation_penalty = max(0, len(legs) - legs["game_id"].nunique()) * 0.05
    realism_penalty = max(0.0, 72 - float(legs["realism_score"].mean())) / 500.0
    return clamp(probability - correlation_penalty - realism_penalty, 0.01, 0.90)


def parlay_summary(legs: pd.DataFrame, stake: float = 10.0) -> dict[str, Any]:
    legs = ensure_columns(legs, ["price_american", "projected_probability", "game_id", "pf_score", "realism_score"])
    if legs.empty:
        return {"legs": 0, "decimal_odds": 0.0, "american_odds": None, "probability": 0.0, "payout": 0.0, "profit": 0.0, "ev": 0.0, "risk": "N/A", "why": "No legs selected."}
    decimal_odds = combined_decimal_odds(legs["price_american"].fillna(-110).tolist())
    probability = parlay_probability(legs)
    payout = stake * decimal_odds
    profit = payout - stake
    ev = probability * profit - (1 - probability) * stake
    avg_pf = float(legs["pf_score"].mean()) if "pf_score" in legs else 50.0
    avg_realism = float(legs["realism_score"].mean()) if "realism_score" in legs else 50.0
    risk = "Low" if len(legs) <= 2 and avg_pf >= 80 and avg_realism >= 76 else "Medium" if len(legs) <= 4 and avg_pf >= 72 and avg_realism >= 68 else "High"
    why = f"{len(legs)} filtered legs, projected hit probability {probability:.1%}, average PF {avg_pf:.0f}/100, realism {avg_realism:.0f}/100, EV {ev:+.2f} on ${stake:.2f}."
    return {"legs": len(legs), "decimal_odds": decimal_odds, "american_odds": decimal_to_american(decimal_odds), "probability": probability, "payout": payout, "profit": profit, "ev": ev, "risk": risk, "why": why}


def suggest_parlays(props_df: pd.DataFrame, min_legs: int = 2, max_legs: int = 6, stake: float = 10.0, limit: int = 12, prop_type: str | list[str] | None = None) -> pd.DataFrame:
    pool = strong_props(props_df, pf_min=STRONG_PF_MIN, edge_min=STRONG_EDGE_MIN, prop_type=prop_type).head(18)
    if len(pool) < min_legs:
        return pd.DataFrame(columns=["legs", "players", "markets", "combined_american", "projected_probability", "ev", "risk", "why", "leg_ids"])
    suggestions = []
    for size in range(min_legs, min(max_legs, len(pool)) + 1):
        for combo in itertools.combinations(pool.index.tolist(), size):
            legs = pool.loc[list(combo)]
            if legs["player_name"].nunique() < len(legs) and size > 2:
                continue
            if float(legs["realism_score"].mean()) < MIN_REALISM_SCORE:
                continue
            summary = parlay_summary(legs, stake=stake)
            if summary["ev"] <= 0:
                continue
            suggestions.append({
                "legs": size,
                "players": " | ".join(legs["player_name"].astype(str).tolist()),
                "markets": " | ".join((legs["prop_type"] + ": " + legs["market_display"] + " " + legs["prop_side"]).astype(str).tolist()),
                "combined_american": summary["american_odds"],
                "projected_probability": summary["probability"],
                "ev": summary["ev"],
                "risk": summary["risk"],
                "why": summary["why"],
                "leg_ids": list(combo),
            })
            if len(suggestions) > 300:
                break
        if len(suggestions) > 300:
            break
    if not suggestions:
        return pd.DataFrame(columns=["legs", "players", "markets", "combined_american", "projected_probability", "ev", "risk", "why", "leg_ids"])
    return pd.DataFrame(suggestions).sort_values(["ev", "projected_probability"], ascending=[False, False]).head(limit)


if __name__ == "__main__":
    print("Models ready")
