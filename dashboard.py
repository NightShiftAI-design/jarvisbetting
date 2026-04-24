"""Sport-first, conservative Streamlit dashboard for Jarvis_Betting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from backtest import Backtester
from config import LOG_FILE
from data_ingestion import (
    SportsDataIngestionService,
    empty_games_frame,
    empty_injuries_frame,
    empty_line_moves_frame,
    empty_odds_frame,
    empty_projections_frame,
    empty_social_trends_frame,
    ensure_frame_schema,
    GAME_COLUMNS,
    INJURY_COLUMNS,
    LINE_MOVE_COLUMNS,
    ODDS_COLUMNS,
    PROJECTION_COLUMNS,
    SOCIAL_TREND_COLUMNS,
)
from database import Game, Injury, LineMovement, OddsHistory, Projection, get_session, init_db
from models import (
    MAX_DISPLAY_EDGE,
    MIN_REALISM_SCORE,
    PROP_COLUMNS,
    PROP_TYPE_ORDER,
    STRONG_PF_MIN,
    build_player_props_frame,
    empty_props_frame,
    get_available_prop_types,
    get_props_by_type,
    parlay_summary,
    strong_props,
    suggest_parlays,
)
from utils import convert_series_to_est, display_bookmaker, est_now, format_est, is_sharp_book, safe_json_loads

PAGE_TITLE = "Jarvis_Betting"
AUTO_REFRESH_MS = 60 * 60 * 1000
BG = "#080B10"
LINE = "rgba(255,255,255,0.08)"
GREEN = "#28D17C"
RED = "#FF5563"
AMBER = "#F6C453"
CYAN = "#45C7F4"
MUTED = "#98A2B3"

SPORT_TABS = [
    {"label": "MLB", "league": "mlb"},
    {"label": "NBA", "league": "nba"},
    {"label": "NFL", "league": "nfl"},
    {"label": "Soccer", "league": "eng.1"},
    {"label": "NHL", "league": "nhl"},
    {"label": "CFB", "league": "college-football"},
    {"label": "WNBA", "league": "wnba"},
]

GAME_ODDS_COLUMNS = ["bookmaker", "market", "selection_name", "point", "price_american", "implied_probability", "best_line", "sharp_book", "pulled_at_est"]
PROP_DISPLAY_COLUMNS = ["pf_score", "realism_score", "prop_type", "player_name", "market_display", "prop_side", "point", "projected_line", "line_delta", "bookmaker", "price_american", "display_edge_pct", "confidence", "hit_rate_l5", "hit_rate_l10", "variance_flag", "why"]
SHARP_COLUMNS = ["league", "matchup", "sportsbook", "market", "selection_name", "opening_price", "current_price", "opening_point", "current_point", "movement_abs", "sharp_score", "why"]


@st.cache_data(ttl=3600, show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    init_db()
    try:
        with get_session() as session:
            games = pd.read_sql(session.query(Game).statement, session.bind)
            projections = pd.read_sql(session.query(Projection).statement, session.bind)
            odds = pd.read_sql(session.query(OddsHistory).statement, session.bind)
            injuries = pd.read_sql(session.query(Injury).statement, session.bind)
            moves = pd.read_sql(session.query(LineMovement).statement, session.bind)
    except Exception:
        return empty_games_frame(), empty_projections_frame(), empty_odds_frame(), empty_injuries_frame(), empty_line_moves_frame()
    return (
        ensure_frame_schema(games, GAME_COLUMNS),
        ensure_frame_schema(projections, PROJECTION_COLUMNS),
        ensure_frame_schema(odds, ODDS_COLUMNS),
        ensure_frame_schema(injuries, INJURY_COLUMNS),
        ensure_frame_schema(moves, LINE_MOVE_COLUMNS),
    )


@st.cache_data(ttl=3600, show_spinner=False)
def load_social_trends() -> pd.DataFrame:
    try:
        return ensure_frame_schema(SportsDataIngestionService().fetch_social_trends(), SOCIAL_TREND_COLUMNS)
    except Exception:
        return empty_social_trends_frame()


def safe_table(frame: pd.DataFrame, columns: list[str], defaults: dict[str, Any] | None = None) -> pd.DataFrame:
    defaults = defaults or {}
    output = frame.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = defaults.get(column, pd.NA)
    return output.reindex(columns=columns)


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        :root {{ --bg:{BG}; --line:{LINE}; --green:{GREEN}; --red:{RED}; --amber:{AMBER}; --cyan:{CYAN}; --muted:{MUTED}; }}
        .stApp {{ background: linear-gradient(180deg, #090D13 0%, #080B10 58%, #05070A 100%); color: #F8FAFC; }}
        [data-testid="stHeader"] {{ background: rgba(8,11,16,0.72); backdrop-filter: blur(14px); border-bottom: 1px solid var(--line); }}
        [data-testid="stSidebar"] {{ background: #0B0F15; border-right: 1px solid var(--line); }}
        .block-container {{ padding-top: 1.1rem; max-width: 1580px; }}
        h1,h2,h3,h4,p,div,span,label {{ color: #F8FAFC; letter-spacing: 0; }}
        .topbar {{ border: 1px solid var(--line); background: linear-gradient(135deg, rgba(20,25,34,0.98), rgba(11,15,21,0.98)); padding: 16px 18px; border-radius: 8px; box-shadow: 0 20px 48px rgba(0,0,0,0.32); margin-bottom: 12px; }}
        .title {{ font-size: 30px; font-weight: 800; line-height: 1.1; }}
        .subtle {{ color: var(--muted); font-size: 13px; }}
        .pill {{ display:inline-block; padding: 4px 8px; border: 1px solid var(--line); border-radius: 999px; color: #D7DEE8; background: rgba(255,255,255,0.04); font-size: 12px; margin-right: 6px; margin-top: 8px; }}
        .kpi {{ border:1px solid var(--line); background: linear-gradient(180deg, rgba(20,25,34,0.98), rgba(16,20,27,0.98)); padding: 13px; border-radius: 8px; min-height: 106px; box-shadow: 0 12px 30px rgba(0,0,0,0.24); }}
        .kpi-label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }}
        .kpi-value {{ font-size: 26px; font-weight: 800; margin-top: 7px; }}
        .kpi-foot {{ color: var(--muted); font-size: 12px; margin-top: 6px; }}
        .game-head {{ border:1px solid var(--line); background: rgba(16,20,27,0.88); border-radius: 8px; padding: 12px; margin: 8px 0; }}
        .teams {{ display:flex; align-items:center; gap:10px; font-size: 18px; font-weight: 750; }}
        .logo {{ width: 30px; height:30px; border-radius: 50%; background: rgba(69,199,244,0.14); border:1px solid rgba(69,199,244,0.25); display:inline-flex; align-items:center; justify-content:center; font-size: 12px; font-weight: 800; overflow:hidden; }}
        .logo img {{ width:100%; height:100%; object-fit:contain; }}
        .why {{ border-left: 3px solid var(--green); background: rgba(40,209,124,0.07); padding: 10px 12px; border-radius: 6px; color: #DDF9E9; font-size: 13px; margin: 7px 0; }}
        .empty {{ border: 1px dashed rgba(255,255,255,0.18); background: rgba(255,255,255,0.03); color: var(--muted); padding: 14px; border-radius: 8px; }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid var(--line); flex-wrap: wrap; }}
        .stTabs [data-baseweb="tab"] {{ background: rgba(255,255,255,0.035); border: 1px solid var(--line); border-bottom: none; border-radius: 8px 8px 0 0; padding: 10px 14px; }}
        .stTabs [aria-selected="true"] {{ background: linear-gradient(180deg, rgba(69,199,244,0.16), rgba(40,209,124,0.08)); border-color: rgba(69,199,244,0.42); }}
        div[data-testid="stMetric"] {{ background: rgba(255,255,255,0.03); border: 1px solid var(--line); border-radius: 8px; padding: 10px; }}
        div[data-testid="stExpander"] {{ border: 1px solid var(--line); border-radius: 8px; background: rgba(16,20,27,0.75); }}
        @media (max-width: 760px) {{ .title {{ font-size: 22px; }} .kpi-value {{ font-size: 22px; }} .teams {{ font-size: 15px; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_hourly_refresh() -> None:
    components.html(f"<script>setTimeout(function(){{window.parent.location.reload();}}, {AUTO_REFRESH_MS});</script>", height=0, width=0)


def kpi(label: str, value: Any, foot: str, color: str) -> str:
    return f"<div class='kpi'><div class='kpi-label'>{label}</div><div class='kpi-value' style='color:{color}'>{value}</div><div class='kpi-foot'>{foot}</div></div>"


def preprocess(games: pd.DataFrame, projections: pd.DataFrame, odds: pd.DataFrame, injuries: pd.DataFrame, moves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    games = ensure_frame_schema(games, GAME_COLUMNS)
    projections = ensure_frame_schema(projections, PROJECTION_COLUMNS)
    odds = ensure_frame_schema(odds, ODDS_COLUMNS)
    injuries = ensure_frame_schema(injuries, INJURY_COLUMNS)
    moves = ensure_frame_schema(moves, LINE_MOVE_COLUMNS)
    games["commence_time_est"] = convert_series_to_est(games["commence_time"]) if not games.empty else pd.Series(dtype="datetime64[ns, America/New_York]")
    projections["created_at_est"] = convert_series_to_est(projections["created_at"]) if not projections.empty else pd.Series(dtype="datetime64[ns, America/New_York]")
    odds["pulled_at_est"] = convert_series_to_est(odds["pulled_at"]) if not odds.empty else pd.Series(dtype="datetime64[ns, America/New_York]")
    injuries["reported_at_est"] = convert_series_to_est(injuries["reported_at"]) if not injuries.empty else pd.Series(dtype="datetime64[ns, America/New_York]")
    moves["detected_at_est"] = convert_series_to_est(moves["detected_at"]) if not moves.empty else pd.Series(dtype="datetime64[ns, America/New_York]")
    return games, projections, odds, injuries, moves


def sport_frames(league: str, games: pd.DataFrame, odds: pd.DataFrame, injuries: pd.DataFrame, moves: pd.DataFrame, props: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    games_s = games[games["league"].astype(str).eq(league)].copy() if not games.empty else empty_games_frame()
    odds_s = odds[odds["league"].astype(str).eq(league)].copy() if not odds.empty else empty_odds_frame()
    injuries_s = injuries[injuries["league"].astype(str).eq(league)].copy() if not injuries.empty else empty_injuries_frame()
    moves_s = moves[moves["league"].astype(str).eq(league)].copy() if not moves.empty else empty_line_moves_frame()
    props_s = props[props["league"].astype(str).eq(league)].copy() if not props.empty else empty_props_frame()
    return games_s, odds_s, injuries_s, moves_s, props_s


def filter_prop_type(props: pd.DataFrame, selected_prop_type: str) -> pd.DataFrame:
    return get_props_by_type(props, selected_prop_type)


def grouped_sections(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    frame = safe_table(frame, PROP_COLUMNS)
    if frame.empty:
        return []
    sections = []
    for prop_type in PROP_TYPE_ORDER:
        if prop_type == "All":
            continue
        group = frame[frame["prop_type"] == prop_type].copy()
        if not group.empty:
            sections.append((prop_type, group.sort_values(["pf_score", "realism_score", "display_edge_pct"], ascending=[False, False, False])))
    return sections


def date_options(games: pd.DataFrame) -> list[pd.Timestamp]:
    if games.empty or "commence_time_est" not in games:
        return []
    dates = games["commence_time_est"].dropna().dt.date.drop_duplicates().sort_values().tolist()
    return [pd.Timestamp(date) for date in dates]


def filter_games_by_date(games: pd.DataFrame, selected_date: Any) -> pd.DataFrame:
    if games.empty or selected_date is None or "commence_time_est" not in games:
        return games
    selected = pd.Timestamp(selected_date).date()
    filtered = games[games["commence_time_est"].dt.date == selected].copy()
    return filtered if not filtered.empty else games.sort_values("commence_time_est").head(12).copy()


def extract_team_logo(game: pd.Series, home_away: str) -> str | None:
    raw = safe_json_loads(game.get("raw_json"))
    competitions = raw.get("competitions") or []
    if not competitions:
        return None
    for competitor in (competitions[0].get("competitors") or []):
        if competitor.get("homeAway") != home_away:
            continue
        logos = ((competitor.get("team") or {}).get("logos") or [])
        if logos:
            return logos[0].get("href")
    return None


def logo_html(name: str, url: str | None) -> str:
    initials = "".join(part[:1] for part in str(name or "TBD").split()[:2]).upper() or "T"
    if url:
        return f"<span class='logo'><img src='{url}' alt='{initials}'></span>"
    return f"<span class='logo'>{initials}</span>"


def game_title(game: pd.Series) -> str:
    away = str(game.get("away_team") or "TBD")
    home = str(game.get("home_team") or "TBD")
    start = format_est(game.get("commence_time_est"), "%a %b %d, %I:%M %p %Z")
    return f"{away} at {home} - {start}"


def render_game_header(game: pd.Series) -> None:
    away = str(game.get("away_team") or "TBD")
    home = str(game.get("home_team") or "TBD")
    start = format_est(game.get("commence_time_est"), "%a %b %d, %I:%M %p %Z")
    status = str(game.get("status") or "Scheduled")
    st.markdown(
        f"""
        <div class='game-head'>
            <div class='teams'>{logo_html(away, extract_team_logo(game, 'away'))}<span>{away}</span><span class='subtle'>at</span>{logo_html(home, extract_team_logo(game, 'home'))}<span>{home}</span></div>
            <div class='subtle'>{start} | {status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def game_odds_frame(odds: pd.DataFrame, game_id: Any) -> pd.DataFrame:
    odds = ensure_frame_schema(odds, [*ODDS_COLUMNS, "pulled_at_est"])
    frame = odds[(odds["game_id"] == game_id) & (odds["market"].astype(str).isin(["h2h", "spreads", "totals"]))].copy()
    if frame.empty:
        return safe_table(pd.DataFrame(), GAME_ODDS_COLUMNS)
    frame["sharp_book"] = frame["bookmaker"].apply(is_sharp_book)
    frame["bookmaker"] = frame["bookmaker"].apply(display_bookmaker)
    frame["price_american"] = pd.to_numeric(frame["price_american"], errors="coerce")
    frame["best_line"] = frame.groupby(["market", "selection_name"])["price_american"].transform("max") == frame["price_american"]
    frame["pulled_at_est"] = frame["pulled_at_est"].dt.strftime("%b %d %I:%M %p %Z") if "pulled_at_est" in frame else ""
    return safe_table(frame.sort_values(["market", "selection_name", "best_line"], ascending=[True, True, False]), GAME_ODDS_COLUMNS)


def style_best_lines(frame: pd.DataFrame) -> Any:
    if frame.empty:
        return frame
    def row_style(row: pd.Series) -> list[str]:
        color = "background-color: rgba(40,209,124,0.16); color: #E9FFF3;" if bool(row.get("best_line")) else ""
        return [color for _ in row]
    return frame.style.apply(row_style, axis=1)


def display_props(frame: pd.DataFrame) -> pd.DataFrame:
    frame = safe_table(frame, PROP_DISPLAY_COLUMNS, {"realism_score": 0, "display_edge_pct": 0.0})
    if not frame.empty:
        frame["bookmaker"] = frame["bookmaker"].apply(display_bookmaker)
    return frame


def game_props_frame(props: pd.DataFrame, game_id: Any, selected_prop_type: str) -> pd.DataFrame:
    props = safe_table(props, PROP_COLUMNS, {"pf_score": 0, "edge_pct": 0.0, "display_edge_pct": 0.0, "confidence": 0.0, "realism_score": 0})
    frame = filter_prop_type(props, selected_prop_type)
    frame = frame[frame["game_id"] == game_id].copy()
    if frame.empty:
        return safe_table(pd.DataFrame(), PROP_DISPLAY_COLUMNS)
    return display_props(frame.sort_values(["strong_pick", "pf_score", "realism_score", "display_edge_pct"], ascending=[False, False, False, False]))


def best_bets_frame(props: pd.DataFrame, game_id: Any, selected_prop_type: str) -> pd.DataFrame:
    game_props = props[props["game_id"] == game_id].copy() if not props.empty else empty_props_frame()
    return display_props(strong_props(game_props, prop_type=selected_prop_type))


def sharp_edges_frame(moves: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    moves = ensure_frame_schema(moves, ["id", "game_id", "sport", "league", "sportsbook", "market", "selection_name", "opening_price", "current_price", "opening_point", "current_point", "movement_abs", "detected_at", "source", "raw_json", "created_at", "updated_at", "detected_at_est"])
    games = ensure_frame_schema(games, ["id", "league", "home_team", "away_team", "commence_time_est"])
    if moves.empty:
        return pd.DataFrame(columns=["league", "matchup", "sportsbook", "market", "selection_name", "opening_price", "current_price", "opening_point", "current_point", "movement_abs", "sharp_score", "why"])
    latest = moves.sort_values("detected_at_est").groupby(["game_id", "sportsbook", "market", "selection_name"], dropna=False).tail(1)
    merged = latest.merge(games[["id", "league", "home_team", "away_team", "commence_time_est"]], left_on="game_id", right_on="id", how="left", suffixes=("", "_game"))
    if "league_game" in merged:
        merged["league"] = merged["league"].fillna(merged["league_game"])
    merged["matchup"] = merged["away_team"].fillna("TBD") + " @ " + merged["home_team"].fillna("TBD")
    merged["sharp_score"] = pd.to_numeric(merged["movement_abs"], errors="coerce").fillna(0).clip(0, 100)
    merged["why"] = merged.apply(lambda row: f"Line moved {row.get('movement_abs', 0):.1f}; current {row.get('current_price')} from opener {row.get('opening_price')}; matchup {row.get('matchup')}; line movement only, not invented public split data.", axis=1)
    return safe_table(merged.sort_values("sharp_score", ascending=False), ["league", "matchup", "sportsbook", "market", "selection_name", "opening_price", "current_price", "opening_point", "current_point", "movement_abs", "sharp_score", "why"])


def render_why_box(row: pd.Series) -> None:
    st.markdown(f"<div class='why'><b>WHY</b><br>{row.get('why', 'No explanation available.')}</div>", unsafe_allow_html=True)


def render_empty(message: str) -> None:
    st.markdown(f"<div class='empty'>{message}</div>", unsafe_allow_html=True)


def render_grouped_best_bets(frame: pd.DataFrame) -> None:
    sections = grouped_sections(frame)
    if not sections:
        render_empty("No high-confidence +EV picks after the conservative realism filter.")
        return
    for prop_type, group in sections:
        st.markdown(f"#### {prop_type}")
        st.dataframe(display_props(group), use_container_width=True, hide_index=True)
        for _, row in group.head(3).iterrows():
            render_why_box(row)


def render_sport_tab(label: str, league: str, games: pd.DataFrame, odds: pd.DataFrame, injuries: pd.DataFrame, moves: pd.DataFrame, props: pd.DataFrame, stake: float, leg_count: int, global_prop_type: str) -> None:
    games_s, odds_s, injuries_s, moves_s, props_s = sport_frames(league, games, odds, injuries, moves, props)
    available_types = get_available_prop_types(props_s)
    default_type = global_prop_type if global_prop_type in available_types else "All"
    filter_cols = st.columns([1, 1, 1])
    dates = date_options(games_s)
    default_date = dates[0].date() if dates else est_now().date()
    selected_date = filter_cols[0].date_input(f"{label} date", value=default_date, key=f"date_{league}")
    selected_prop_type = filter_cols[1].selectbox("Prop Type", available_types, index=available_types.index(default_type), key=f"prop_type_{league}")
    parlay_prop_type = filter_cols[2].selectbox("Parlay Legs", available_types, index=available_types.index(default_type), key=f"parlay_prop_type_{league}")
    props_s = filter_prop_type(props_s, selected_prop_type)
    slate = filter_games_by_date(games_s, selected_date)
    strong_s = strong_props(props_s, prop_type="All")
    sharp_s = sharp_edges_frame(moves_s, games_s)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("Games", len(slate), "Today's/upcoming slate", CYAN), unsafe_allow_html=True)
    c2.markdown(kpi("Realistic EV Picks", len(strong_s), f"PF {STRONG_PF_MIN}+ and realism {MIN_REALISM_SCORE}+", GREEN), unsafe_allow_html=True)
    c3.markdown(kpi("Props", len(props_s), f"Filter: {selected_prop_type}", AMBER), unsafe_allow_html=True)
    c4.markdown(kpi("Max Edge", f"{MAX_DISPLAY_EDGE:.0%}", "Displayed cap", RED), unsafe_allow_html=True)

    if slate.empty:
        render_empty(f"No {label} games are stored yet. Use Refresh Data after installing dependencies, or ingest odds from the command line.")
    else:
        st.markdown("### Games")
        game_list = slate[["id", "away_team", "home_team", "commence_time_est", "status"]].copy()
        game_list["start_est"] = game_list["commence_time_est"].dt.strftime("%a %b %d, %I:%M %p %Z")
        st.dataframe(game_list.reindex(columns=["away_team", "home_team", "start_est", "status"]), use_container_width=True, hide_index=True)

    st.markdown("### Quick Picks")
    render_grouped_best_bets(strong_s)

    for _, game in slate.head(10).iterrows():
        with st.expander(game_title(game), expanded=False):
            render_game_header(game)
            game_id = game.get("id")
            live_tab, props_tab, research_tab, best_tab = st.tabs(["Live / Game Odds", "Player Props", "Research / Trends", "Best Bets"])

            with live_tab:
                game_odds = game_odds_frame(odds_s, game_id)
                if game_odds.empty:
                    render_empty("No live/game odds stored for this matchup yet.")
                else:
                    st.dataframe(style_best_lines(game_odds), use_container_width=True, hide_index=True)

            with props_tab:
                game_props = game_props_frame(props_s, game_id, "All")
                if game_props.empty:
                    render_empty("No props match the selected prop type for this game.")
                else:
                    st.dataframe(
                        game_props,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "pf_score": st.column_config.ProgressColumn("PF", min_value=0, max_value=100),
                            "realism_score": st.column_config.ProgressColumn("Realism", min_value=0, max_value=100),
                            "display_edge_pct": st.column_config.ProgressColumn("EV Edge", min_value=-0.1, max_value=MAX_DISPLAY_EDGE, format="%.3f"),
                            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f"),
                            "hit_rate_l5": st.column_config.ProgressColumn("L5", min_value=0, max_value=1, format="%.2f"),
                            "hit_rate_l10": st.column_config.ProgressColumn("L10", min_value=0, max_value=1, format="%.2f"),
                        },
                    )

            with research_tab:
                game_props_raw = props_s[props_s["game_id"] == game_id].copy() if not props_s.empty else empty_props_frame()
                if game_props_raw.empty:
                    render_empty("No trends available for the selected prop type.")
                else:
                    left, right = st.columns([1.15, 1])
                    with left:
                        fig = px.scatter(game_props_raw.head(200), x="display_edge_pct", y="realism_score", size="confidence", color="prop_type", hover_name="player_name", hover_data=["why"], title="Realism vs capped EV edge")
                        fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=390)
                        st.plotly_chart(fig, use_container_width=True)
                    with right:
                        research = game_props_raw.groupby("prop_type", dropna=False).agg(avg_pf=("pf_score", "mean"), avg_realism=("realism_score", "mean"), avg_edge=("display_edge_pct", "mean"), props=("player_name", "count")).reset_index().sort_values("avg_realism", ascending=False)
                        st.dataframe(research, use_container_width=True, hide_index=True)
                injury_game = injuries_s[(injuries_s["game_id"] == game_id) | (injuries_s["team"].isin([game.get("home_team"), game.get("away_team")]))].copy() if not injuries_s.empty else empty_injuries_frame()
                if not injury_game.empty:
                    st.markdown("#### Injury notes")
                    st.dataframe(safe_table(injury_game, ["team", "player_name", "status", "injury_type", "reason"]), use_container_width=True, hide_index=True)

            with best_tab:
                best = best_bets_frame(props_s, game_id, "All")
                render_grouped_best_bets(best)

    st.markdown("### Parlay Builder")
    parlay_pool = strong_props(get_props_by_type(props_s, parlay_prop_type), prop_type="All")
    if parlay_pool.empty:
        render_empty(f"No parlay legs meet the conservative filter for {parlay_prop_type}. Need PF {STRONG_PF_MIN}+, positive EV, realistic line distance, and realism {MIN_REALISM_SCORE}+.")
    else:
        suggestions = suggest_parlays(parlay_pool, min_legs=2, max_legs=leg_count, stake=stake, limit=10, prop_type="All")
        left, right = st.columns([1.15, 1])
        with left:
            if suggestions.empty:
                render_empty("No positive-EV parlay combinations found from the current conservative leg pool.")
            else:
                st.dataframe(suggestions.drop(columns=["leg_ids"], errors="ignore"), use_container_width=True, hide_index=True)
                selected = st.selectbox(f"Load {label} parlay", list(range(len(suggestions))), format_func=lambda idx: f"{suggestions.iloc[idx]['legs']} legs | EV {suggestions.iloc[idx]['ev']:+.2f} | {suggestions.iloc[idx]['risk']}", key=f"parlay_select_{league}")
                if st.button(f"Load {label} slip", use_container_width=True, key=f"load_slip_{league}"):
                    st.session_state[f"parlay_{league}"] = suggestions.iloc[selected]["leg_ids"]
                    st.rerun()
            options = parlay_pool.head(120).copy()
            options["label"] = options.apply(lambda row: f"{row['prop_type']} | {row['player_name']} | {row['market_display']} {row['prop_side']} {row.get('point', '')} | PF {int(row['pf_score'])} | Real {int(row['realism_score'])}", axis=1)
            chosen = st.multiselect("Manual legs", options.index.tolist(), default=st.session_state.get(f"parlay_{league}", []), format_func=lambda idx: options.loc[idx, "label"] if idx in options.index else str(idx), key=f"manual_{league}")
            st.session_state[f"parlay_{league}"] = chosen
        with right:
            leg_ids = [idx for idx in st.session_state.get(f"parlay_{league}", []) if idx in parlay_pool.index]
            legs = parlay_pool.loc[leg_ids].copy() if leg_ids else empty_props_frame()
            summary = parlay_summary(legs, stake=stake)
            s1, s2, s3 = st.columns(3)
            s1.metric("Legs", summary["legs"])
            s2.metric("Odds", summary["american_odds"] if summary["american_odds"] is not None else "N/A")
            s3.metric("EV", f"{summary['ev']:+.2f}")
            st.markdown(f"<div class='why'><b>WHY</b><br>{summary['why']}</div>", unsafe_allow_html=True)
            if not legs.empty:
                st.dataframe(display_props(legs), use_container_width=True, hide_index=True)

    st.markdown("### Sharp Edges")
    sharp_s = sharp_edges_frame(moves_s, games_s)
    if sharp_s.empty:
        render_empty("No line movement records yet. The table is schema-safe and will populate after movement is stored.")
    else:
        st.dataframe(sharp_s, use_container_width=True, hide_index=True)


def best_bets_frame(props: pd.DataFrame, game_id: Any, selected_prop_type: str) -> pd.DataFrame:
    game_props = props[props["game_id"] == game_id].copy() if not props.empty else empty_props_frame()
    return display_props(strong_props(game_props, prop_type=selected_prop_type))


def sharp_edges_frame(moves: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    moves = ensure_frame_schema(moves, ["id", "game_id", "sport", "league", "sportsbook", "market", "selection_name", "opening_price", "current_price", "opening_point", "current_point", "movement_abs", "detected_at", "source", "raw_json", "created_at", "updated_at", "detected_at_est"])
    games = ensure_frame_schema(games, ["id", "league", "home_team", "away_team", "commence_time_est"])
    if moves.empty:
        return pd.DataFrame(columns=["league", "matchup", "sportsbook", "market", "selection_name", "opening_price", "current_price", "opening_point", "current_point", "movement_abs", "sharp_score", "why"])
    latest = moves.sort_values("detected_at_est").groupby(["game_id", "sportsbook", "market", "selection_name"], dropna=False).tail(1)
    merged = latest.merge(games[["id", "league", "home_team", "away_team", "commence_time_est"]], left_on="game_id", right_on="id", how="left", suffixes=("", "_game"))
    if "league_game" in merged:
        merged["league"] = merged["league"].fillna(merged["league_game"])
    merged["matchup"] = merged["away_team"].fillna("TBD") + " @ " + merged["home_team"].fillna("TBD")
    merged["sharp_score"] = pd.to_numeric(merged["movement_abs"], errors="coerce").fillna(0).clip(0, 100)
    merged["why"] = merged.apply(lambda row: f"Line moved {row.get('movement_abs', 0):.1f}; current {row.get('current_price')} from opener {row.get('opening_price')}; matchup {row.get('matchup')}; line movement only, not invented public split data.", axis=1)
    return safe_table(merged.sort_values("sharp_score", ascending=False), ["league", "matchup", "sportsbook", "market", "selection_name", "opening_price", "current_price", "opening_point", "current_point", "movement_abs", "sharp_score", "why"])


def load_log_tail(limit: int = 180) -> str:
    path = Path(LOG_FILE)
    if not path.exists():
        return "No log file yet."
    return "".join(path.read_text(encoding="utf-8", errors="ignore").splitlines(True)[-limit:])


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state="expanded")
    inject_css()
    inject_hourly_refresh()

    st.sidebar.markdown("## Controls")
    st.sidebar.caption("Manual refresh or hourly auto-refresh only. All times are EST/ET.")
    global_prop_type = st.sidebar.selectbox("Global Prop Type", PROP_TYPE_ORDER, index=0)
    stake = st.sidebar.number_input("Parlay stake", min_value=1.0, max_value=10000.0, value=10.0, step=5.0)
    leg_count = st.sidebar.slider("Max parlay legs", 2, 6, 3, 1)
    st.sidebar.markdown(f"PF threshold: **{STRONG_PF_MIN}+**")
    st.sidebar.markdown(f"Max displayed edge: **{MAX_DISPLAY_EDGE:.0%}**")
    st.sidebar.markdown(f"Min realism: **{MIN_REALISM_SCORE}+**")
    if st.sidebar.button("Refresh Data", use_container_width=True):
        with st.spinner("Refreshing live odds and props..."):
            try:
                SportsDataIngestionService().ingest_all_primary_sports()
                st.sidebar.success("Refresh complete")
            except Exception as exc:
                st.sidebar.error(f"Refresh failed: {exc}")
        st.cache_data.clear()
        st.rerun()
    if st.sidebar.button("Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    games, projections, odds, injuries, moves = preprocess(*load_data())
    social = load_social_trends()
    props = build_player_props_frame(odds, games, injuries_df=injuries, social_df=social)
    props = safe_table(props, PROP_COLUMNS, {"pf_score": 0, "edge_pct": 0.0, "display_edge_pct": 0.0, "confidence": 0.0, "realism_score": 0})
    filtered_for_header = get_props_by_type(props, global_prop_type)

    st.markdown(
        f"""
        <div class='topbar'>
            <div class='title'>Jarvis_Betting</div>
            <div class='subtle'>Conservative prop intelligence with capped edges, realism scoring, prop-type filters, and aggressive best-bet filtering.</div>
            <span class='pill'>ET {format_est(est_now(), '%b %d %I:%M %p %Z')}</span>
            <span class='pill'>PF {STRONG_PF_MIN}+ only</span>
            <span class='pill'>Displayed edge capped at {MAX_DISPLAY_EDGE:.0%}</span>
            <span class='pill'>Strong picks: {len(strong_props(filtered_for_header))}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs([sport["label"] for sport in SPORT_TABS])
    for tab, sport in zip(tabs, SPORT_TABS):
        with tab:
            render_sport_tab(
                label=sport["label"],
                league=sport["league"],
                games=games,
                odds=odds,
                injuries=injuries,
                moves=moves,
                props=props,
                stake=stake,
                leg_count=leg_count,
                global_prop_type=global_prop_type,
            )

    with st.expander("Runtime logs"):
        st.code(load_log_tail(), language="log")
    with st.expander("Backtester"):
        try:
            back_df, summary = Backtester().run_moneyline_backtest()
            st.dataframe(pd.DataFrame([summary.__dict__]), use_container_width=True, hide_index=True)
            if not back_df.empty:
                fig = px.bar(back_df.head(100), x="selection", y="profit_units", color="won", title="Historical bet profit")
                fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=420)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.info(f"Backtest unavailable: {exc}")


if __name__ == "__main__":
    main()
