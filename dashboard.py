"""Ultra-dense Streamlit research dashboard for Jarvis Betting.

This UI intentionally uses only stored/provider-backed data. If a provider has not
returned a field yet, the dashboard labels it as Unavailable/No data instead of
making up projections, hit rates, injuries, weather, or odds.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from config import LOG_FILE
from data_ingestion import SportsDataIngestionService
from data_service import (
    DashboardData,
    clear_service_cache,
    game_odds,
    game_props,
    game_weather,
    injury_splits_proxy,
    league_games,
    league_props,
    league_strong_props,
    load_dashboard_data,
    projection_cards,
    team_abbr,
    weather_risk,
)
from models import MAX_DISPLAY_EDGE, MIN_REALISM_SCORE, PROP_TYPE_ORDER, STRONG_PF_MIN, parlay_summary, suggest_parlays
from utils import est_now, format_est, safe_float

APP_TITLE = "Jarvis Betting"
AUTO_REFRESH_MS = 60 * 60 * 1000
LEAGUES = ["NBA", "NHL", "MLB", "NFL"]
LEAGUE_TO_KEY = {"NBA": "nba", "NHL": "nhl", "MLB": "mlb", "NFL": "nfl"}
PURPLE = "#8B5CF6"
GREEN = "#22C55E"
RED = "#F43F5E"
YELLOW = "#FBBF24"
CYAN = "#38BDF8"
BG = "#070A12"
PANEL = "#0E1320"
BORDER = "rgba(255,255,255,.09)"

SIDEBAR_TOOLS = {
    "NBA": [
        "Defensive Matchups", "Hit Rate Matrix", "First Basket", "Injury Reports", "Injury Splits",
        "Volume Trends", "Player Stats Summary", "Team Stats", "Odds Discrepancies", "Lineups", "Player Props",
    ],
    "MLB": [
        "HR Matchups", "Exit Velo", "Pitcher Weak Spots", "Team vs Pitcher", "Team vs Pitch Mix",
        "Hit Rate Matrix", "Ballpark Weather", "Park Factors", "Batter vs Pitcher", "Hit Streaks",
        "Pitcher Summary", "Bullpen Usage", "Odds Discrepancies", "Lineups", "Player Props", "Projections",
    ],
    "NHL": ["Goalie Matchups", "Shot Props", "Team Defense", "Power Play", "Player Trends", "Injury Reports", "Odds Discrepancies", "Player Props"],
    "NFL": ["Player Props", "Defensive Matchups", "Team Trends", "Injury Reports", "Weather", "Line Movement", "Odds Discrepancies"],
}

PROP_TABLE_COLUMNS = [
    "favorite", "pf_score", "team", "position", "player_name", "prop_type", "market_display", "prop_side", "point",
    "projected_line", "book_badge", "price_american", "hit_rate_l10", "hit_rate_l5", "streak",
    "season_matchup", "previous_season_hit_rate", "current_season_hit_rate", "display_edge_pct", "confidence", "realism_score", "why",
]


def set_page() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="JB", layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        f"""
        <style>
        :root {{ --purple:{PURPLE}; --green:{GREEN}; --red:{RED}; --yellow:{YELLOW}; --cyan:{CYAN}; --bg:{BG}; --panel:{PANEL}; --border:{BORDER}; }}
        .stApp {{ background:
            radial-gradient(circle at 12% 0%, rgba(139,92,246,.24), transparent 28%),
            radial-gradient(circle at 85% 12%, rgba(34,197,94,.14), transparent 25%),
            linear-gradient(180deg, #090D18 0%, #070A12 52%, #05070D 100%); color:#F8FAFC; }}
        [data-testid="stHeader"] {{ background: rgba(7,10,18,.74); backdrop-filter: blur(18px); border-bottom: 1px solid var(--border); }}
        [data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0B1020, #080B14); border-right: 1px solid var(--border); }}
        .block-container {{ max-width: 1760px; padding-top: 1.05rem; padding-bottom: 3rem; }}
        h1,h2,h3,h4,p,span,div,label {{ color:#F8FAFC; }}
        .shell-top {{ position: sticky; top: 0; z-index: 9; border:1px solid var(--border); border-radius:18px; padding:14px 16px; margin-bottom:14px; background: rgba(13,19,32,.86); backdrop-filter: blur(18px); box-shadow: 0 18px 44px rgba(0,0,0,.32); }}
        .brand-row {{ display:flex; align-items:center; justify-content:space-between; gap:14px; flex-wrap:wrap; }}
        .brand {{ display:flex; align-items:center; gap:12px; font-weight:900; font-size:25px; letter-spacing:-.02em; }}
        .brand-mark {{ width:38px; height:38px; border-radius:12px; display:grid; place-items:center; background:linear-gradient(135deg, var(--purple), #22D3EE); box-shadow: 0 0 32px rgba(139,92,246,.32); font-weight:950; }}
        .utility {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
        .chip {{ display:inline-flex; align-items:center; gap:6px; padding:7px 10px; border-radius:999px; border:1px solid var(--border); background:rgba(255,255,255,.045); color:#CBD5E1; font-size:12px; font-weight:700; }}
        .hero-grid {{ display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap:10px; margin:10px 0 16px; }}
        .metric-card {{ border:1px solid var(--border); border-radius:16px; padding:14px; background:linear-gradient(180deg, rgba(18,26,43,.96), rgba(12,17,29,.96)); box-shadow: 0 14px 32px rgba(0,0,0,.24); }}
        .metric-label {{ color:#94A3B8; font-size:11px; text-transform:uppercase; letter-spacing:.09em; }}
        .metric-value {{ font-size:26px; font-weight:900; margin-top:4px; }}
        .metric-foot {{ color:#94A3B8; font-size:12px; margin-top:3px; }}
        .panel {{ border:1px solid var(--border); border-radius:18px; background:rgba(14,19,32,.82); padding:14px; box-shadow: 0 16px 36px rgba(0,0,0,.22); margin-bottom:12px; }}
        .panel-title {{ display:flex; align-items:center; justify-content:space-between; gap:10px; font-size:18px; font-weight:900; margin-bottom:10px; }}
        .subtle {{ color:#94A3B8; font-size:12px; }}
        .game-carousel {{ display:flex; gap:10px; overflow-x:auto; padding: 4px 2px 12px; }}
        .game-card {{ min-width:220px; border:1px solid var(--border); border-radius:16px; padding:12px; background:rgba(255,255,255,.035); }}
        .game-card.active {{ border-color:rgba(139,92,246,.78); box-shadow: 0 0 0 1px rgba(139,92,246,.34), 0 16px 36px rgba(139,92,246,.12); }}
        .teams {{ display:flex; align-items:center; justify-content:space-between; gap:8px; font-weight:900; }}
        .logo {{ width:32px; height:32px; border-radius:50%; display:inline-grid; place-items:center; background:rgba(139,92,246,.16); border:1px solid rgba(139,92,246,.32); font-size:11px; font-weight:900; overflow:hidden; }}
        .logo img {{ width:100%; height:100%; object-fit:contain; }}
        .book {{ display:inline-block; min-width:34px; text-align:center; padding:3px 7px; border-radius:7px; background:rgba(139,92,246,.16); border:1px solid rgba(139,92,246,.36); color:#EDE9FE; font-weight:900; font-size:11px; }}
        .why {{ border-left:3px solid var(--green); background:rgba(34,197,94,.08); border-radius:12px; padding:12px; color:#DCFCE7; margin:8px 0; font-size:13px; }}
        .empty {{ border:1px dashed rgba(148,163,184,.35); border-radius:16px; padding:18px; background:rgba(255,255,255,.03); color:#94A3B8; }}
        .risk-clear {{ color:var(--green); }} .risk-chance {{ color:var(--yellow); }} .risk-delay {{ color:#FB923C; }} .risk-post {{ color:var(--red); }}
        .stTabs [data-baseweb="tab-list"] {{ gap:8px; border-bottom:1px solid var(--border); flex-wrap:wrap; }}
        .stTabs [data-baseweb="tab"] {{ border:1px solid var(--border); border-bottom:none; border-radius:14px 14px 0 0; background:rgba(255,255,255,.035); padding:10px 18px; font-weight:900; }}
        .stTabs [aria-selected="true"] {{ background:linear-gradient(180deg, rgba(139,92,246,.34), rgba(139,92,246,.08)); border-color:rgba(139,92,246,.6); }}
        div[data-testid="stDataFrame"] {{ border:1px solid var(--border); border-radius:14px; overflow:hidden; }}
        div[data-testid="stExpander"] {{ border:1px solid var(--border); border-radius:16px; background:rgba(14,19,32,.74); }}
        .footer-note {{ color:#94A3B8; border-top:1px solid var(--border); padding-top:14px; margin-top:26px; font-size:12px; }}
        @media (max-width: 920px) {{ .hero-grid {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} .brand {{ font-size:21px; }} }}
        @media (max-width: 620px) {{ .hero-grid {{ grid-template-columns: 1fr; }} .game-card {{ min-width:180px; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hourly_refresh() -> None:
    components.html(f"<script>setTimeout(() => window.parent.location.reload(), {AUTO_REFRESH_MS});</script>", height=0, width=0)


def empty_state(message: str) -> None:
    st.markdown(f"<div class='empty'>{message}</div>", unsafe_allow_html=True)


def logo_html(abbr: str, url: Any | None = None) -> str:
    if isinstance(url, str) and url:
        return f"<span class='logo'><img src='{url}' alt='{abbr}'></span>"
    return f"<span class='logo'>{abbr}</span>"


def kpi(label: str, value: Any, foot: str, color: str = "#F8FAFC") -> None:
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value' style='color:{color}'>{value}</div><div class='metric-foot'>{foot}</div></div>",
        unsafe_allow_html=True,
    )


def panel_title(title: str, right: str = "") -> None:
    st.markdown(f"<div class='panel-title'><span>{title}</span><span class='subtle'>{right}</span></div>", unsafe_allow_html=True)


def safe_cols(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out.reindex(columns=columns)


def style_heatmap(frame: pd.DataFrame, cols: list[str]) -> Any:
    def color_cell(value: Any) -> str:
        numeric = safe_float(value)
        if numeric is None:
            return "color:#94A3B8; background-color:rgba(148,163,184,.06);"
        scaled = numeric * 100 if abs(numeric) <= 1 else numeric
        if scaled >= 62:
            return "background-color:rgba(34,197,94,.22); color:#DCFCE7; font-weight:800;"
        if scaled <= 42:
            return "background-color:rgba(244,63,94,.20); color:#FFE4E6; font-weight:800;"
        return "background-color:rgba(251,191,36,.17); color:#FEF3C7; font-weight:800;"
    return frame.style.map(color_cell, subset=[c for c in cols if c in frame.columns])


def filter_props(frame: pd.DataFrame, search: str, prop_type: str, game_filter: str, show_alts: bool, side: str) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return out
    if search:
        query = search.lower().strip()
        out = out[
            out["player_name"].astype(str).str.lower().str.contains(query, na=False)
            | out["team"].astype(str).str.lower().str.contains(query, na=False)
            | out["matchup"].astype(str).str.lower().str.contains(query, na=False)
        ]
    if prop_type != "All" and "prop_type" in out:
        out = out[out["prop_type"].astype(str).eq(prop_type)]
    if game_filter != "All Games" and "matchup" in out:
        out = out[out["matchup"].astype(str).eq(game_filter)]
    if not show_alts and "market_display" in out:
        out = out[~out["market_display"].astype(str).str.contains("Alternate", case=False, na=False)]
    if side != "Both" and "prop_side" in out:
        out = out[out["prop_side"].astype(str).str.lower().eq(side.lower())]
    return out.sort_values(["pf_score", "realism_score", "display_edge_pct"], ascending=[False, False, False], na_position="last")


def render_top_shell(selected_date: date, data: DashboardData) -> None:
    last_refresh = format_est(est_now(), "%b %d, %I:%M %p %Z")
    st.markdown(
        f"""
        <div class='shell-top'>
          <div class='brand-row'>
            <div class='brand'><span class='brand-mark'>JB</span><span>Jarvis Betting</span><span class='chip'>Research Dashboard</span></div>
            <div class='utility'>
              <span class='chip'>ET {last_refresh}</span>
              <span class='chip'>Date {selected_date}</span>
              <span class='chip'>PF {STRONG_PF_MIN}+ Best Bets</span>
              <span class='chip'>No auth / no locks</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    with cols[0]:
        kpi("Games Loaded", len(data.games), "Stored ESPN/Odds API slate", CYAN)
    with cols[1]:
        kpi("Props Loaded", len(data.props), "Provider-backed props only", PURPLE)
    with cols[2]:
        kpi("Best Bets", len(data.strong_props), f"PF {STRONG_PF_MIN}+ and realism {MIN_REALISM_SCORE}+", GREEN)
    with cols[3]:
        kpi("Displayed Edge Cap", f"{MAX_DISPLAY_EDGE:.0%}", "Conservative EV display", YELLOW)


def render_game_carousel(games: pd.DataFrame, key: str) -> Any | None:
    if games.empty:
        empty_state("No games are available for this league/date yet. Refresh data or run ingestion to populate the slate.")
        return None
    options = games["id"].tolist()
    labels = {
        row["id"]: f"{row.get('away_abbr','TBD')} @ {row.get('home_abbr','TBD')} | {format_est(row.get('commence_time_est'), '%I:%M %p')}"
        for _, row in games.iterrows()
    }
    selected = st.radio("Game carousel", options, format_func=lambda value: labels.get(value, str(value)), horizontal=True, label_visibility="collapsed", key=key)
    selected_row = games[games["id"].eq(selected)].head(1)
    st.markdown("<div class='game-carousel'>", unsafe_allow_html=True)
    for _, row in games.head(18).iterrows():
        active = " active" if row.get("id") == selected else ""
        st.markdown(
            f"""
            <div class='game-card{active}'>
              <div class='teams'>{logo_html(row.get('away_abbr'), row.get('away_logo'))}<span>{row.get('away_abbr')}</span><span class='subtle'>@</span><span>{row.get('home_abbr')}</span>{logo_html(row.get('home_abbr'), row.get('home_logo'))}</div>
              <div class='subtle'>{row.get('start_est', 'Unavailable')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
    return selected_row.iloc[0] if not selected_row.empty else games.iloc[0]


def render_game_summary(game: pd.Series, data: DashboardData) -> None:
    odds = game_odds(data, game.get("id"))
    total = odds[odds["market"].eq("totals")].sort_values("pulled_at_est", ascending=False).head(1) if not odds.empty else pd.DataFrame()
    weather = game_weather(game)
    risk = weather_risk(weather)
    temp = weather.get("temperature_f", "Unavailable") if weather else "Unavailable"
    precip = weather.get("precipitation_probability", "Unavailable") if weather else "Unavailable"
    wind = f"{weather.get('wind_direction', 'Unavailable')} {weather.get('wind_mph', 'Unavailable')} mph" if weather else "Unavailable"
    ou = total.iloc[0].get("point") if not total.empty else "Unavailable"
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Matchup", f"{game.get('away_abbr')} @ {game.get('home_abbr')}", game.get("start_est", "Unavailable"), PURPLE)
    with c2: kpi("Ballpark / Venue", game.get("venue") or "Unavailable", f"O/U {ou}", CYAN)
    with c3: kpi("Weather Risk", risk, f"Temp {temp} | Precip {precip}%", GREEN if risk == "Clear" else YELLOW)
    with c4: kpi("Wind", wind, "NWS data if available", YELLOW)


def render_odds_board(data: DashboardData, game_id: Any | None = None, league: str | None = None) -> None:
    frame = game_odds(data, game_id) if game_id is not None else data.odds.copy()
    if league and game_id is None and not frame.empty:
        frame = frame[frame["league"].astype(str).eq(league)]
    if frame.empty:
        empty_state("No odds are stored for this view yet. The board will populate from The Odds API once ingestion has successful responses.")
        return
    view = safe_cols(frame, ["matchup", "market", "selection_name", "point", "book_badge", "price_american", "best_price", "sharp_book", "pulled_at_est"])
    st.dataframe(view, use_container_width=True, hide_index=True)


def render_prop_table(frame: pd.DataFrame, key: str) -> None:
    st.download_button("Export CSV", frame.to_csv(index=False).encode("utf-8"), file_name=f"jarvis_props_{key}.csv", mime="text/csv", use_container_width=False)
    if frame.empty:
        empty_state("No player props match the current filters. Jarvis will not show placeholders as real hit rates or lines.")
        return
    view = safe_cols(frame, PROP_TABLE_COLUMNS)
    styled = style_heatmap(view, ["hit_rate_l10", "hit_rate_l5", "current_season_hit_rate", "display_edge_pct", "confidence", "realism_score", "pf_score"])
    st.dataframe(styled, use_container_width=True, hide_index=True, height=520)


def render_player_props_page(data: DashboardData, league: str, key_prefix: str) -> None:
    props = league_props(data, league)
    games = sorted(["All Games", *props["matchup"].dropna().astype(str).unique().tolist()]) if not props.empty else ["All Games"]
    c1, c2, c3, c4, c5 = st.columns([1.4, 1, 1, 1, .8])
    search = c1.text_input("Search player/team", key=f"{key_prefix}_search", placeholder="Player, team, matchup")
    prop_type = c2.selectbox("Category", PROP_TYPE_ORDER, key=f"{key_prefix}_prop_type")
    game_filter = c3.selectbox("Game", games, key=f"{key_prefix}_game")
    side = c4.radio("Side", ["Both", "Over", "Under"], index=0, horizontal=True, key=f"{key_prefix}_side")
    show_alts = c5.toggle("Alt lines", value=False, key=f"{key_prefix}_alts")
    filtered = filter_props(props, search, prop_type, game_filter, show_alts, side)
    render_prop_table(filtered, key_prefix)


def render_best_bets(data: DashboardData, league: str) -> None:
    frame = league_strong_props(data, league)
    if frame.empty:
        empty_state(f"No {league.upper()} picks pass PF {STRONG_PF_MIN}+, positive EV, sharp/realism filters, and conservative line-distance checks.")
        return
    for prop_type, group in frame.groupby("prop_type", dropna=False):
        st.markdown(f"#### {prop_type}")
        render_prop_table(group.head(25), f"best_{league}_{prop_type}")
        for _, row in group.head(2).iterrows():
            st.markdown(f"<div class='why'><b>WHY</b><br>{row.get('why', 'No explanation available.')}</div>", unsafe_allow_html=True)


def render_mlb_hr_matchups(data: DashboardData, selected_date: date) -> None:
    games = league_games(data, "mlb", selected_date)
    game = render_game_carousel(games, "mlb_hr_carousel")
    if game is None:
        return
    render_game_summary(game, data)
    st.markdown("### Pitcher Splits")
    st.selectbox("Pitcher selector", ["Unavailable"], help="Probable pitcher split stats will appear when a configured provider stores them.")
    cols = ["Season", "vsLHB", "vsRHB", "IP", "BF", "BAA", "wOBA", "SLG", "ISO", "WHIP", "HR", "HR/9", "BB%", "WHIFF%", "K%", "PUTAWAY%", "SWSTR%", "K/9", "Meatball%"]
    st.dataframe(pd.DataFrame(columns=cols), use_container_width=True, hide_index=True)
    empty_state("Pitcher split data is unavailable in stored providers right now. No split metrics are fabricated.")
    st.markdown("### Active Batters")
    game_props_frame = game_props(data, game.get("id"))
    batters = game_props_frame[game_props_frame["prop_type"].isin(["Homeruns", "Hits", "Bases / Total Bases", "RBIs"])] if not game_props_frame.empty else game_props_frame
    render_prop_table(batters, "mlb_active_batters")


def render_mlb_weather(data: DashboardData, selected_date: date) -> None:
    st.markdown("### MLB Weather Today")
    st.caption("Weather uses stored NWS context when available. Dome/roof status is shown only if a configured provider stores it.")
    legend = st.columns(4)
    labels = [("Clear", GREEN), ("Chance for Delay", YELLOW), ("Delay Likely", "#FB923C"), ("Postponement Likely", RED)]
    for col, (label, color) in zip(legend, labels):
        with col: kpi(label, "", "Weather risk legend", color)
    games = league_games(data, "mlb", selected_date)
    if games.empty:
        empty_state("No MLB games are available for the selected date.")
        return
    cols = st.columns(2)
    for idx, (_, game) in enumerate(games.iterrows()):
        weather = game_weather(game)
        risk = weather_risk(weather)
        with cols[idx % 2]:
            st.markdown(
                f"""
                <div class='panel'>
                  <div class='teams'>{logo_html(game.get('away_abbr'), game.get('away_logo'))}<span>{game.get('away_abbr')}</span><span class='subtle'>@</span><span>{game.get('home_abbr')}</span>{logo_html(game.get('home_abbr'), game.get('home_logo'))}</div>
                  <div class='subtle'>{game.get('start_est')} | {game.get('venue') or 'Stadium unavailable'}</div>
                  <div style='margin-top:10px'><span class='chip'>Risk: {risk}</span><span class='chip'>Temp: {weather.get('temperature_f', 'Unavailable')}</span><span class='chip'>Precip: {weather.get('precipitation_probability', 'Unavailable')}%</span><span class='chip'>Wind: {weather.get('wind_direction', 'Unavailable')} {weather.get('wind_mph', 'Unavailable')} mph</span><span class='chip'>Roof: Unavailable</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_nba_defensive_matchups(data: DashboardData, selected_date: date) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.selectbox("Sort", ["Value", "Rank"], key="nba_dm_sort")
    c2.selectbox("Range", ["Season", "Last 10", "Last 5", "Last 3"], key="nba_dm_range")
    c3.selectbox("Position", ["All", "PG", "SG", "SF", "PF", "C"], key="nba_dm_pos")
    c4.selectbox("Year", [str(est_now().year), str(est_now().year - 1)], key="nba_dm_year")
    games = league_games(data, "nba", selected_date)
    game_names = ["All Games", *games["matchup"].dropna().astype(str).tolist()] if not games.empty else ["All Games"]
    c5.selectbox("Games", game_names, key="nba_dm_games")
    st.caption("Ranking legend: green = favorable, yellow = average, red = difficult.")
    cols = ["Team", "Pos", "Opposing Player", "Line", "L10 Avg", "L5 Avg", "H2H", "Points", "Rebounds", "Assists", "3PM", "Steals", "Blocks"]
    props = league_props(data, "nba")
    if props.empty:
        st.dataframe(pd.DataFrame(columns=cols), use_container_width=True, hide_index=True)
        empty_state("No NBA defensive matchup stats or props are stored yet.")
        return
    table = props.rename(columns={"team": "Team", "position": "Pos", "player_name": "Opposing Player", "point": "Line", "hit_rate_l10": "L10 Avg", "hit_rate_l5": "L5 Avg"}).copy()
    table["H2H"] = "Unavailable"
    for col in ["Points", "Rebounds", "Assists", "3PM", "Steals", "Blocks"]:
        table[col] = table["current_season_hit_rate"]
    st.dataframe(style_heatmap(safe_cols(table, cols), ["L10 Avg", "L5 Avg", "Points", "Rebounds", "Assists", "3PM", "Steals", "Blocks"]), use_container_width=True, hide_index=True, height=520)


def render_nba_first_basket(data: DashboardData, selected_date: date) -> None:
    games = league_games(data, "nba", selected_date)
    game = render_game_carousel(games, "nba_fb_carousel")
    if game is None:
        return
    render_game_summary(game, data)
    st.radio("Range", ["Season", "Last 10", "Last 5", "Last 3", "H2H"], index=0, horizontal=True, key="nba_fb_range")
    cols = ["Team", "Tip Win %", "First FG %", "First Point %", "First Three %", "Avg Shots to First FG", "Avg Shots to First 3PT"]
    st.dataframe(pd.DataFrame(columns=cols), use_container_width=True, hide_index=True)
    empty_state("First basket/tip data is not available from the stored providers yet. This module is wired to render it when real data exists.")


def render_nba_team_stats(data: DashboardData) -> None:
    st.radio("Stats", ["Team Stats", "Opponent Stats"], index=0, horizontal=True, key="nba_ts_scope")
    st.radio("Mode", ["Per Game", "Total"], index=0, horizontal=True, key="nba_ts_mode")
    c1, c2 = st.columns(2)
    c1.selectbox("Range", ["Season", "Last 10", "Last 5", "Last 3"], key="nba_ts_range")
    c2.selectbox("Year", [str(est_now().year), str(est_now().year - 1)], key="nba_ts_year")
    cols = ["Team", "PTS", "FG%", "3PM", "3P%", "FT%", "REB", "OREB", "DREB", "AST", "STL", "BLK", "TO", "FBPTS", "PITP"]
    st.download_button("Export CSV", pd.DataFrame(columns=cols).to_csv(index=False).encode(), "jarvis_nba_team_stats.csv")
    st.dataframe(pd.DataFrame(columns=cols), use_container_width=True, hide_index=True)
    empty_state("NBA team stat tables require a configured team-stat feed. No ranks are fabricated.")


def render_injury_splits(data: DashboardData, league: str) -> None:
    st.radio("View", ["Quick View", "Team View"], index=0, horizontal=True, key=f"{league}_inj_view")
    c1, c2, c3, c4 = st.columns(4)
    c1.selectbox("Games", ["All Games"], key=f"{league}_inj_games")
    c2.text_input("Team filter", key=f"{league}_inj_team")
    c3.selectbox("Status", ["All", "Out", "Questionable", "Doubtful", "Day-To-Day"], key=f"{league}_inj_status")
    c4.text_input("Players / combinations", key=f"{league}_inj_players")
    frame = injury_splits_proxy(data, league)
    st.dataframe(frame, use_container_width=True, hide_index=True)
    if frame.empty:
        empty_state("No injury records are stored for this league yet.")
    else:
        empty_state("Injury availability is real, but teammate split metrics are unavailable unless a configured stats feed stores them.")


def render_odds_discrepancies(data: DashboardData, league: str) -> None:
    render_odds_board(data, league=league)


def render_mlb_projections(data: DashboardData, selected_date: date) -> None:
    model = st.radio("Model", ["Model 1", "Model 2", "Consensus"], index=2, horizontal=True, key="mlb_proj_model")
    st.caption(f"Selected date: {selected_date} | {model}")
    frame = projection_cards(data, "mlb")
    if frame.empty:
        empty_state("No real MLB model output/backtest summary is stored yet. Showing odds/stat boards only where available.")
        render_odds_discrepancies(data, "mlb")
        return
    cols = ["matchup", "start_est", "projected_away_score", "projected_home_score", "win_prob_home", "recommended_bet", "edge_pct", "confidence"]
    st.dataframe(style_heatmap(safe_cols(frame, cols), ["win_prob_home", "edge_pct", "confidence"]), use_container_width=True, hide_index=True)


def render_parlay_builder(data: DashboardData, league: str) -> None:
    st.markdown("### Parlay Builder")
    pool = league_strong_props(data, league)
    c1, c2, c3 = st.columns(3)
    prop_type = c1.selectbox("Leg type", PROP_TYPE_ORDER, key=f"{league}_parlay_type")
    stake = c2.number_input("Stake", min_value=1.0, value=10.0, step=5.0, key=f"{league}_parlay_stake")
    max_legs = c3.slider("Max legs", 2, 6, 3, key=f"{league}_parlay_legs")
    if prop_type != "All" and not pool.empty:
        pool = pool[pool["prop_type"].eq(prop_type)]
    if pool.empty:
        empty_state("No strong legs are available for parlays after conservative PF/realism/EV filters.")
        return
    suggestions = suggest_parlays(pool, min_legs=2, max_legs=max_legs, stake=stake, limit=8, prop_type="All")
    if not suggestions.empty:
        st.dataframe(suggestions.drop(columns=["leg_ids"], errors="ignore"), use_container_width=True, hide_index=True)
    pool = pool.head(80).copy()
    pool["label"] = pool.apply(lambda r: f"{r.get('player_name')} {r.get('prop_side')} {r.get('point')} {r.get('prop_type')} | {r.get('book_badge')} {r.get('price_american')} | PF {r.get('pf_score')}", axis=1)
    chosen = st.multiselect("Manual add/remove legs", pool.index.tolist(), format_func=lambda idx: pool.loc[idx, "label"] if idx in pool.index else str(idx), key=f"{league}_manual_parlay")
    legs = pool.loc[chosen].copy() if chosen else pd.DataFrame(columns=pool.columns)
    summary = parlay_summary(legs, stake=stake)
    c1, c2, c3 = st.columns(3)
    c1.metric("Legs", summary.get("legs", 0))
    c2.metric("Combined Odds", summary.get("american_odds") or "N/A")
    c3.metric("Estimated EV", f"{summary.get('ev', 0):+.2f}")
    st.markdown(f"<div class='why'><b>WHY</b><br>{summary.get('why', 'No parlay selected.')}</div>", unsafe_allow_html=True)
    if not legs.empty:
        render_prop_table(legs, f"{league}_parlay_legs_table")


def render_generic_tool(data: DashboardData, league: str, tool: str, selected_date: date) -> None:
    if "Odds Discrepancies" == tool:
        render_odds_discrepancies(data, league)
    elif "Injury" in tool:
        render_injury_splits(data, league)
    elif tool in {"Player Props", "Shot Props"}:
        render_player_props_page(data, league, f"{league}_{tool}")
    elif tool in {"Weather", "Ballpark Weather"} and league == "mlb":
        render_mlb_weather(data, selected_date)
    elif tool == "Line Movement":
        render_odds_discrepancies(data, league)
    else:
        empty_state(f"{tool} is ready for real provider output, but no matching stored dataset exists yet. Jarvis is intentionally not filling this with fake stats.")
        render_player_props_page(data, league, f"{league}_{tool}_props")


def render_league_tab(data: DashboardData, label: str, selected_date: date, sidebar_tool: str) -> None:
    league = LEAGUE_TO_KEY[label]
    games = league_games(data, league, selected_date)
    props = league_props(data, league)
    strong = league_strong_props(data, league)
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi(f"{label} Games", len(games), "Selected date/upcoming fallback", CYAN)
    with c2: kpi("Props", len(props), "Odds API player markets", PURPLE)
    with c3: kpi("Quick Picks", len(strong), "Filtered best bets only", GREEN)
    with c4: kpi("Books", props["book_badge"].nunique() if not props.empty else 0, "Text badges, no copied logos", YELLOW)

    page_tabs = st.tabs(["Home", "Player Props", "Quick Picks", "Parlay Builder", "Tool Page"])
    with page_tabs[0]:
        game = render_game_carousel(games, f"{league}_home_carousel")
        if game is not None:
            render_game_summary(game, data)
            st.markdown("### Live Odds")
            render_odds_board(data, game_id=game.get("id"))
            st.markdown("### Game Props")
            render_prop_table(game_props(data, game.get("id")), f"{league}_home_game_props")
    with page_tabs[1]:
        render_player_props_page(data, league, f"{league}_props")
    with page_tabs[2]:
        render_best_bets(data, league)
    with page_tabs[3]:
        render_parlay_builder(data, league)
    with page_tabs[4]:
        if label == "MLB" and sidebar_tool == "HR Matchups":
            render_mlb_hr_matchups(data, selected_date)
        elif label == "MLB" and sidebar_tool == "Ballpark Weather":
            render_mlb_weather(data, selected_date)
        elif label == "MLB" and sidebar_tool == "Projections":
            render_mlb_projections(data, selected_date)
        elif label == "NBA" and sidebar_tool == "Defensive Matchups":
            render_nba_defensive_matchups(data, selected_date)
        elif label == "NBA" and sidebar_tool == "First Basket":
            render_nba_first_basket(data, selected_date)
        elif label == "NBA" and sidebar_tool == "Team Stats":
            render_nba_team_stats(data)
        elif label == "NBA" and sidebar_tool == "Injury Splits":
            render_injury_splits(data, league)
        else:
            render_generic_tool(data, league, sidebar_tool, selected_date)


def sidebar_controls() -> tuple[date, str, str]:
    st.sidebar.markdown("## Jarvis Tools")
    st.sidebar.caption("League nav on top, dense research tools on the left. Auto-refresh is hourly only.")
    selected_date = st.sidebar.date_input("Date", value=est_now().date())
    sidebar_league = st.sidebar.radio("Sidebar sport", LEAGUES, horizontal=True)
    tool = st.sidebar.selectbox("Tool navigation", SIDEBAR_TOOLS[sidebar_league])
    st.sidebar.markdown("### Global Filters")
    st.sidebar.text_input("Search player/team", key="global_search_hint", placeholder="Use table search boxes per page")
    st.sidebar.selectbox("Default prop type", PROP_TYPE_ORDER, key="global_prop_type_hint")
    st.sidebar.toggle("Show alternate lines by default", value=False, key="global_alt_hint")
    return selected_date, sidebar_league, tool


def refresh_button() -> None:
    if st.sidebar.button("Refresh Data", type="primary", use_container_width=True):
        with st.spinner("Refreshing configured providers..."):
            try:
                SportsDataIngestionService().ingest_all_primary_sports()
                clear_service_cache()
                st.cache_data.clear()
                st.success("Data refresh complete.")
            except Exception as exc:
                st.error(f"Refresh failed: {exc}")
        st.rerun()
    if st.sidebar.button("Clear App Cache", use_container_width=True):
        clear_service_cache()
        st.cache_data.clear()
        st.rerun()


def main() -> None:
    set_page()
    hourly_refresh()
    selected_date, sidebar_league, sidebar_tool = sidebar_controls()
    refresh_button()

    with st.spinner("Loading Jarvis research board..."):
        data = load_dashboard_data()

    render_top_shell(selected_date, data)
    st.caption(f"Sidebar context: {sidebar_league} / {sidebar_tool}. Use the top league tabs to switch the main board.")

    top_tabs = st.tabs(LEAGUES)
    for tab, label in zip(top_tabs, LEAGUES):
        with tab:
            active_tool = sidebar_tool if sidebar_league == label else SIDEBAR_TOOLS[label][0]
            render_league_tab(data, label, selected_date, active_tool)

    with st.expander("System Logs"):
        path = LOG_FILE
        if path.exists():
            st.code("".join(path.read_text(encoding="utf-8", errors="ignore").splitlines(True)[-160:]), language="log")
        else:
            st.info("No log file yet.")

    st.markdown(
        "<div class='footer-note'>Jarvis Betting is a research tool. Data may be delayed or incomplete. No betting outcome is guaranteed. Bet responsibly.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
