# Jarvis Betting

Jarvis Betting is a Python/Streamlit sports betting research dashboard for NBA, NHL, MLB, and NFL. It uses configured provider data where available and shows explicit unavailable/empty states instead of presenting fake odds, stats, hit rates, injuries, weather, or projections.

## Setup

```bash
cd /Users/gshyamp/Documents/Jarvis_Betting
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python database.py
streamlit run dashboard.py
```

## Data Providers

- ESPN public API for schedules, teams, logos, and scoreboard context.
- The Odds API for game odds and player prop markets.
- API-Football / API-Sports for soccer context where configured.
- TheSportsDB for supported public sports lookups.
- NWS weather context for supported US venue coordinates when available.
- Rotowire/ESPN injury context only when real public data can be read.

## Environment / Keys

The project currently reads the provided keys from `config.py` as requested by the original build. Do not expose those keys in frontend JavaScript. The Streamlit dashboard calls Python services server-side.

## Run Data Refresh

Use the dashboard `Refresh Data` button, or call ingestion from Python:

```bash
python data_ingestion.py
```

## Dashboard

The dashboard includes league tabs, sport-specific sidebar tools, dense prop tables, CSV export, sportsbook text badges, team logo fallbacks, weather boards, injury views, projection cards when real model output exists, conservative best bets, and a parlay builder using only strong filtered legs.

Jarvis Betting is a research tool. Data may be delayed or incomplete. No betting outcome is guaranteed. Bet responsibly.
