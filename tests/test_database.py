from database import Base, Game, engine, init_db


def test_database_metadata_contains_games_table() -> None:
    init_db()
    assert "games" in Base.metadata.tables
    assert Game.__tablename__ == "games"
    assert engine is not None
