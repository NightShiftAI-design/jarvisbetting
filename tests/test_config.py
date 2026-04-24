from config import ESPN_BASE_URL, THE_ODDS_BASE_URL


def test_base_urls_present() -> None:
    assert ESPN_BASE_URL.startswith("https://")
    assert THE_ODDS_BASE_URL.startswith("https://")
