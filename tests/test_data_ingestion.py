from data_ingestion import ESPNClient, OddsAPIClient, SportsDataIngestionService


def test_client_objects_construct() -> None:
    assert ESPNClient().base_url.startswith("https://")
    assert OddsAPIClient().base_url.startswith("https://")
    assert isinstance(SportsDataIngestionService(), SportsDataIngestionService)
