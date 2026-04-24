from utils import american_to_decimal, american_to_implied_probability


def test_odds_conversions() -> None:
    assert round(american_to_decimal(100), 2) == 2.00
    assert american_to_implied_probability(-150) is not None
