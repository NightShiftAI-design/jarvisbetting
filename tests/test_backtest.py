from backtest import Backtester


def test_backtester_constructs() -> None:
    assert isinstance(Backtester(), Backtester)
