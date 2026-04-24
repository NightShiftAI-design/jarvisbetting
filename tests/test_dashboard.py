from dashboard import load_data


def test_dashboard_loader_callable() -> None:
    assert callable(load_data)
