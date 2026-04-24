import pandas as pd

from models import FeatureBuilder, JarvisPredictor


def test_predictor_smoke() -> None:
    builder = FeatureBuilder()
    predictor = JarvisPredictor()
    frame = builder.build(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert frame.empty
    assert predictor.features
