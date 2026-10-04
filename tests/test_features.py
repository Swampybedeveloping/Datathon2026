import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detection.features import make_features
from fraud_detection.predict import predict

ROOT = Path(__file__).resolve().parents[1]


def notebook_function(notebook, name="def make_features"):
    cells = json.loads((ROOT / "notebooks" / notebook).read_text())["cells"]
    return next("".join(c["source"]) for c in cells if "".join(c["source"]).startswith(name))


def test_package_matches_notebooks():
    package = inspect.getsource(make_features).strip()
    assert notebook_function("fraud_detection.ipynb").strip() == package
    assert notebook_function("predict.ipynb").strip() == package


def test_bad_values_are_guarded():
    raw = pd.read_csv(ROOT / "data" / "test.csv").head(3)
    raw["transaction_amount"] = raw["transaction_amount"].astype(object)
    raw.loc[0, "transaction_amount"] = "not a number"
    raw.loc[1, "transaction_amount"] = 10_000_000
    raw.loc[2, "spend_last_24h"] = np.inf
    out = make_features(raw)
    assert np.isnan(out.loc[0, "transaction_amount"])
    assert out.loc[1, "transaction_amount"] == 9000.0
    assert not np.isinf(out.select_dtypes("number")).any().any()
    assert len(out) == len(raw)


def test_predict_matches_saved_predictions():
    test = pd.read_csv(ROOT / "data" / "test.csv")
    result = predict(test)
    saved = pd.read_csv(ROOT / "outputs" / "test_predictions.csv")
    assert list(result["id"]) == list(saved["id"])
    np.testing.assert_allclose(result["fraud_score"], saved["fraud_score"], rtol=1e-9)
    assert set(result["flag"].unique()) <= {0, 1}
