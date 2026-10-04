"""Score a CSV of transactions with the saved model.

    python -m fraud_detection.predict data/test.csv outputs/test_predictions.csv

The output has one row per input row, in the same order: `id` (when the input has one), `fraud_score`
(probability of fraud, 0 to 1) and `flag` (1 = suspected fraud at the model's Operating Threshold).
"""
import argparse
import warnings
from pathlib import Path

import joblib
import pandas as pd

from fraud_detection.features import make_features

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "model.pkl"

# NumPy on macOS (Accelerate BLAS) emits spurious matmul warnings during scikit-learn predictions; results are unaffected.
warnings.filterwarnings("ignore", message=r".* encountered in matmul", category=RuntimeWarning)


def predict(df, model_path=DEFAULT_MODEL):
    """Fraud Scores and 0/1 Flags for every row of `df`."""
    model, features = joblib.load(model_path)
    X = make_features(df)[features]
    result = pd.DataFrame({"fraud_score": model.predict_proba(X)[:, 1], "flag": model.predict(X)}, index=df.index)
    if "id" in df.columns:
        result.insert(0, "id", df["id"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path, help="CSV with the raw transaction columns")
    parser.add_argument("output", type=Path, help="where to write the predictions CSV")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="path to model.pkl")
    args = parser.parse_args()

    result = predict(pd.read_csv(args.input), args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"wrote {len(result)} predictions to {args.output} ({result['flag'].mean():.2%} flagged)")


if __name__ == "__main__":
    main()
