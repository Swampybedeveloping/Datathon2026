"""Feature engineering shared by training (notebooks/fraud_detection.ipynb, section 4.2) and scoring.

The function is self-contained (imports inside) so it can be pasted anywhere, as the competition required.
Keep it identical to the `make_features` cell in the notebook; tests/test_features.py checks this.
"""


def make_features(df):
    """Guard against bad values, then add the engineered features (section 4.1)."""
    import numpy as np
    import pandas as pd

    numeric = ["transaction_amount", "transaction_hour", "transactions_last_24h", "spend_last_24h",
               "account_age", "new_device", "transactions_last_1h"]
    categorical = ["merchant_category", "country", "transaction_channel"]
    out = df.copy()

    # Guards against bad values in unseen data
    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)   # text -> missing
    for col in categorical:
        values = out[col].astype(object)
        out[col] = values.astype(str).where(values.notna(), np.nan)          # categories as text
    training_max = {"transaction_amount": 9000.0, "spend_last_24h": 15670.31, "account_age": 4000.0,
                    "transactions_last_24h": 27.0, "transactions_last_1h": 9.0}       # from section 2.2
    for col, upper in training_max.items():
        out[col] = out[col].clip(lower=0, upper=upper)                       # stay inside the training range

    # Engineered features (4.1)
    hour = out["transaction_hour"]
    out["log_amount"] = np.log1p(out["transaction_amount"])
    out["amount_vs_spend_24h"] = np.log1p(out["transaction_amount"]) - np.log1p(out["spend_last_24h"])
    out["log_account_age"] = np.log1p(out["account_age"])
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["is_night"] = hour.isin([23, 0, 1, 2, 3]).astype(float).where(hour.notna())

    # Extra features for the final model (section 7.6)
    out["avg_spend_per_txn"] = np.log1p(out["spend_last_24h"] / out["transactions_last_24h"].clip(lower=1))
    out["amount_vs_avg_txn"] = out["log_amount"] - out["avg_spend_per_txn"]
    out["share_last_1h"] = out["transactions_last_1h"] / out["transactions_last_24h"].clip(lower=1)
    out["newdev_x_logamt"] = out["new_device"].fillna(0) * out["log_amount"]
    out["young_x_newdev"] = out["new_device"].fillna(0) * (out["account_age"] < 230)

    # No infinities
    engineered = ["log_amount", "amount_vs_spend_24h", "log_account_age", "hour_sin", "hour_cos", "is_night",
                  "avg_spend_per_txn", "amount_vs_avg_txn", "share_last_1h", "newdev_x_logamt", "young_x_newdev"]
    out[numeric + engineered] = out[numeric + engineered].replace([np.inf, -np.inf], np.nan)
    return out
