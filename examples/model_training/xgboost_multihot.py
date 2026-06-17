import logging

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import f1_score, jaccard_score

from ehrdrec.loading import MIMIC3Loader
from ehrdrec.processing import MultiHotProcessor


logging.getLogger("ehrdrec").setLevel(logging.INFO)
logging.basicConfig()

ATC_LEVEL      = 5
MIN_ADMISSIONS = 2
LOOK_BACK      = 3  # number of previous admissions to include as context


def build_lookback_arrays(
    frame: pl.LazyFrame,
    feature_cols: list[str],
    target_col: str,
    look_back: int,
) -> tuple[np.ndarray, np.ndarray]:
    df = frame.sort(["patient_id", "admission_time"]).collect()

    feature_dim = sum(len(df[col][0]) for col in feature_cols)
    target_dim  = len(df[target_col][0])

    rows_X = []
    rows_y = []

    for _, patient_df in df.group_by("patient_id", maintain_order=True):
        n_visits = patient_df.height

        for visit_idx in range(n_visits):
            current_row = patient_df.row(visit_idx, named=True)

            # Current visit features
            current_feats = []
            for col in feature_cols:
                current_feats.extend(current_row[col])

            # Previous visits: up to look_back admissions of (features + medications)
            # Each slot is zero-padded if there is no prior admission
            history_feats = []
            for offset in range(look_back, 0, -1):
                prior_idx = visit_idx - offset
                if prior_idx >= 0:
                    prior_row = patient_df.row(prior_idx, named=True)
                    slot = []
                    for col in feature_cols:
                        slot.extend(prior_row[col])
                    slot.extend(prior_row[target_col])
                else:
                    slot = [0.0] * (feature_dim + target_dim)
                history_feats.extend(slot)

            rows_X.append(current_feats + history_feats)
            rows_y.append(current_row[target_col])

    return np.array(rows_X, dtype=np.float32), np.array(rows_y, dtype=np.float32)


if __name__ == "__main__":
    loader = MIMIC3Loader()
    data = loader.load("/home/cararc/data/mimic-iii-1.4")

    processor = MultiHotProcessor()
    processed = processor.process(
        data,
        minimum_admissions=MIN_ADMISSIONS,
        atc_level=ATC_LEVEL,
        force_reload=False,
    )

    feature_cols = ["diagnosis_multihot", "procedure_multihot"]
    target_col   = "medication_multihot"

    print("Building arrays...")
    X_train, y_train = build_lookback_arrays(processed.train_frame, feature_cols, target_col, LOOK_BACK)
    X_val,   y_val   = build_lookback_arrays(processed.val_frame,   feature_cols, target_col, LOOK_BACK)
    X_test,  y_test  = build_lookback_arrays(processed.test_frame,  feature_cols, target_col, LOOK_BACK)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Output dim: {y_train.shape[1]}")

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        device="cuda",
        n_jobs=-1,
    )

    print("Training...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=10,
    )

    print("Evaluating...")
    y_pred = model.predict(X_test)

    jaccard = jaccard_score(y_test, y_pred, average="samples", zero_division=0)
    f1      = f1_score(y_test, y_pred, average="samples", zero_division=0)

    print(f"Jaccard (samples): {jaccard:.4f}")
    print(f"F1      (samples): {f1:.4f}")
