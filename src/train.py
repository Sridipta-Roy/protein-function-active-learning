"""
Baseline supervised models for protein function classification (Day 5).

Trains on handcrafted features only to answer: how much signal
exists in simple biological sequence features before using ESM embeddings?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

RANDOM_STATE = 42


def _scaling_transformer(feature_cols: list[str]) -> ColumnTransformer:
    """StandardScaler over all numeric feature columns.
    """
    return ColumnTransformer(
        transformers=[("num", StandardScaler(), feature_cols)],
        remainder="drop",
    )


def make_pipelines(feature_cols: list[str], n_classes: int) -> dict[str, Pipeline]:
    """Build one Pipeline per baseline model.
    """
    passthrough = ColumnTransformer(
        transformers=[("num", "passthrough", feature_cols)],
        remainder="drop",
    )

    pipelines: dict[str, Pipeline] = {
        "logreg": Pipeline([
            ("prep", _scaling_transformer(feature_cols)),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "random_forest": Pipeline([
            ("prep", passthrough),
            ("clf", RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),
        "xgboost": Pipeline([
            ("prep", passthrough),
            ("clf", XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),
        "histgb": Pipeline([
            ("prep", passthrough),
            ("clf", HistGradientBoostingClassifier(
                max_iter=400,
                learning_rate=0.1,
                max_depth=6,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ])
    }

    return pipelines


def cv_macro_f1(pipeline: Pipeline, X, y, n_splits: int = 5) -> tuple[float, float]:
    """Stratified k-fold CV on the training set. Returns (mean, std) macro-F1."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        pipeline, X, y, cv=skf, scoring="f1_macro", n_jobs=-1
    )
    return scores.mean(), scores.std()


def evaluate_model(pipeline: Pipeline, X_test, y_test, class_names) -> dict:
    """Fit-already pipeline -> test metrics dict.

    `y_test` and predictions are integer-encoded here; we decode `y_pred` back to
    string class names so the confusion matrix and reports downstream read
    naturally. Metrics are label-invariant so they're computed on the integers.
    """
    y_pred = pipeline.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
        "per_class": report,
        "y_pred": np.asarray(class_names)[y_pred],  # int -> string class names
    }


def run_baselines(
    X_train, y_train, X_test, y_test,
    feature_cols: list[str],
    class_names: list[str],
    n_splits: int = 5,
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, dict]]:
    """Train + evaluate every baseline.

    Returns:
      metrics_df  : tidy table (one row per model) for results/baseline_metrics.csv
      fitted      : dict name -> fitted Pipeline
      evals       : dict name -> full evaluate_model() output (for confusion matrix)
    """
    pipelines = make_pipelines(feature_cols, n_classes=len(class_names))

    # XGBoost needs integer class labels (LogReg/RF tolerate strings, but we
    # encode once for all models so the pipeline is uniform). class_names sets a
    # fixed int<->name mapping; evaluate_model decodes predictions back to names.
    y_train_enc, _ = encode_labels(y_train, class_names)
    y_test_enc, _ = encode_labels(y_test, class_names)

    rows, fitted, evals = [], {}, {}
    for name, pipe in pipelines.items():
        cv_mean, cv_std = cv_macro_f1(pipe, X_train, y_train_enc, n_splits=n_splits)
        pipe.fit(X_train, y_train_enc)
        ev = evaluate_model(pipe, X_test, y_test_enc, class_names)

        fitted[name] = pipe
        evals[name] = ev
        rows.append({
            "model": name,
            "cv_macro_f1_mean": round(cv_mean, 4),
            "cv_macro_f1_std": round(cv_std, 4),
            "test_accuracy": round(ev["accuracy"], 4),
            "test_macro_f1": round(ev["macro_f1"], 4),
            "test_weighted_f1": round(ev["weighted_f1"], 4),
        })

    metrics_df = pd.DataFrame(rows).sort_values("test_macro_f1", ascending=False)
    return metrics_df, fitted, evals


def encode_labels(y: pd.Series, class_order: list[str]) -> tuple[np.ndarray, LabelEncoder]:
    """Encode string labels to ints using a fixed class order (stable across runs)."""
    le = LabelEncoder()
    le.classes_ = np.array(class_order)
    return le.transform(y), le
