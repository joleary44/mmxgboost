"""Train and evaluate an optional XGBoost model."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline

from . import config
from .train_logistic import load_processed_games, split_train_validation
from .utils import get_available_feature_columns, save_model_and_metadata


def import_xgboost_classifier():
    """Import XGBoost only when needed so the project still runs without it."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("XGBoost is not installed.")
        print("You can still use the logistic baseline.")
        print("To install XGBoost, run: pip install xgboost")
        raise SystemExit(0)
    except Exception as exc:
        print("XGBoost is installed, but it could not be loaded.")
        print(f"Details: {exc}")
        print("On macOS, this usually means the OpenMP runtime is missing.")
        print("Try installing it with: brew install libomp")
        raise SystemExit(0)
    return XGBClassifier


def build_xgboost_pipeline() -> Pipeline:
    """Create a simple XGBoost pipeline."""
    xgb_classifier = import_xgboost_classifier()
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                xgb_classifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=config.RANDOM_SEED,
                ),
            ),
        ]
    )


def evaluate_classifier(
    model: Pipeline,
    validation_df: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, float]:
    """Calculate the same metrics as the logistic baseline for easy comparison."""
    x_valid = validation_df[feature_columns]
    y_valid = validation_df[config.TARGET_COLUMN]
    predicted_probabilities = model.predict_proba(x_valid)[:, 1]
    predicted_labels = (predicted_probabilities >= 0.5).astype(int)

    metrics = {
        "log_loss": float(log_loss(y_valid, predicted_probabilities)),
        "accuracy": float(accuracy_score(y_valid, predicted_labels)),
        "brier_score": float(brier_score_loss(y_valid, predicted_probabilities)),
    }
    return metrics


def train_xgboost_model() -> tuple[Pipeline, dict[str, Any]]:
    """Train the optional XGBoost model and save it."""
    dataframe = load_processed_games()
    feature_columns = get_available_feature_columns(dataframe)
    if not feature_columns:
        raise ValueError("No valid feature columns were found in the processed dataset.")

    train_df, validation_df = split_train_validation(dataframe)
    x_train = train_df[feature_columns]
    y_train = train_df[config.TARGET_COLUMN]

    model = build_xgboost_pipeline()
    print("Training XGBoost model...")
    model.fit(x_train, y_train)

    metrics = evaluate_classifier(model, validation_df, feature_columns)
    metadata = {
        "model_type": "xgboost",
        "feature_columns": feature_columns,
        "train_seasons": config.TRAIN_SEASONS,
        "validation_seasons": config.VALIDATION_SEASONS,
        "metrics": metrics,
    }
    save_model_and_metadata(
        model=model,
        model_path=config.XGBOOST_MODEL_PATH,
        metadata_path=config.XGBOOST_METADATA_PATH,
        metadata=metadata,
    )

    print("Training complete.")
    print(f"Saved model to: {config.XGBOOST_MODEL_PATH}")
    print(f"Validation log loss: {metrics['log_loss']:.4f}")
    print(f"Validation accuracy: {metrics['accuracy']:.4f}")
    print(f"Validation Brier score: {metrics['brier_score']:.4f}")
    return model, metadata


def main() -> None:
    """Command line entry point."""
    train_xgboost_model()


if __name__ == "__main__":
    main()
