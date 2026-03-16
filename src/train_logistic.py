"""Train and evaluate the logistic regression baseline model."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .utils import get_available_feature_columns, save_model_and_metadata


def load_processed_games() -> pd.DataFrame:
    """Load the processed modeling dataset from disk."""
    if not config.PROCESSED_MODEL_GAMES_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {config.PROCESSED_MODEL_GAMES_PATH}\n"
            "Run `python -m src.feature_engineering` first."
        )
    return pd.read_csv(config.PROCESSED_MODEL_GAMES_PATH)


def split_train_validation(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by season to avoid leakage between training and validation."""
    train_df = dataframe[dataframe[config.SEASON_COLUMN].isin(config.TRAIN_SEASONS)].copy()
    validation_df = dataframe[dataframe[config.SEASON_COLUMN].isin(config.VALIDATION_SEASONS)].copy()

    if train_df.empty:
        raise ValueError("Training split is empty. Check TRAIN_SEASONS in src/config.py.")
    if validation_df.empty:
        raise ValueError("Validation split is empty. Check VALIDATION_SEASONS in src/config.py.")
    return train_df, validation_df


def build_logistic_pipeline() -> Pipeline:
    """Create a simple, reliable pipeline for baseline classification."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    solver="liblinear",
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
    """Calculate beginner-friendly evaluation metrics."""
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


def train_logistic_model() -> tuple[Pipeline, dict[str, Any]]:
    """Train the logistic regression baseline and save it."""
    dataframe = load_processed_games()
    feature_columns = get_available_feature_columns(dataframe)
    if not feature_columns:
        raise ValueError("No valid feature columns were found in the processed dataset.")

    train_df, validation_df = split_train_validation(dataframe)
    x_train = train_df[feature_columns]
    y_train = train_df[config.TARGET_COLUMN]

    model = build_logistic_pipeline()
    print("Training logistic regression model...")
    model.fit(x_train, y_train)

    metrics = evaluate_classifier(model, validation_df, feature_columns)
    metadata = {
        "model_type": "logistic_regression",
        "feature_columns": feature_columns,
        "train_seasons": config.TRAIN_SEASONS,
        "validation_seasons": config.VALIDATION_SEASONS,
        "metrics": metrics,
    }
    save_model_and_metadata(
        model=model,
        model_path=config.LOGISTIC_MODEL_PATH,
        metadata_path=config.LOGISTIC_METADATA_PATH,
        metadata=metadata,
    )

    print("Training complete.")
    print(f"Saved model to: {config.LOGISTIC_MODEL_PATH}")
    print(f"Validation log loss: {metrics['log_loss']:.4f}")
    print(f"Validation accuracy: {metrics['accuracy']:.4f}")
    print(f"Validation Brier score: {metrics['brier_score']:.4f}")
    return model, metadata


def main() -> None:
    """Command line entry point."""
    train_logistic_model()


if __name__ == "__main__":
    main()
