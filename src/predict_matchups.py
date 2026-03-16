"""Predict the probability that Team A beats Team B in a chosen season."""

from __future__ import annotations

import argparse

import pandas as pd

from . import config
from .load_data import DataValidationError, load_seeds, load_team_ratings, load_teams
from .utils import (
    get_team_name,
    get_team_name_for_season,
    load_model_and_metadata,
    parse_seed_value,
    resolve_team_identifier_for_season,
)


def prepare_team_context() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the small data tables needed for matchup prediction."""
    ratings_df = load_team_ratings().copy()
    seeds_df = load_seeds().copy()
    teams_df = load_teams().copy()
    seeds_df["seed_num"] = seeds_df["Seed"].apply(parse_seed_value)
    return ratings_df, seeds_df, teams_df


def load_selected_model(model_name: str):
    """Load the saved model and metadata requested by the user."""
    model_name = model_name.lower()
    if model_name == "logistic":
        # Rebuild the logistic baseline from CSV data instead of relying on a
        # pickled artifact. This avoids scikit-learn/joblib compatibility
        # issues across local and cloud environments.
        from .train_logistic import train_logistic_model

        return train_logistic_model()
    if model_name == "xgboost":
        try:
            return load_model_and_metadata(config.XGBOOST_MODEL_PATH, config.XGBOOST_METADATA_PATH)
        except Exception:
            try:
                from .train_xgboost import train_xgboost_model

                return train_xgboost_model()
            except BaseException as exc:
                raise ValueError(
                    "XGBoost is not available in this environment. "
                    "Use the logistic model instead."
                ) from exc
    raise ValueError("Model must be either 'logistic' or 'xgboost'.")


def get_team_season_row(ratings_df: pd.DataFrame, season: int, team_id: int) -> pd.Series:
    """Fetch one team's season-level ratings row."""
    matches = ratings_df[(ratings_df["Season"] == season) & (ratings_df["TeamID"] == team_id)]
    if matches.empty:
        raise ValueError(
            f"No ratings row found for TeamID {team_id} in season {season}. "
            "Check team_ratings.csv."
        )
    return matches.iloc[0]


def get_team_seed(seeds_df: pd.DataFrame, season: int, team_id: int) -> float:
    """Fetch one team's numeric tournament seed."""
    matches = seeds_df[(seeds_df["Season"] == season) & (seeds_df["TeamID"] == team_id)]
    if matches.empty:
        raise ValueError(
            f"No seed found for TeamID {team_id} in season {season}. Check seeds.csv."
        )
    return float(matches.iloc[0]["seed_num"])


def build_matchup_features(
    season: int,
    team_a_id: int,
    team_b_id: int,
    ratings_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Create a one-row feature dataframe for Team A versus Team B."""
    team_a = get_team_season_row(ratings_df, season, team_a_id)
    team_b = get_team_season_row(ratings_df, season, team_b_id)
    team_a_seed = get_team_seed(seeds_df, season, team_a_id)
    team_b_seed = get_team_seed(seeds_df, season, team_b_id)

    row = {
        "rating_diff": float(team_a["Rating"]) - float(team_b["Rating"]),
        "adjo_diff": float(team_a["AdjO"]) - float(team_b["AdjO"]),
        "adjd_diff": float(team_a["AdjD"]) - float(team_b["AdjD"]),
        "tempo_diff": float(team_a["Tempo"]) - float(team_b["Tempo"]),
        "seed_diff": team_a_seed - team_b_seed,
    }
    if "SOS" in ratings_df.columns:
        row["sos_diff"] = float(team_a.get("SOS", 0.0)) - float(team_b.get("SOS", 0.0))
    if "Luck" in ratings_df.columns:
        row["luck_diff"] = float(team_a.get("Luck", 0.0)) - float(team_b.get("Luck", 0.0))

    return pd.DataFrame([{column: row.get(column, 0.0) for column in feature_columns}])


def get_matchup_prediction_details(
    season: int,
    team_a: str,
    team_b: str,
    model_name: str,
) -> dict[str, object]:
    """Return a full matchup prediction payload for CLI tools or apps."""
    ratings_df, seeds_df, teams_df = prepare_team_context()
    model, metadata = load_selected_model(model_name)
    feature_columns = metadata["feature_columns"]

    team_a_id = resolve_team_identifier_for_season(team_a, season, teams_df, ratings_df)
    team_b_id = resolve_team_identifier_for_season(team_b, season, teams_df, ratings_df)

    feature_df = build_matchup_features(
        season=season,
        team_a_id=team_a_id,
        team_b_id=team_b_id,
        ratings_df=ratings_df,
        seeds_df=seeds_df,
        feature_columns=feature_columns,
    )
    probability = float(model.predict_proba(feature_df)[0, 1])

    team_a_name = get_team_name_for_season(team_a_id, season, ratings_df, teams_df=teams_df)
    team_b_name = get_team_name_for_season(team_b_id, season, ratings_df, teams_df=teams_df)
    return {
        "season": season,
        "model_type": metadata["model_type"],
        "team_a_id": team_a_id,
        "team_b_id": team_b_id,
        "team_a_name": team_a_name,
        "team_b_name": team_b_name,
        "team_a_win_probability": probability,
        "team_b_win_probability": 1 - probability,
        "feature_values": feature_df.iloc[0].to_dict(),
        "metadata": metadata,
    }


def predict_matchup_probability(season: int, team_a: str, team_b: str, model_name: str) -> None:
    """Resolve teams, build features, and print the prediction."""
    prediction = get_matchup_prediction_details(season, team_a, team_b, model_name)
    print(f"Season: {season}")
    print(f"Model: {prediction['model_type']}")
    print(
        f"Probability that {prediction['team_a_name']} beats "
        f"{prediction['team_b_name']}: {prediction['team_a_win_probability']:.4f}"
    )
    print(
        f"Probability that {prediction['team_b_name']} beats "
        f"{prediction['team_a_name']}: {prediction['team_b_win_probability']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Predict a March Madness matchup.")
    parser.add_argument("--season", type=int, required=True, help="Season year, for example 2024")
    parser.add_argument("--team-a", required=True, help="Team A name or numeric TeamID")
    parser.add_argument("--team-b", required=True, help="Team B name or numeric TeamID")
    parser.add_argument(
        "--model",
        default="logistic",
        choices=["logistic", "xgboost"],
        help="Which saved model to use",
    )
    return parser.parse_args()


def main() -> None:
    """Command line entry point."""
    args = parse_args()
    try:
        predict_matchup_probability(
            season=args.season,
            team_a=args.team_a,
            team_b=args.team_b,
            model_name=args.model,
        )
    except (DataValidationError, FileNotFoundError, ValueError) as exc:
        print(exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
