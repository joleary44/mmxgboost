"""Build the tournament modeling dataset from raw CSV files."""

from __future__ import annotations

import pandas as pd

from . import config
from .load_data import DataValidationError, load_all_input_data
from .utils import parse_seed_value


def prepare_seed_dataframe(seeds_df: pd.DataFrame) -> pd.DataFrame:
    """Convert seed text values into numeric seeds."""
    seeds = seeds_df.copy()
    seeds["seed_num"] = seeds["Seed"].apply(parse_seed_value)
    return seeds[["Season", "TeamID", "Seed", "seed_num"]]


def prepare_ratings_dataframe(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns needed for modeling."""
    keep_columns = ["Season", "TeamID", "AdjO", "AdjD", "Tempo", "Rating"]
    for optional_column in ["SOS", "Luck"]:
        if optional_column in ratings_df.columns:
            keep_columns.append(optional_column)
    return ratings_df[keep_columns].copy()


def create_balanced_tournament_rows(tournament_df: pd.DataFrame) -> pd.DataFrame:
    """Create mirrored rows so Team A is not always the winner or loser."""
    winner_rows = pd.DataFrame(
        {
            "Season": tournament_df["Season"],
            "TeamAID": tournament_df["WTeamID"],
            "TeamBID": tournament_df["LTeamID"],
            "TeamAScore": tournament_df["WScore"],
            "TeamBScore": tournament_df["LScore"],
            "result": 1,
        }
    )
    loser_rows = pd.DataFrame(
        {
            "Season": tournament_df["Season"],
            "TeamAID": tournament_df["LTeamID"],
            "TeamBID": tournament_df["WTeamID"],
            "TeamAScore": tournament_df["LScore"],
            "TeamBScore": tournament_df["WScore"],
            "result": 0,
        }
    )
    combined = pd.concat([winner_rows, loser_rows], ignore_index=True)
    combined["score_diff"] = combined["TeamAScore"] - combined["TeamBScore"]
    return combined


def merge_team_side_features(
    games_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge ratings and seed data for Team A and Team B."""
    team_a_ratings = ratings_df.add_prefix("TeamA_").rename(
        columns={"TeamA_Season": "Season", "TeamA_TeamID": "TeamAID"}
    )
    team_b_ratings = ratings_df.add_prefix("TeamB_").rename(
        columns={"TeamB_Season": "Season", "TeamB_TeamID": "TeamBID"}
    )
    team_a_seeds = seeds_df.add_prefix("TeamA_").rename(
        columns={"TeamA_Season": "Season", "TeamA_TeamID": "TeamAID"}
    )
    team_b_seeds = seeds_df.add_prefix("TeamB_").rename(
        columns={"TeamB_Season": "Season", "TeamB_TeamID": "TeamBID"}
    )

    merged = games_df.merge(team_a_ratings, on=["Season", "TeamAID"], how="left")
    merged = merged.merge(team_b_ratings, on=["Season", "TeamBID"], how="left")
    merged = merged.merge(team_a_seeds[["Season", "TeamAID", "TeamA_seed_num"]], on=["Season", "TeamAID"], how="left")
    merged = merged.merge(team_b_seeds[["Season", "TeamBID", "TeamB_seed_num"]], on=["Season", "TeamBID"], how="left")
    return merged


def create_difference_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """Create model features as Team A minus Team B values."""
    dataframe = games_df.copy()
    dataframe["rating_diff"] = dataframe["TeamA_Rating"] - dataframe["TeamB_Rating"]
    dataframe["adjo_diff"] = dataframe["TeamA_AdjO"] - dataframe["TeamB_AdjO"]
    dataframe["adjd_diff"] = dataframe["TeamA_AdjD"] - dataframe["TeamB_AdjD"]
    dataframe["tempo_diff"] = dataframe["TeamA_Tempo"] - dataframe["TeamB_Tempo"]
    dataframe["seed_diff"] = dataframe["TeamA_seed_num"] - dataframe["TeamB_seed_num"]

    if "TeamA_SOS" in dataframe.columns and "TeamB_SOS" in dataframe.columns:
        dataframe["sos_diff"] = dataframe["TeamA_SOS"] - dataframe["TeamB_SOS"]
    if "TeamA_Luck" in dataframe.columns and "TeamB_Luck" in dataframe.columns:
        dataframe["luck_diff"] = dataframe["TeamA_Luck"] - dataframe["TeamB_Luck"]

    return dataframe


def build_modeling_dataset() -> pd.DataFrame:
    """Build and return the processed tournament game dataset."""
    data = load_all_input_data()
    tournament_df = data["tournament_results"]
    seeds_df = prepare_seed_dataframe(data["seeds"])
    ratings_df = prepare_ratings_dataframe(data["team_ratings"])

    games = create_balanced_tournament_rows(tournament_df)
    games = merge_team_side_features(games, ratings_df, seeds_df)
    games = create_difference_features(games)

    required_feature_sources = [
        "TeamA_Rating",
        "TeamB_Rating",
        "TeamA_AdjO",
        "TeamB_AdjO",
        "TeamA_AdjD",
        "TeamB_AdjD",
        "TeamA_Tempo",
        "TeamB_Tempo",
        "TeamA_seed_num",
        "TeamB_seed_num",
    ]
    before_drop = len(games)
    games = games.dropna(subset=required_feature_sources).reset_index(drop=True)
    dropped_rows = before_drop - len(games)

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    games.to_csv(config.PROCESSED_MODEL_GAMES_PATH, index=False)

    print("Feature engineering complete.")
    print(f"Saved processed dataset to: {config.PROCESSED_MODEL_GAMES_PATH}")
    print(f"Rows kept: {len(games)}")
    print(f"Rows dropped because of missing required features: {dropped_rows}")
    return games


def main() -> None:
    """Command line entry point."""
    try:
        build_modeling_dataset()
    except DataValidationError as exc:
        print("Data validation error while building features:")
        print(exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
