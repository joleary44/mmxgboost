"""Functions for loading and validating raw input CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from . import config
from .utils import build_team_lookup, normalize_team_name


class DataValidationError(Exception):
    """Raised when an input file is missing or malformed."""


TEAM_RATINGS_ALIASES = {
    "Season": ["Season", "season", "YEAR", "Year"],
    "TeamID": ["TeamID", "TeamId", "team_id", "ID"],
    "TeamName": ["TeamName", "Team", "team", "School", "TEAM"],
    "AdjO": ["AdjO", "AdjOE", "Adj Off", "Adj_Off", "OffEff"],
    "AdjD": ["AdjD", "AdjDE", "Adj Def", "Adj_Def", "DefEff"],
    "Tempo": ["Tempo", "AdjT", "TempoAdj", "Adj Tempo"],
    "Rating": ["Rating", "AdjEM", "PowerRating", "BARTHAG", "BartRating"],
    "SOS": ["SOS", "SoS", "StrengthOfSchedule"],
    "Luck": ["Luck", "luck"],
}


def ensure_project_directories() -> None:
    """Create the standard output folders if they do not already exist."""
    for path in [
        config.DATA_DIR,
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.MODELS_DIR,
        config.OUTPUTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def read_csv_checked(
    file_path: Path,
    required_columns: Iterable[str],
    optional_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Read a CSV file and validate its required columns."""
    if not file_path.exists():
        raise DataValidationError(
            f"Missing required file: {file_path}\n"
            "Please place the CSV in the data/raw/ folder."
        )

    try:
        dataframe = pd.read_csv(file_path)
    except Exception as exc:  # pragma: no cover - defensive error wrapper
        raise DataValidationError(f"Could not read CSV file: {file_path}\n{exc}") from exc

    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        optional_text = ""
        if optional_columns:
            optional_text = f"\nOptional columns: {list(optional_columns)}"
        raise DataValidationError(
            f"File {file_path} is missing required columns: {missing_columns}"
            f"\nFound columns: {list(dataframe.columns)}{optional_text}"
        )

    return dataframe.copy()


def find_first_matching_column(dataframe: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name from candidates that exists in the dataframe."""
    existing_columns = set(dataframe.columns)
    for candidate in candidates:
        if candidate in existing_columns:
            return candidate
    return None


def normalize_team_ratings_columns(raw_ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Rename common KenPom-style column names into the project's standard names."""
    dataframe = raw_ratings_df.copy()
    rename_map: dict[str, str] = {}

    for standard_name, candidate_names in TEAM_RATINGS_ALIASES.items():
        matched_column = find_first_matching_column(dataframe, candidate_names)
        if matched_column is not None:
            rename_map[matched_column] = standard_name

    dataframe = dataframe.rename(columns=rename_map)
    return dataframe


def attach_team_ids_from_names(ratings_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    """Fill TeamID by matching TeamName values against teams.csv when needed."""
    dataframe = ratings_df.copy()
    if "TeamID" in dataframe.columns and dataframe["TeamID"].notna().all():
        return dataframe

    if "TeamName" not in dataframe.columns:
        raise DataValidationError(
            "team_ratings.csv is missing TeamID, and no TeamName column was found to map IDs.\n"
            "Include either a TeamID column or a Team/TeamName column."
        )

    lookup = build_team_lookup(teams_df)
    normalized_to_id = pd.Series(lookup, name="TeamID")
    dataframe["_normalized_team_name"] = dataframe["TeamName"].astype(str).map(normalize_team_name)
    dataframe["TeamID"] = dataframe["_normalized_team_name"].map(normalized_to_id)

    missing_names = sorted(
        dataframe.loc[dataframe["TeamID"].isna(), "TeamName"].astype(str).drop_duplicates().tolist()
    )
    if missing_names:
        preview = missing_names[:10]
        raise DataValidationError(
            "Some team names in team_ratings.csv could not be matched to teams.csv.\n"
            f"Examples: {preview}\n"
            "Make the team names match teams.csv more closely or add a TeamID column."
        )

    dataframe["TeamID"] = dataframe["TeamID"].astype(int)
    dataframe = dataframe.drop(columns=["_normalized_team_name"])
    return dataframe


def load_regular_season_results() -> pd.DataFrame:
    """Load regular season game results."""
    return read_csv_checked(
        config.REGULAR_SEASON_RESULTS_PATH,
        required_columns=["Season", "WTeamID", "LTeamID", "WScore", "LScore"],
        optional_columns=["NumOT"],
    )


def load_tournament_results() -> pd.DataFrame:
    """Load historical tournament game results."""
    return read_csv_checked(
        config.TOURNAMENT_RESULTS_PATH,
        required_columns=["Season", "WTeamID", "LTeamID", "WScore", "LScore"],
        optional_columns=["NumOT"],
    )


def load_seeds() -> pd.DataFrame:
    """Load tournament seeds for each season."""
    return read_csv_checked(
        config.SEEDS_PATH,
        required_columns=["Season", "TeamID", "Seed"],
    )


def load_teams() -> pd.DataFrame:
    """Load the team ID to team name mapping."""
    return read_csv_checked(
        config.TEAMS_PATH,
        required_columns=["TeamID", "TeamName"],
    )


def load_team_ratings() -> pd.DataFrame:
    """Load season-level team ratings."""
    teams_df = load_teams()
    raw_ratings_df = read_csv_checked(
        config.TEAM_RATINGS_PATH,
        required_columns=[],
    )
    ratings_df = normalize_team_ratings_columns(raw_ratings_df)
    ratings_df = attach_team_ids_from_names(ratings_df, teams_df)

    required_columns = ["Season", "TeamID", "AdjO", "AdjD", "Tempo", "Rating"]
    missing_columns = [column for column in required_columns if column not in ratings_df.columns]
    if missing_columns:
        alias_help = {
            "Season": TEAM_RATINGS_ALIASES["Season"],
            "TeamID": TEAM_RATINGS_ALIASES["TeamID"],
            "TeamName": TEAM_RATINGS_ALIASES["TeamName"],
            "AdjO": TEAM_RATINGS_ALIASES["AdjO"],
            "AdjD": TEAM_RATINGS_ALIASES["AdjD"],
            "Tempo": TEAM_RATINGS_ALIASES["Tempo"],
            "Rating": TEAM_RATINGS_ALIASES["Rating"],
        }
        raise DataValidationError(
            f"team_ratings.csv is missing required columns after normalization: {missing_columns}\n"
            f"Found columns: {list(ratings_df.columns)}\n"
            f"Accepted aliases: {alias_help}"
        )

    numeric_columns = ["Season", "TeamID", "AdjO", "AdjD", "Tempo", "Rating"]
    for optional_column in ["SOS", "Luck"]:
        if optional_column in ratings_df.columns:
            numeric_columns.append(optional_column)

    for column in numeric_columns:
        ratings_df[column] = pd.to_numeric(ratings_df[column], errors="coerce")

    before_drop = len(ratings_df)
    ratings_df = ratings_df.dropna(subset=["Season", "TeamID", "AdjO", "AdjD", "Tempo", "Rating"]).copy()
    dropped_rows = before_drop - len(ratings_df)
    if dropped_rows > 0:
        print(f"Warning: dropped {dropped_rows} rating rows with missing numeric values.")

    ratings_df["Season"] = ratings_df["Season"].astype(int)
    ratings_df["TeamID"] = ratings_df["TeamID"].astype(int)
    return ratings_df


def load_all_input_data() -> dict[str, pd.DataFrame]:
    """Load the CSV files used by the current baseline pipeline.

    The project can train the baseline model without regular season results, so
    that file is loaded on a best-effort basis and returned as an empty
    dataframe when it is not present.
    """
    ensure_project_directories()
    try:
        regular_season_results = load_regular_season_results()
    except DataValidationError as exc:
        print("Warning: regular season results were not loaded.")
        print(exc)
        print("The current baseline does not require this file, so processing will continue.")
        regular_season_results = pd.DataFrame()

    return {
        "regular_season_results": regular_season_results,
        "tournament_results": load_tournament_results(),
        "seeds": load_seeds(),
        "teams": load_teams(),
        "team_ratings": load_team_ratings(),
    }
