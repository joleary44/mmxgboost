"""Convert a custom Kaggle March Madness archive into the project's raw CSVs.

This helper is designed for archives that contain files such as:

- KenPom Barttorvik.csv
- Tournament Matchups.csv

It writes the standard project files into data/raw/ so the rest of the
pipeline can run without manual cleanup.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import config
from .load_data import DataValidationError, ensure_project_directories
from .utils import normalize_team_name


DEFAULT_ARCHIVE_DIR = Path.home() / "Downloads" / "archive"


def require_columns(dataframe: pd.DataFrame, required_columns: list[str], file_label: str) -> None:
    """Raise a clear error if a source dataframe is missing required columns."""
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise DataValidationError(
            f"{file_label} is missing required columns: {missing_columns}\n"
            f"Found columns: {list(dataframe.columns)}"
        )


def read_archive_csv(archive_dir: Path, filename: str) -> pd.DataFrame:
    """Read one CSV file from the archive directory."""
    file_path = archive_dir / filename
    if not file_path.exists():
        raise DataValidationError(
            f"Could not find {filename} in {archive_dir}\n"
            "Please point the script at the folder that contains the extracted Kaggle CSV files."
        )
    return pd.read_csv(file_path)


def build_team_ratings(ratings_source_df: pd.DataFrame) -> pd.DataFrame:
    """Convert the archive ratings file into team_ratings.csv format."""
    require_columns(
        ratings_source_df,
        ["YEAR", "TEAM ID", "TEAM", "KADJ O", "KADJ D", "KADJ T", "KADJ EM"],
        "KenPom Barttorvik.csv",
    )

    ratings_df = ratings_source_df.rename(
        columns={
            "YEAR": "Season",
            "TEAM ID": "TeamID",
            "TEAM": "TeamName",
            "SEED": "Seed",
            "KADJ O": "AdjO",
            "KADJ D": "AdjD",
            "KADJ T": "Tempo",
            "KADJ EM": "Rating",
            "ELITE SOS": "SOS",
        }
    )

    keep_columns = ["Season", "TeamID", "TeamName", "AdjO", "AdjD", "Tempo", "Rating"]
    if "Seed" in ratings_df.columns:
        keep_columns.append("Seed")
    if "SOS" in ratings_df.columns:
        keep_columns.append("SOS")

    ratings_df = ratings_df[keep_columns].copy()
    numeric_columns = [column for column in keep_columns if column != "TeamName"]
    for column in numeric_columns:
        ratings_df[column] = pd.to_numeric(ratings_df[column], errors="coerce")

    ratings_df = ratings_df.dropna(subset=["Season", "TeamID", "AdjO", "AdjD", "Tempo", "Rating"]).copy()
    ratings_df["Season"] = ratings_df["Season"].astype(int)
    ratings_df["TeamID"] = ratings_df["TeamID"].astype(int)
    ratings_df["TeamName"] = ratings_df["TeamName"].astype(str)
    ratings_df = ratings_df.drop_duplicates(subset=["Season", "TeamID"]).sort_values(["Season", "TeamID"])
    return ratings_df.reset_index(drop=True)


def map_ratings_to_tournament_team_ids(
    ratings_df: pd.DataFrame,
    matchup_df: pd.DataFrame,
) -> pd.DataFrame:
    """Map ratings rows onto the team IDs used by the tournament matchup file."""
    matchup_team_ids = matchup_df.rename(
        columns={"YEAR": "Season", "TEAM NO": "TournamentTeamID", "TEAM": "TeamName"}
    )[["Season", "TournamentTeamID", "TeamName"]].copy()
    matchup_team_ids["Season"] = pd.to_numeric(matchup_team_ids["Season"], errors="coerce")
    matchup_team_ids["TournamentTeamID"] = pd.to_numeric(
        matchup_team_ids["TournamentTeamID"], errors="coerce"
    )
    matchup_team_ids = matchup_team_ids.dropna(subset=["Season", "TournamentTeamID", "TeamName"]).copy()
    matchup_team_ids["Season"] = matchup_team_ids["Season"].astype(int)
    matchup_team_ids["TournamentTeamID"] = matchup_team_ids["TournamentTeamID"].astype(int)
    matchup_team_ids["_normalized_team_name"] = matchup_team_ids["TeamName"].astype(str).map(normalize_team_name)
    matchup_team_ids = matchup_team_ids.drop_duplicates(
        subset=["Season", "_normalized_team_name"]
    )[["Season", "TournamentTeamID", "_normalized_team_name"]]

    remapped_ratings = ratings_df.copy()
    remapped_ratings["_normalized_team_name"] = remapped_ratings["TeamName"].astype(str).map(normalize_team_name)
    remapped_ratings = remapped_ratings.merge(
        matchup_team_ids,
        on=["Season", "_normalized_team_name"],
        how="left",
    )

    matched_mask = remapped_ratings["TournamentTeamID"].notna()
    remapped_ratings.loc[matched_mask, "TeamID"] = remapped_ratings.loc[
        matched_mask, "TournamentTeamID"
    ].astype(int)
    remapped_ratings = remapped_ratings.drop(columns=["TournamentTeamID", "_normalized_team_name"])
    remapped_ratings = remapped_ratings.drop_duplicates(subset=["Season", "TeamID"]).sort_values(
        ["Season", "TeamID"]
    )
    return remapped_ratings.reset_index(drop=True)


def build_teams_table(ratings_df: pd.DataFrame, matchup_df: pd.DataFrame) -> pd.DataFrame:
    """Create teams.csv from the archive data."""
    matchup_teams = matchup_df.rename(columns={"TEAM NO": "TeamID", "TEAM": "TeamName"})[
        ["TeamID", "TeamName"]
    ].copy()
    ratings_teams = ratings_df[["TeamID", "TeamName"]].copy()
    teams_df = pd.concat([ratings_teams, matchup_teams], ignore_index=True)
    teams_df["TeamID"] = pd.to_numeric(teams_df["TeamID"], errors="coerce")
    teams_df = teams_df.dropna(subset=["TeamID", "TeamName"]).copy()
    teams_df["TeamID"] = teams_df["TeamID"].astype(int)
    teams_df["TeamName"] = teams_df["TeamName"].astype(str)
    teams_df = teams_df.drop_duplicates(subset=["TeamID"]).sort_values("TeamID").reset_index(drop=True)
    return teams_df


def format_seed_values(seed_series: pd.Series) -> pd.Series:
    """Convert numeric-looking seeds into the simple string format used by the project."""
    numeric_seed = pd.to_numeric(seed_series, errors="coerce")
    formatted_seed = numeric_seed.astype("Int64").astype(str).str.zfill(2)
    return "R" + formatted_seed


def build_seeds_table(matchup_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Create seeds.csv using tournament data plus ratings-based future seeds."""
    require_columns(
        matchup_df,
        ["YEAR", "TEAM NO", "SEED"],
        "Tournament Matchups.csv",
    )
    historical_seeds_df = matchup_df.rename(columns={"YEAR": "Season", "TEAM NO": "TeamID", "SEED": "Seed"})[
        ["Season", "TeamID", "Seed"]
    ].copy()
    historical_seeds_df["Season"] = pd.to_numeric(historical_seeds_df["Season"], errors="coerce")
    historical_seeds_df["TeamID"] = pd.to_numeric(historical_seeds_df["TeamID"], errors="coerce")
    historical_seeds_df["Seed"] = pd.to_numeric(historical_seeds_df["Seed"], errors="coerce")
    historical_seeds_df = historical_seeds_df.dropna(subset=["Season", "TeamID", "Seed"]).copy()
    historical_seeds_df["Season"] = historical_seeds_df["Season"].astype(int)
    historical_seeds_df["TeamID"] = historical_seeds_df["TeamID"].astype(int)
    historical_seeds_df["Seed"] = format_seed_values(historical_seeds_df["Seed"])

    future_seeds_df = pd.DataFrame(columns=["Season", "TeamID", "Seed"])
    if "Seed" in ratings_df.columns:
        future_seeds_df = ratings_df[["Season", "TeamID", "Seed"]].copy()
        future_seeds_df["Season"] = pd.to_numeric(future_seeds_df["Season"], errors="coerce")
        future_seeds_df["TeamID"] = pd.to_numeric(future_seeds_df["TeamID"], errors="coerce")
        future_seeds_df["Seed"] = pd.to_numeric(future_seeds_df["Seed"], errors="coerce")
        future_seeds_df = future_seeds_df.dropna(subset=["Season", "TeamID", "Seed"]).copy()
        future_seeds_df["Season"] = future_seeds_df["Season"].astype(int)
        future_seeds_df["TeamID"] = future_seeds_df["TeamID"].astype(int)
        future_seeds_df["Seed"] = format_seed_values(future_seeds_df["Seed"])

    seeds_df = pd.concat([historical_seeds_df, future_seeds_df], ignore_index=True)
    seeds_df = seeds_df.drop_duplicates(subset=["Season", "TeamID"]).sort_values(["Season", "TeamID"])
    return seeds_df.reset_index(drop=True)


def build_tournament_results(matchup_df: pd.DataFrame) -> pd.DataFrame:
    """Create tournament_results.csv by pairing adjacent rows within each season and round.

    The archive matchup file is ordered so that each game appears as two
    consecutive rows: the two teams in that game.
    """
    require_columns(
        matchup_df,
        ["YEAR", "TEAM NO", "CURRENT ROUND", "SCORE"],
        "Tournament Matchups.csv",
    )

    dataframe = matchup_df.copy()
    dataframe["YEAR"] = pd.to_numeric(dataframe["YEAR"], errors="coerce")
    dataframe["TEAM NO"] = pd.to_numeric(dataframe["TEAM NO"], errors="coerce")
    dataframe["CURRENT ROUND"] = pd.to_numeric(dataframe["CURRENT ROUND"], errors="coerce")
    dataframe["SCORE"] = pd.to_numeric(dataframe["SCORE"], errors="coerce")
    dataframe = dataframe.dropna(subset=["YEAR", "TEAM NO", "CURRENT ROUND", "SCORE"]).copy()
    dataframe["YEAR"] = dataframe["YEAR"].astype(int)
    dataframe["TEAM NO"] = dataframe["TEAM NO"].astype(int)
    dataframe["CURRENT ROUND"] = dataframe["CURRENT ROUND"].astype(int)
    dataframe["SCORE"] = dataframe["SCORE"].astype(int)

    output_rows: list[dict[str, int]] = []
    for season, season_df in dataframe.groupby("YEAR", sort=True):
        season_rows = season_df.reset_index(drop=True)
        if len(season_rows) % 2 != 0:
            raise DataValidationError(
                f"Tournament Matchups.csv has an odd number of rows for season {season}."
            )

        for start_index in range(0, len(season_rows), 2):
            team_one = season_rows.iloc[start_index]
            team_two = season_rows.iloc[start_index + 1]

            if team_one["CURRENT ROUND"] != team_two["CURRENT ROUND"]:
                raise DataValidationError(
                    f"Could not pair tournament rows cleanly in season {season} near row {start_index}. "
                    "Expected adjacent rows from the same round."
                )
            if team_one["SCORE"] == team_two["SCORE"]:
                raise DataValidationError(
                    f"Found a tied score in season {season}, which should not happen in NCAA tournament results."
                )

            if team_one["SCORE"] > team_two["SCORE"]:
                winner = team_one
                loser = team_two
            else:
                winner = team_two
                loser = team_one

            output_rows.append(
                {
                    "Season": season,
                    "WTeamID": int(winner["TEAM NO"]),
                    "LTeamID": int(loser["TEAM NO"]),
                    "WScore": int(winner["SCORE"]),
                    "LScore": int(loser["SCORE"]),
                }
            )

    results_df = pd.DataFrame(output_rows).sort_values(["Season"]).reset_index(drop=True)
    return results_df


def write_output_files(
    ratings_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
    tournament_results_df: pd.DataFrame,
) -> None:
    """Write the converted data into the project's raw data folder."""
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ratings_df.to_csv(config.TEAM_RATINGS_PATH, index=False)
    teams_df.to_csv(config.TEAMS_PATH, index=False)
    seeds_df.to_csv(config.SEEDS_PATH, index=False)
    tournament_results_df.to_csv(config.TOURNAMENT_RESULTS_PATH, index=False)


def convert_archive(archive_dir: Path) -> None:
    """Run the full archive conversion process."""
    ensure_project_directories()
    print(f"Reading archive files from: {archive_dir}")

    ratings_source_df = read_archive_csv(archive_dir, "KenPom Barttorvik.csv")
    matchup_df = read_archive_csv(archive_dir, "Tournament Matchups.csv")

    ratings_df = build_team_ratings(ratings_source_df)
    ratings_df = map_ratings_to_tournament_team_ids(ratings_df, matchup_df)
    teams_df = build_teams_table(ratings_df, matchup_df)
    seeds_df = build_seeds_table(matchup_df, ratings_df)
    tournament_results_df = build_tournament_results(matchup_df)

    write_output_files(ratings_df, teams_df, seeds_df, tournament_results_df)

    print("Archive conversion complete.")
    print(f"Wrote: {config.TEAM_RATINGS_PATH}")
    print(f"Wrote: {config.TEAMS_PATH}")
    print(f"Wrote: {config.SEEDS_PATH}")
    print(f"Wrote: {config.TOURNAMENT_RESULTS_PATH}")
    print(f"Ratings rows: {len(ratings_df)}")
    print(f"Teams rows: {len(teams_df)}")
    print(f"Seeds rows: {len(seeds_df)}")
    print(f"Tournament result rows: {len(tournament_results_df)}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert a custom Kaggle March Madness archive into project CSV files."
    )
    parser.add_argument(
        "--archive-dir",
        default=str(DEFAULT_ARCHIVE_DIR),
        help="Path to the extracted archive folder",
    )
    return parser.parse_args()


def main() -> None:
    """Command line entry point."""
    args = parse_args()
    try:
        convert_archive(Path(args.archive_dir).expanduser())
    except DataValidationError as exc:
        print(exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
