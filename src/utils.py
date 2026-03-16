"""Shared helper functions used across the project."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from . import config


def parse_seed_value(seed_text: Any) -> float:
    """Extract the numeric tournament seed from strings like 'W01' or 'X16b'."""
    if pd.isna(seed_text):
        return float("nan")

    match = re.search(r"(\d+)", str(seed_text))
    if not match:
        return float("nan")
    return float(match.group(1))


def get_available_feature_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return the configured features that actually exist in a dataframe."""
    candidate_columns = config.BASE_FEATURE_COLUMNS + config.OPTIONAL_FEATURE_COLUMNS
    return [column for column in candidate_columns if column in dataframe.columns]


def save_model_and_metadata(
    model: Any,
    model_path: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
) -> None:
    """Save a trained model and a small JSON metadata file."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def load_model_and_metadata(model_path: Path, metadata_path: Path) -> tuple[Any, dict[str, Any]]:
    """Load a saved model and metadata JSON."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Train the model first before running predictions."
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            "Train the model first before running predictions."
        )

    model = joblib.load(model_path)
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    return model, metadata


def normalize_team_name(name: str) -> str:
    """Normalize team names to make name matching more forgiving."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def build_team_lookup(teams_df: pd.DataFrame) -> dict[str, int]:
    """Create a mapping from normalized team names to team IDs."""
    lookup: dict[str, int] = {}
    for row in teams_df.itertuples(index=False):
        lookup[normalize_team_name(str(row.TeamName))] = int(row.TeamID)
    return lookup


def resolve_team_identifier(team_value: str, teams_df: pd.DataFrame) -> int:
    """Resolve either a numeric team ID or a team name string into a TeamID."""
    if str(team_value).isdigit():
        return int(team_value)

    lookup = build_team_lookup(teams_df)
    normalized = normalize_team_name(str(team_value))
    if normalized not in lookup:
        raise ValueError(
            f"Could not find team '{team_value}' in teams.csv. "
            "Check the spelling or use a numeric TeamID instead."
        )
    return lookup[normalized]


def resolve_team_identifier_for_season(
    team_value: str,
    season: int,
    teams_df: pd.DataFrame,
    season_reference_df: pd.DataFrame,
    team_id_column: str = "TeamID",
    team_name_column: str = "TeamName",
    season_column: str = "Season",
) -> int:
    """Resolve a team for one season, which avoids duplicate-name ambiguity across years."""
    if str(team_value).isdigit():
        return int(team_value)

    season_df = season_reference_df[season_reference_df[season_column] == season].copy()
    if team_name_column not in season_df.columns or team_id_column not in season_df.columns:
        return resolve_team_identifier(team_value, teams_df)

    season_lookup: dict[str, int] = {}
    for row in season_df.itertuples(index=False):
        season_lookup[normalize_team_name(str(getattr(row, team_name_column)))] = int(
            getattr(row, team_id_column)
        )

    normalized = normalize_team_name(str(team_value))
    if normalized in season_lookup:
        return season_lookup[normalized]

    return resolve_team_identifier(team_value, teams_df)


def get_team_name(team_id: int, teams_df: pd.DataFrame) -> str:
    """Return a human-readable team name for a TeamID."""
    matches = teams_df.loc[teams_df["TeamID"] == team_id, "TeamName"]
    if matches.empty:
        return str(team_id)
    return str(matches.iloc[0])


def get_team_name_for_season(
    team_id: int,
    season: int,
    season_reference_df: pd.DataFrame,
    teams_df: pd.DataFrame | None = None,
    team_id_column: str = "TeamID",
    team_name_column: str = "TeamName",
    season_column: str = "Season",
) -> str:
    """Return a team name using season-aware data first, then fall back to teams.csv."""
    season_matches = season_reference_df[
        (season_reference_df[season_column] == season)
        & (season_reference_df[team_id_column] == team_id)
    ]
    if not season_matches.empty and team_name_column in season_matches.columns:
        return str(season_matches.iloc[0][team_name_column])

    if teams_df is not None:
        return get_team_name(team_id, teams_df)
    return str(team_id)
