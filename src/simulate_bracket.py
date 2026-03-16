"""Simulate a bracket many times and estimate advancement probabilities."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .load_data import DataValidationError, read_csv_checked
from .predict_matchups import build_matchup_features, load_selected_model, prepare_team_context
from .utils import get_team_name_for_season, resolve_team_identifier_for_season


ROUND_LABELS = {
    64: "Round of 32",
    32: "Sweet 16",
    16: "Elite 8",
    8: "Final Four",
    4: "Championship Game",
    2: "Champion",
}


def parse_play_in_placeholder(team_entry: str) -> tuple[str, str] | None:
    """Parse placeholders like 'Winner of Team A vs Team B'."""
    prefix = "Winner of "
    if not team_entry.startswith(prefix):
        return None

    matchup_text = team_entry[len(prefix) :]
    parts = matchup_text.split(" vs ")
    if len(parts) != 2:
        raise ValueError(
            f"Could not parse play-in placeholder '{team_entry}'. "
            "Use the format 'Winner of Team A vs Team B'."
        )
    return parts[0].strip(), parts[1].strip()


def read_bracket_csv(bracket_file: str) -> pd.DataFrame:
    """Read the bracket CSV using a normal Path-based flow."""
    return read_csv_checked(
        file_path=Path(bracket_file),
        required_columns=["Season", "Slot", "Round", "Team1", "Team2"],
    )


def predict_game_probability(
    season: int,
    team1_id: int,
    team2_id: int,
    model,
    feature_columns: list[str],
    ratings_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
) -> float:
    """Predict Team 1's win probability against Team 2."""
    features = build_matchup_features(
        season=season,
        team_a_id=team1_id,
        team_b_id=team2_id,
        ratings_df=ratings_df,
        seeds_df=seeds_df,
        feature_columns=feature_columns,
    )
    return float(model.predict_proba(features)[0, 1])


def resolve_team_entry(
    team_entry: str,
    season: int,
    model,
    feature_columns: list[str],
    ratings_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    rng: np.random.Generator | None = None,
) -> int:
    """Resolve one bracket entry into a TeamID, including First Four placeholders."""
    play_in_matchup = parse_play_in_placeholder(team_entry)
    if play_in_matchup is None:
        return resolve_team_identifier_for_season(team_entry, season, teams_df, ratings_df)

    if rng is None:
        raise ValueError("A random generator is required to resolve play-in placeholders.")

    team1_name, team2_name = play_in_matchup
    team1_id = resolve_team_identifier_for_season(team1_name, season, teams_df, ratings_df)
    team2_id = resolve_team_identifier_for_season(team2_name, season, teams_df, ratings_df)
    win_probability = predict_game_probability(
        season=season,
        team1_id=team1_id,
        team2_id=team2_id,
        model=model,
        feature_columns=feature_columns,
        ratings_df=ratings_df,
        seeds_df=seeds_df,
    )
    return team1_id if rng.random() < win_probability else team2_id


def extract_possible_team_ids(
    team_entry: str,
    season: int,
    ratings_df: pd.DataFrame,
    teams_df: pd.DataFrame,
) -> set[int]:
    """Return all TeamIDs that could appear for one bracket entry."""
    play_in_matchup = parse_play_in_placeholder(team_entry)
    if play_in_matchup is None:
        return {resolve_team_identifier_for_season(team_entry, season, teams_df, ratings_df)}

    team1_name, team2_name = play_in_matchup
    return {
        resolve_team_identifier_for_season(team1_name, season, teams_df, ratings_df),
        resolve_team_identifier_for_season(team2_name, season, teams_df, ratings_df),
    }


def simulate_single_bracket(
    first_round_games: list[tuple[str, str]],
    season: int,
    model,
    feature_columns: list[str],
    ratings_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, set[int]]:
    """Simulate one bracket by repeatedly pairing adjacent winners."""
    round_tracking: dict[str, set[int]] = defaultdict(set)
    current_teams: list[tuple[int, int]] = []
    for team1_entry, team2_entry in first_round_games:
        team1_id = resolve_team_entry(
            team_entry=team1_entry,
            season=season,
            model=model,
            feature_columns=feature_columns,
            ratings_df=ratings_df,
            seeds_df=seeds_df,
            teams_df=teams_df,
            rng=rng,
        )
        team2_id = resolve_team_entry(
            team_entry=team2_entry,
            season=season,
            model=model,
            feature_columns=feature_columns,
            ratings_df=ratings_df,
            seeds_df=seeds_df,
            teams_df=teams_df,
            rng=rng,
        )
        current_teams.append((team1_id, team2_id))
    current_round = len(first_round_games) * 2

    while current_teams:
        winners: list[int] = []
        for team1_id, team2_id in current_teams:
            win_probability = predict_game_probability(
                season=season,
                team1_id=team1_id,
                team2_id=team2_id,
                model=model,
                feature_columns=feature_columns,
                ratings_df=ratings_df,
                seeds_df=seeds_df,
            )
            winner = team1_id if rng.random() < win_probability else team2_id
            winners.append(winner)

        next_round_label = ROUND_LABELS.get(current_round)
        if next_round_label:
            for winner in winners:
                round_tracking[next_round_label].add(winner)

        if len(winners) == 1:
            break

        next_games: list[tuple[int, int]] = []
        for index in range(0, len(winners), 2):
            next_games.append((winners[index], winners[index + 1]))

        current_teams = next_games
        current_round = current_round // 2

    return round_tracking


def run_bracket_simulation(
    season: int,
    bracket_file: str,
    model_name: str,
    n_sims: int,
) -> pd.DataFrame:
    """Run many simulations and save advancement probabilities."""
    bracket_df = read_bracket_csv(bracket_file)
    bracket_df = bracket_df[bracket_df["Season"] == season].copy()
    if bracket_df.empty:
        raise ValueError(f"No bracket rows found for season {season} in {bracket_file}.")
    if len(bracket_df) % 2 != 0:
        raise ValueError("The bracket file must contain an even number of first-round games.")
    if len(bracket_df) & (len(bracket_df) - 1) != 0:
        raise ValueError(
            "The bracket file should contain a power-of-two number of first-round games "
            "(for example 4, 8, 16, or 32)."
        )
    bracket_df = bracket_df.sort_values("Slot").reset_index(drop=True)

    ratings_df, seeds_df, teams_df = prepare_team_context()
    model, metadata = load_selected_model(model_name)
    feature_columns = metadata["feature_columns"]

    first_round_games: list[tuple[str, str]] = []
    all_team_ids: set[int] = set()
    for row in bracket_df.itertuples(index=False):
        team1_entry = str(row.Team1)
        team2_entry = str(row.Team2)
        first_round_games.append((team1_entry, team2_entry))
        all_team_ids.update(extract_possible_team_ids(team1_entry, season, ratings_df, teams_df))
        all_team_ids.update(extract_possible_team_ids(team2_entry, season, ratings_df, teams_df))

    counts: dict[int, dict[str, int]] = {
        team_id: {label: 0 for label in ROUND_LABELS.values()} for team_id in all_team_ids
    }

    rng = np.random.default_rng(config.RANDOM_SEED)
    print(f"Running {n_sims} bracket simulations...")
    for simulation_index in range(n_sims):
        if (simulation_index + 1) % max(1, n_sims // 10) == 0:
            print(f"Completed {simulation_index + 1} / {n_sims} simulations")

        result = simulate_single_bracket(
            first_round_games=first_round_games,
            season=season,
            model=model,
            feature_columns=feature_columns,
            ratings_df=ratings_df,
            seeds_df=seeds_df,
            teams_df=teams_df,
            rng=rng,
        )
        for round_label, team_ids in result.items():
            for team_id in team_ids:
                counts[team_id][round_label] += 1

    output_rows = []
    for team_id in sorted(all_team_ids):
        row = {
            "Season": season,
            "TeamID": team_id,
            "TeamName": get_team_name_for_season(team_id, season, ratings_df, teams_df=teams_df),
        }
        for round_label in ROUND_LABELS.values():
            row[round_label] = counts[team_id][round_label] / n_sims
        output_rows.append(row)

    results_df = pd.DataFrame(output_rows).sort_values("Champion", ascending=False).reset_index(drop=True)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(config.BRACKET_RESULTS_PATH, index=False)

    print(f"Saved simulation results to: {config.BRACKET_RESULTS_PATH}")
    return results_df


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simulate a March Madness bracket.")
    parser.add_argument("--season", type=int, required=True, help="Season year, for example 2024")
    parser.add_argument(
        "--bracket-file",
        default=str(config.DEFAULT_BRACKET_PATH),
        help="Path to a CSV containing first-round bracket matchups",
    )
    parser.add_argument(
        "--model",
        default="logistic",
        choices=["logistic", "xgboost"],
        help="Which saved model to use",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=config.DEFAULT_SIMULATION_COUNT,
        help="Number of Monte Carlo simulations to run",
    )
    return parser.parse_args()


def main() -> None:
    """Command line entry point."""
    args = parse_args()
    try:
        run_bracket_simulation(
            season=args.season,
            bracket_file=args.bracket_file,
            model_name=args.model,
            n_sims=args.n_sims,
        )
    except (DataValidationError, FileNotFoundError, ValueError) as exc:
        print(exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
