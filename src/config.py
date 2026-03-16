"""Project-wide configuration values.

This file keeps paths and settings in one place so beginners do not have to
search through multiple scripts to change common options.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

REGULAR_SEASON_RESULTS_PATH = RAW_DATA_DIR / "regular_season_results.csv"
TOURNAMENT_RESULTS_PATH = RAW_DATA_DIR / "tournament_results.csv"
SEEDS_PATH = RAW_DATA_DIR / "seeds.csv"
TEAMS_PATH = RAW_DATA_DIR / "teams.csv"
TEAM_RATINGS_PATH = RAW_DATA_DIR / "team_ratings.csv"
PROCESSED_MODEL_GAMES_PATH = PROCESSED_DATA_DIR / "model_games.csv"
DEFAULT_BRACKET_PATH = RAW_DATA_DIR / "sample_bracket.csv"

LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic_model.joblib"
LOGISTIC_METADATA_PATH = MODELS_DIR / "logistic_metadata.json"
XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_model.joblib"
XGBOOST_METADATA_PATH = MODELS_DIR / "xgboost_metadata.json"
BRACKET_RESULTS_PATH = OUTPUTS_DIR / "bracket_simulation_results.csv"

REQUIRED_FILES = {
    "regular_season_results": REGULAR_SEASON_RESULTS_PATH,
    "tournament_results": TOURNAMENT_RESULTS_PATH,
    "seeds": SEEDS_PATH,
    "teams": TEAMS_PATH,
    "team_ratings": TEAM_RATINGS_PATH,
}

BASE_FEATURE_COLUMNS = [
    "rating_diff",
    "adjo_diff",
    "adjd_diff",
    "tempo_diff",
    "seed_diff",
]

OPTIONAL_FEATURE_COLUMNS = [
    "sos_diff",
    "luck_diff",
]

TRAIN_SEASONS = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022]
VALIDATION_SEASONS = [2023, 2024]

RANDOM_SEED = 42
DEFAULT_SIMULATION_COUNT = 5000

TARGET_COLUMN = "result"
SEASON_COLUMN = "Season"
