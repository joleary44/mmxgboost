# March Madness Prediction Project

This project is a beginner-friendly Python pipeline for predicting NCAA men's March Madness game outcomes.
It uses:

- Historical tournament game results from CSV files
- Team strength metrics from a manually provided ratings CSV
- A logistic regression baseline built with scikit-learn
- An optional XGBoost model
- Monte Carlo simulation to estimate bracket advancement probabilities

The code is designed to be simple, readable, and easy to run from the command line.

## Project Structure

```text
march_madness_model/
  data/
    raw/
    processed/
  models/
  notebooks/
  outputs/
  src/
    __init__.py
    config.py
    feature_engineering.py
    load_data.py
    predict_matchups.py
    simulate_bracket.py
    train_logistic.py
    train_xgboost.py
    utils.py
  README.md
  requirements.txt
```

## What You Need To Provide

Place your input CSV files in:

`march_madness_model/data/raw/`

Expected files:

- `regular_season_results.csv`
- `tournament_results.csv`
- `seeds.csv`
- `teams.csv`
- `team_ratings.csv`

For the current baseline:

- `tournament_results.csv`, `seeds.csv`, `teams.csv`, and `team_ratings.csv` are required.
- `regular_season_results.csv` is supported and loaded safely, but it is not required until you add regular-season features later.

There is also a sample bracket file included:

- `sample_bracket.csv`

## Expected Input Columns

### 1. `regular_season_results.csv`

Required columns:

- `Season`
- `WTeamID`
- `LTeamID`
- `WScore`
- `LScore`

Optional:

- `NumOT`

### 2. `tournament_results.csv`

Required columns:

- `Season`
- `WTeamID`
- `LTeamID`
- `WScore`
- `LScore`

Optional:

- `NumOT`

### 3. `seeds.csv`

Required columns:

- `Season`
- `TeamID`
- `Seed`

### 4. `teams.csv`

Required columns:

- `TeamID`
- `TeamName`

### 5. `team_ratings.csv`

Standard required columns:

- `Season`
- `TeamID`
- `AdjO`
- `AdjD`
- `Tempo`
- `Rating`

Optional:

- `TeamName`
- `SOS`
- `Luck`

The loader also accepts common exported names from KenPom-style files:

- `AdjEM` can be used for `Rating`
- `AdjOE` can be used for `AdjO`
- `AdjDE` can be used for `AdjD`
- `AdjT` can be used for `Tempo`
- `Team` or `School` can be used for `TeamName`

If `TeamID` is missing, the project will try to match `TeamName` to [`teams.csv`](/Users/jamesoleary/Documents/New%20project/march_madness_model/data/raw/teams.csv) automatically.
This is helpful for exported ranking sheets that list only team names.

## Beginner Setup

### 1. Open a terminal in the project folder

```bash
cd "/Users/jamesoleary/Documents/New project/march_madness_model"
```

### 2. Create a virtual environment

Mac/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Put your CSV files into `data/raw/`

Your folder should look something like this:

```text
data/raw/
  regular_season_results.csv
  tournament_results.csv
  seeds.csv
  teams.csv
  team_ratings.csv
  sample_bracket.csv
```

If your original file is an Apple Numbers sheet like `kenpomrank.numbers`, export it to CSV first and save the exported file as `team_ratings.csv`.

## Exact Run Commands

### Optional: launch the Streamlit app

If you want a browser-based interface instead of using the terminal, run:

```bash
streamlit run streamlit_app.py
```

The app lets you:

- choose a saved model
- pick teams from dropdowns
- run matchup predictions
- run bracket simulations
- download simulation results as a CSV

### Deploy on Streamlit Community Cloud

If your repository is public, you can deploy this app for free on Streamlit Community Cloud.

1. Push this project to a public GitHub repository.
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Sign in with GitHub.
4. Click to create a new app.
5. Select your repository and branch.
6. Set the main file path to:

```text
streamlit_app.py
```

7. Deploy the app.

Because this project keeps its CSV data and trained model files inside the repository,
the deployed app can use them directly after the repo is pushed.

### Optional: convert a custom Kaggle archive automatically

If you downloaded the custom archive that contains files like `KenPom Barttorvik.csv`
and `Tournament Matchups.csv`, you can convert it into the standard project files with:

```bash
python -m src.prepare_custom_kaggle_archive --archive-dir ~/Downloads/archive
```

This writes:

- `data/raw/team_ratings.csv`
- `data/raw/teams.csv`
- `data/raw/seeds.csv`
- `data/raw/tournament_results.csv`

After that, continue with the normal pipeline below.

### Step 1. Build the modeling dataset

```bash
python -m src.feature_engineering
```

This creates:

- `data/processed/model_games.csv`

### Step 2. Train the logistic regression baseline

```bash
python -m src.train_logistic
```

This creates:

- `models/logistic_model.joblib`
- `models/logistic_metadata.json`

### Step 3. Optionally train the XGBoost model

```bash
python -m src.train_xgboost
```

If `xgboost` is not installed, the script will print a helpful message and exit safely.

This may create:

- `models/xgboost_model.joblib`
- `models/xgboost_metadata.json`

### Step 4. Predict a single matchup

Example:

```bash
python -m src.predict_matchups --season 2024 --team-a "Houston" --team-b "Duke" --model logistic
```

You can also use numeric team IDs:

```bash
python -m src.predict_matchups --season 2024 --team-a 248 --team-b 1181 --model logistic
```

### Step 5. Simulate a bracket with Monte Carlo

Example:

```bash
python -m src.simulate_bracket --season 2024 --bracket-file data/raw/sample_bracket.csv --model logistic --n-sims 5000
```

This creates:

- `outputs/bracket_simulation_results.csv`

Important:

- The included `sample_bracket.csv` is a small template to show the format.
- For a real March Madness simulation, replace it with your full first-round bracket file.

## Sample Bracket Format

The simulator expects a CSV with one row per first-round game.

Required columns:

- `Season`
- `Slot`
- `Round`
- `Team1`
- `Team2`

Example:

```csv
Season,Slot,Round,Team1,Team2
2024,R1W1,64,UConn,Stetson
2024,R1W2,64,Florida Atlantic,Northwestern
```

Notes:

- `Team1` and `Team2` can be team names or numeric team IDs.
- First Four placeholders are also supported in the format `Winner of Team A vs Team B`.
- The simulator automatically builds later rounds from the first-round winners.
- The included sample file is only a format example. Replace it with the real tournament bracket you want to simulate.

## How The Modeling Dataset Is Built

Each row represents one historical tournament game.

To avoid positional bias:

- The pipeline creates two rows per actual game.
- In one row, Team A is the winner and `result = 1`.
- In the mirrored row, Team A is the loser and `result = 0`.

This keeps the training data balanced around team ordering and helps avoid the model learning that “Team A” wins simply because of row position.

Features are season-level differences between Team A and Team B, such as:

- `rating_diff`
- `adjo_diff`
- `adjd_diff`
- `tempo_diff`
- `seed_diff`
- `sos_diff` if available
- `luck_diff` if available

## Train / Validation Split

The project uses a season-based split from `src/config.py`.

This is important because a random row split would mix games from the same season into both train and validation sets, which can lead to overly optimistic evaluation.

## Output Files

### Processed data

- `data/processed/model_games.csv`

### Models

- `models/logistic_model.joblib`
- `models/logistic_metadata.json`
- `models/xgboost_model.joblib` if trained
- `models/xgboost_metadata.json` if trained

### Simulation results

- `outputs/bracket_simulation_results.csv`

## Troubleshooting

### “File not found” error

Make sure your CSVs are in:

`march_madness_model/data/raw/`

### “Missing required columns” error

Check your column names carefully. They must match the names listed above.

### “No rows available after feature engineering”

This usually means:

- Ratings are missing for some teams or seasons
- Seeds are missing
- The input seasons in your data do not overlap well

### “Model file not found” when predicting or simulating

Train the model first:

```bash
python -m src.train_logistic
```

### “xgboost is not installed”

You can still use the baseline logistic regression model.

To install XGBoost manually:

```bash
pip install xgboost
```

## File Guide

### [`src/config.py`](/Users/jamesoleary/Documents/New project/march_madness_model/src/config.py)

Stores file paths, feature names, train/validation seasons, and default settings.

### [`src/load_data.py`](/Users/jamesoleary/Documents/New project/march_madness_model/src/load_data.py)

Loads CSV files safely and validates columns.

### [`src/feature_engineering.py`](/Users/jamesoleary/Documents/New project/march_madness_model/src/feature_engineering.py)

Builds the tournament modeling dataset and saves it to disk.

### [`src/prepare_custom_kaggle_archive.py`](/Users/jamesoleary/Documents/New project/march_madness_model/src/prepare_custom_kaggle_archive.py)

Converts a custom Kaggle archive folder into the raw CSV files expected by the rest of the project.

### [`src/train_logistic.py`](/Users/jamesoleary/Documents/New project/march_madness_model/src/train_logistic.py)

Trains and evaluates the logistic regression baseline.

### [`src/train_xgboost.py`](/Users/jamesoleary/Documents/New project/march_madness_model/src/train_xgboost.py)

Trains an upgraded XGBoost model if the package is available.

### [`src/predict_matchups.py`](/Users/jamesoleary/Documents/New project/march_madness_model/src/predict_matchups.py)

Predicts the probability that Team A beats Team B for a chosen season.

### [`src/simulate_bracket.py`](/Users/jamesoleary/Documents/New project/march_madness_model/src/simulate_bracket.py)

Runs Monte Carlo bracket simulations and outputs advancement probabilities.

### [`src/utils.py`](/Users/jamesoleary/Documents/New project/march_madness_model/src/utils.py)

Contains helper utilities for saving models, parsing seeds, and looking up teams.

### [`streamlit_app.py`](/Users/jamesoleary/Documents/New project/march_madness_model/streamlit_app.py)

Provides a beginner-friendly browser app for matchup predictions and bracket simulations.

## Next Upgrades

Good improvements after the baseline is working:

1. Add regular-season derived features such as win percentage and average margin.
2. Add conference information or home/away-neutral context if your data has it.
3. Tune hyperparameters with time-aware validation across multiple seasons.
4. Build a notebook for exploratory analysis and feature inspection.
5. Extend the simulator to support full official bracket slot logic.
