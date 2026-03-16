"""Beginner-friendly Streamlit interface for March Madness experiments."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.load_data import DataValidationError, load_team_ratings
from src.predict_matchups import get_matchup_prediction_details, load_selected_model
from src.simulate_bracket import run_bracket_simulation


st.set_page_config(page_title="March Madness Predictor", layout="wide")


@st.cache_data
def load_ratings_table() -> pd.DataFrame:
    """Load season-level ratings for team dropdowns and summaries."""
    return load_team_ratings()


@st.cache_data
def list_bracket_files() -> list[str]:
    """Return bracket CSV files available in data/raw."""
    return sorted(
        str(path.relative_to(config.PROJECT_ROOT))
        for path in config.RAW_DATA_DIR.glob("*.csv")
        if "bracket" in path.name.lower()
    )


@st.cache_data
def load_bracket_preview(relative_path: str) -> pd.DataFrame:
    """Load one bracket file for preview inside the app."""
    return pd.read_csv(config.PROJECT_ROOT / relative_path)


@st.cache_data
def discover_loadable_models() -> tuple[list[str], dict[str, str]]:
    """Return models that can actually be loaded in the current environment."""
    models: list[str] = []
    errors: dict[str, str] = {}
    for model_name in ["logistic", "xgboost"]:
        try:
            load_selected_model(model_name)
            models.append(model_name)
        except BaseException as exc:
            errors[model_name] = str(exc)
    return models, errors


def model_label(model_name: str) -> str:
    """Display a short friendly label for each model option."""
    return "Logistic Regression" if model_name == "logistic" else "XGBoost"


def app_main() -> None:
    """Render the full Streamlit app."""
    st.title("March Madness Predictor")
    st.write(
        "Use the trained models to compare teams, explore features, and simulate a bracket "
        "without leaving the browser."
    )

    try:
        ratings_df = load_ratings_table()
    except (DataValidationError, FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    sidebar = st.sidebar
    sidebar.header("Settings")

    models, model_errors = discover_loadable_models()
    if not models:
        st.error("No models could be loaded in this environment.")
        if model_errors:
            st.write("Model load details:")
            st.json(model_errors)
        st.stop()
    if model_errors:
        sidebar.caption("Some models are unavailable in this environment.")

    available_seasons = sorted(ratings_df["Season"].unique().tolist(), reverse=True)
    selected_model = sidebar.selectbox(
        "Model",
        options=models,
        format_func=model_label,
    )
    selected_season = sidebar.selectbox("Season", options=available_seasons)

    season_ratings = ratings_df[ratings_df["Season"] == selected_season].copy()
    season_team_names = sorted(season_ratings["TeamName"].dropna().astype(str).unique().tolist())

    tab_matchup, tab_bracket = st.tabs(["Matchup Predictor", "Bracket Simulator"])

    with tab_matchup:
        st.subheader("Single Game Prediction")
        col_a, col_b = st.columns(2)
        with col_a:
            team_a_name = st.selectbox("Team A", options=season_team_names, index=0)
        with col_b:
            default_index = 1 if len(season_team_names) > 1 else 0
            team_b_name = st.selectbox("Team B", options=season_team_names, index=default_index)

        if team_a_name == team_b_name:
            st.warning("Choose two different teams to run a prediction.")
        elif st.button("Predict Matchup", type="primary"):
            try:
                prediction = get_matchup_prediction_details(
                    season=selected_season,
                    team_a=team_a_name,
                    team_b=team_b_name,
                    model_name=selected_model,
                )
                st.success("Prediction complete.")
                metric_a, metric_b = st.columns(2)
                with metric_a:
                    st.metric(
                        label=f"{prediction['team_a_name']} win probability",
                        value=f"{prediction['team_a_win_probability']:.1%}",
                    )
                with metric_b:
                    st.metric(
                        label=f"{prediction['team_b_name']} win probability",
                        value=f"{prediction['team_b_win_probability']:.1%}",
                    )

                st.write("Feature values used by the model:")
                feature_df = pd.DataFrame(
                    [
                        {
                            "feature": key,
                            "value": value,
                        }
                        for key, value in prediction["feature_values"].items()
                    ]
                )
                st.dataframe(feature_df, use_container_width=True, hide_index=True)

                model_metadata = prediction["metadata"]
                st.caption(
                    "Validation metrics for this saved model: "
                    f"log loss {model_metadata['metrics']['log_loss']:.4f}, "
                    f"accuracy {model_metadata['metrics']['accuracy']:.4f}, "
                    f"Brier score {model_metadata['metrics']['brier_score']:.4f}"
                )
            except (DataValidationError, FileNotFoundError, ValueError) as exc:
                st.error(str(exc))

    with tab_bracket:
        st.subheader("Monte Carlo Bracket Simulation")
        bracket_files = list_bracket_files()
        if not bracket_files:
            st.info("No bracket CSVs were found in data/raw.")
        else:
            bracket_file = st.selectbox("Bracket file", options=bracket_files)
            n_sims = st.number_input(
                "Number of simulations",
                min_value=10,
                max_value=20000,
                value=500,
                step=10,
            )
            st.write("Bracket preview:")
            st.dataframe(load_bracket_preview(bracket_file), use_container_width=True, hide_index=True)

            if st.button("Run Bracket Simulation", type="primary"):
                try:
                    with st.spinner("Running simulation... this may take a little while."):
                        results_df = run_bracket_simulation(
                            season=int(selected_season),
                            bracket_file=str(config.PROJECT_ROOT / bracket_file),
                            model_name=selected_model,
                            n_sims=int(n_sims),
                        )
                    st.success("Simulation complete.")
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        label="Download results CSV",
                        data=results_df.to_csv(index=False).encode("utf-8"),
                        file_name="bracket_simulation_results.csv",
                        mime="text/csv",
                    )
                except (DataValidationError, FileNotFoundError, ValueError) as exc:
                    st.error(str(exc))


if __name__ == "__main__":
    app_main()
