"""
Uber Fare Prediction - Streamlit Application.

Logic aligned with user's notebook pipeline.
"""

from __future__ import annotations

from datetime import datetime, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DATA_PATH = Path("uber.csv")
BEST_MODEL_PATH = Path("best_model.pkl")
NYC_MIN_FARE = 2.50

JFK_COORDS = (40.6413, -73.7781)
LGA_COORDS = (40.7769, -73.8740)
EWR_COORDS = (40.6895, -74.1745)

# ── Matches notebook's X = df[[...]] exactly ──────────────────────────────────
FEATURE_COLUMNS = [
    "distance_km",
    "passenger_count",
    "hour",
    "day_of_week",
    "month",
    "year",
    "is_rush_hour",
    "is_weekend",
    "is_night",
]

FAMOUS_LOCATIONS = {
    "Times Square": (40.7580, -73.9855),
    "JFK Airport": JFK_COORDS,
    "LaGuardia Airport": LGA_COORDS,
    "Grand Central": (40.7527, -73.9772),
    "Brooklyn Bridge": (40.7061, -73.9969),
    "Central Park": (40.7812, -73.9665),
    "Wall Street": (40.7064, -74.0094),
    "Custom": None,
}


# ── Haversine — same formula as notebook ──────────────────────────────────────
def haversine_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    a = np.clip(a, 0, 1)
    return 2 * R * np.arcsin(np.sqrt(a))


# ── Data loading — matches notebook cleaning steps exactly ────────────────────
@st.cache_data(show_spinner=True)
def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop unused columns (notebook cell: df.drop(columns=['Unnamed: 0', 'key']))
    for col in ("Unnamed: 0", "key"):
        if col in df.columns:
            df = df.drop(columns=[col])

    # Drop rows with missing coordinates (notebook: dropna on coordinate cols)
    gps_cols = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
    df = df.dropna(subset=gps_cols)

    return df.reset_index(drop=True)


# ── Feature engineering — matches notebook cells exactly ─────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Distance (notebook: haversine with lat1, lng1, lat2, lng2 order)
    out["distance_km"] = haversine_km(
        out["pickup_latitude"].values,
        out["pickup_longitude"].values,
        out["dropoff_latitude"].values,
        out["dropoff_longitude"].values,
    )

    # Datetime parsing (notebook: pd.to_datetime with utc=True)
    out["pickup_datetime"] = pd.to_datetime(out["pickup_datetime"], errors="coerce", utc=True)
    out = out.dropna(subset=["pickup_datetime"])

    # Time features (notebook column names: hour, day_of_week, month, year)
    out["hour"]        = out["pickup_datetime"].dt.hour
    out["day_of_week"] = out["pickup_datetime"].dt.dayofweek   # 0=Monday, 6=Sunday
    out["month"]       = out["pickup_datetime"].dt.month
    out["year"]        = out["pickup_datetime"].dt.year

    # Binary flags (notebook: is_rush_hour, is_night, is_weekend)
    out["is_rush_hour"] = (
        (out["hour"].between(7, 9)) | (out["hour"].between(17, 19))
    ).astype(int)
    out["is_night"]   = ((out["hour"] >= 20) | (out["hour"] <= 5)).astype(int)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

    return out


# ── Outlier removal — matches notebook thresholds exactly ─────────────────────
def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    # Passenger count: between 1 and 6 (notebook cell)
    df = df[df["passenger_count"].between(1, 6)]

    # Distance: between 0.1 and 99th percentile (notebook cell)
    upper_dist = df["distance_km"].quantile(0.99)
    df = df[df["distance_km"].between(0.1, upper_dist)]

    # Fare: between $2.50 and $100 (notebook cell)
    df = df[df["fare_amount"].between(2.5, 100)]

    return df.reset_index(drop=True)


# ── Model training — no scaler, matches notebook pipeline exactly ──────────────
@st.cache_resource(show_spinner=True)
def train_models(data_path: str):
    df_clean = load_and_clean_data(data_path)
    df_feat  = engineer_features(df_clean)
    df_feat  = remove_outliers(df_feat)

    X = df_feat[FEATURE_COLUMNS].copy()
    y = df_feat["fare_amount"].copy()

    # 80/20 split, random_state=42 (notebook cell)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # No StandardScaler — notebook does not use one
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, random_state=42
        ),
    }

    results: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    fitted_models: dict[str, object] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)          # no scaling
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        fitted_models[name] = model
        results.append({
            "Model": name,
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "MAE":  float(mean_absolute_error(y_test, y_pred)),
            "R2":   float(r2_score(y_test, y_pred)),
        })

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    best_name  = str(results_df.iloc[0]["Model"])
    best_model = fitted_models[best_name]
    best_rmse  = float(results_df.iloc[0]["RMSE"])
    joblib.dump(best_model, BEST_MODEL_PATH)

    # Feature importance from the best tree-based model
    tree_candidates = [m for m in ("RandomForestRegressor", "GradientBoostingRegressor") if m in fitted_models]
    best_tree_name  = min(
        tree_candidates,
        key=lambda n: float(results_df.loc[results_df["Model"] == n, "RMSE"].iloc[0]),
    )
    best_tree_model = fitted_models[best_tree_name]
    feature_importance_df = pd.DataFrame({
        "feature":    FEATURE_COLUMNS,
        "importance": best_tree_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "clean_df":             df_feat,        # post-outlier-removal df for visualisations
        "feature_df":           df_feat,
        "y_test":               y_test,
        "best_model_name":      best_name,
        "best_model":           best_model,
        "best_rmse":            best_rmse,
        "results_df":           results_df,
        "predictions":          predictions,
        "feature_importance_df": feature_importance_df,
    }


# ── Single-trip feature builder — no scaler, matches notebook columns ─────────
def derive_single_trip_features(
    pickup_lat: float,
    pickup_lng: float,
    dropoff_lat: float,
    dropoff_lng: float,
    passenger_count: int,
    pickup_dt: datetime,
) -> pd.DataFrame:
    distance_km = float(
        haversine_km(
            np.array([pickup_lat]),
            np.array([pickup_lng]),
            np.array([dropoff_lat]),
            np.array([dropoff_lng]),
        )[0]
    )

    hour        = pickup_dt.hour
    day_of_week = pickup_dt.weekday()
    month       = pickup_dt.month
    year        = pickup_dt.year
    is_weekend  = int(day_of_week >= 5)
    is_rush_hour = int((hour >= 7 and hour <= 9) or (hour >= 17 and hour <= 19))
    is_night    = int((hour >= 20) or (hour <= 5))

    row = pd.DataFrame([{
        "distance_km":     distance_km,
        "passenger_count": int(passenger_count),
        "hour":            hour,
        "day_of_week":     day_of_week,
        "month":           month,
        "year":            year,
        "is_rush_hour":    is_rush_hour,
        "is_weekend":      is_weekend,
        "is_night":        is_night,
    }])
    return row[FEATURE_COLUMNS]


# ── Map rendering — unchanged from Cursor ─────────────────────────────────────
def render_map(pickup_lat: float, pickup_lng: float, dropoff_lat: float, dropoff_lng: float) -> None:
    points = pd.DataFrame([
        {"lat": pickup_lat,  "lng": pickup_lng,  "name": "Pickup",  "color": [0, 102, 255]},
        {"lat": dropoff_lat, "lng": dropoff_lng, "name": "Dropoff", "color": [255, 0, 0]},
    ])
    route = pd.DataFrame([{"path": [[pickup_lng, pickup_lat], [dropoff_lng, dropoff_lat]]}])

    scatter_layer = pdk.Layer("ScatterplotLayer", data=points, get_position="[lng, lat]", get_fill_color="color", get_radius=180)
    line_layer    = pdk.Layer("PathLayer", data=route, get_path="path", get_width=4, width_min_pixels=4, get_color=[0, 0, 0, 180])
    text_layer    = pdk.Layer("TextLayer", data=points, get_position="[lng, lat]", get_text="name", get_size=14, get_color=[30, 30, 30], get_alignment_baseline="'bottom'")

    view_state = pdk.ViewState(
        latitude=(pickup_lat + dropoff_lat) / 2,
        longitude=(pickup_lng + dropoff_lng) / 2,
        zoom=10.5,
        pitch=0,
    )
    st.pydeck_chart(pdk.Deck(layers=[line_layer, scatter_layer, text_layer], initial_view_state=view_state))


# ── Main UI — identical to Cursor's, only prediction call changed ──────────────
def main() -> None:
    st.set_page_config(page_title="Uber Fare Prediction", layout="wide")
    st.title("Uber Fare Prediction (NYC)")
    st.caption("Interactive fare estimator trained on Uber historical trips")

    if not DATA_PATH.exists():
        st.error("Dataset file not found. Place `uber.csv` in this folder and rerun.")
        st.stop()

    artifacts      = train_models(str(DATA_PATH))
    results_df     = artifacts["results_df"]
    feature_importance_df = artifacts["feature_importance_df"]
    rmse_best      = artifacts["best_rmse"]

    st.subheader("Model Evaluation")
    st.dataframe(
        results_df.style.format({"RMSE": "{:.3f}", "MAE": "{:.3f}", "R2": "{:.3f}"}),
        use_container_width=True,
    )
    st.write(f"Best model by RMSE: **{artifacts['best_model_name']}**")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.header("Trip Inputs")
    pickup_location  = st.sidebar.selectbox("Pickup location",  list(FAMOUS_LOCATIONS.keys()), index=0)
    dropoff_location = st.sidebar.selectbox("Dropoff location", list(FAMOUS_LOCATIONS.keys()), index=1)
    default_pickup   = FAMOUS_LOCATIONS[pickup_location]  or (40.7580, -73.9855)
    default_dropoff  = FAMOUS_LOCATIONS[dropoff_location] or (40.6413, -73.7781)

    pickup_lat  = st.sidebar.number_input("Pickup latitude",    min_value=40.4, max_value=41.8, value=float(default_pickup[0]),  step=0.0001, format="%.4f")
    pickup_lng  = st.sidebar.number_input("Pickup longitude",   min_value=-74.5, max_value=-72.8, value=float(default_pickup[1]), step=0.0001, format="%.4f")
    dropoff_lat = st.sidebar.number_input("Dropoff latitude",   min_value=40.4, max_value=41.8, value=float(default_dropoff[0]), step=0.0001, format="%.4f")
    dropoff_lng = st.sidebar.number_input("Dropoff longitude",  min_value=-74.5, max_value=-72.8, value=float(default_dropoff[1]), step=0.0001, format="%.4f")
    passengers  = st.sidebar.slider("Passenger count", min_value=1, max_value=6, value=2)
    pickup_date = st.sidebar.date_input("Pickup date", value=datetime(2015, 6, 17).date())
    pickup_time = st.sidebar.time_input("Pickup time", value=time(8, 0))
    pickup_dt   = datetime.combine(pickup_date, pickup_time)
    predict_clicked = st.sidebar.button("Predict Fare", type="primary", use_container_width=True)

    # ── Tabs — identical structure ─────────────────────────────────────────────
    tab_results, tab_viz = st.tabs(["Prediction & Map", "Visualizations"])

    with tab_results:
        st.subheader("Prediction Results")
        st.info("Choose locations/time in the sidebar and click **Predict Fare**.")
        if predict_clicked:
            trip_features = derive_single_trip_features(
                pickup_lat, pickup_lng, dropoff_lat, dropoff_lng, passengers, pickup_dt
            )
            # No scaler transform — notebook trains without StandardScaler
            raw_pred       = float(artifacts["best_model"].predict(trip_features)[0])
            predicted_fare = max(raw_pred, NYC_MIN_FARE)
            low  = max(predicted_fare - rmse_best, NYC_MIN_FARE)
            high = predicted_fare + rmse_best

            distance_km     = float(trip_features.loc[0, "distance_km"])
            distance_mi     = distance_km * 0.621371
            est_duration_min = (distance_km / 20.0) * 60.0

            col1, col2 = st.columns([1.2, 1])
            with col1:
                st.markdown(f"<h1 style='font-size: 3rem; margin-bottom: 0;'>${predicted_fare:,.2f}</h1>", unsafe_allow_html=True)
                st.markdown(f"**Confidence range (±1 RMSE):** `${low:,.2f} - ${high:,.2f}`")
                st.write(f"- Distance: **{distance_km:.2f} km** (**{distance_mi:.2f} miles**)")
                st.write(f"- Estimated duration (@20 km/h): **{est_duration_min:.1f} minutes**")
                st.write(f"- Rush hour: **{int(trip_features.loc[0, 'is_rush_hour'])}**")
                st.write(f"- Weekend: **{int(trip_features.loc[0, 'is_weekend'])}**")
                st.write(f"- Night: **{int(trip_features.loc[0, 'is_night'])}**")
                # Airport trip removed — not a feature in your notebook
            with col2:
                top10   = feature_importance_df.head(10).sort_values("importance", ascending=True)
                fig_imp = px.bar(top10, x="importance", y="feature", orientation="h", title="Top 10 Feature Importances")
                fig_imp.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=20))
                st.plotly_chart(fig_imp, use_container_width=True)

            st.markdown("### Route Map")
            render_map(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng)

    with tab_viz:
        st.subheader("Model and Data Visualizations")
        best_name    = artifacts["best_model_name"]
        y_test       = artifacts["y_test"]
        y_pred_best  = artifacts["predictions"][best_name]
        scatter_df   = pd.DataFrame({"Actual Fare": y_test.values, "Predicted Fare": y_pred_best})
        fig_scatter  = px.scatter(scatter_df, x="Actual Fare", y="Predicted Fare", title=f"Actual vs Predicted ({best_name})")
        max_fare     = float(max(scatter_df["Actual Fare"].max(), scatter_df["Predicted Fare"].max()))
        fig_scatter.add_trace(go.Scatter(x=[0, max_fare], y=[0, max_fare], mode="lines", name="y=x", line=dict(color="red", dash="dash")))
        st.plotly_chart(fig_scatter, use_container_width=True)

        fig_hist = px.histogram(artifacts["clean_df"], x="fare_amount", nbins=80, title="Training Data Fare Distribution")
        st.plotly_chart(fig_hist, use_container_width=True)

        sample_feat = artifacts["feature_df"].sample(n=min(12000, len(artifacts["feature_df"])), random_state=42)
        fig_fd = px.scatter(
            sample_feat,
            x="distance_km",
            y="fare_amount",
            color="hour",
            title="Fare vs Distance (colored by hour of day)",
            opacity=0.45,
        )
        st.plotly_chart(fig_fd, use_container_width=True)

        fig_rmse = px.bar(results_df, x="Model", y="RMSE", color="Model", title="RMSE Comparison Across Models", text=results_df["RMSE"].round(2))
        st.plotly_chart(fig_rmse, use_container_width=True)


if __name__ == "__main__":
    main()