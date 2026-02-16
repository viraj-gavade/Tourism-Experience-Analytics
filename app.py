import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load models & data
# -----------------------------
visit_model = joblib.load("models/visit_mode_model.pkl")
predicted_ratings = pd.read_csv("models/predicted_ratings.csv", index_col=0)
recommender_df = pd.read_csv("data/processed/recommender_dataset.csv")
master_df = pd.read_csv("data/processed/master_dataset_with_ids.csv")

# Ensure index type consistency
predicted_ratings.index = predicted_ratings.index.astype(int)

# -----------------------------
# Default realistic row
# -----------------------------
default_row = master_df.iloc[0]

# -----------------------------
# Page title
# -----------------------------
st.title("🌍 Travel Recommendation System")
st.write("Predict visit mode and get personalized attraction suggestions.")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("User Details")

user_id = st.sidebar.selectbox(
    "Select User ID",
    sorted(recommender_df["UserId"].unique())
)

visit_year = st.sidebar.slider(
    "Visit Year", 2015, 2025, int(default_row["VisitYear"])
)

visit_month = st.sidebar.slider(
    "Visit Month", 1, 12, int(default_row["VisitMonth"])
)

continent = st.sidebar.selectbox(
    "Continent ID",
    sorted(master_df["ContinentId"].dropna().unique())
)

country = st.sidebar.selectbox(
    "Country ID",
    sorted(master_df["CountryId"].dropna().unique())
)

city = st.sidebar.selectbox(
    "City ID",
    sorted(master_df["CityId"].dropna().unique())
)

attraction_type = st.sidebar.selectbox(
    "Attraction Type ID",
    sorted(master_df["AttractionTypeId"].dropna().unique())
)

# -----------------------------
# Feature engineering
# -----------------------------
month_sin = np.sin(2 * np.pi * visit_month / 12)
month_cos = np.cos(2 * np.pi * visit_month / 12)

X_input = pd.DataFrame([{
    "VisitYear": visit_year,
    "VisitMonth": visit_month,
    "ContinentId": continent,
    "CountryId": country,
    "CityId": city,
    "AttractionTypeId": attraction_type,
    "Month_sin": month_sin,
    "Month_cos": month_cos
}])

# -----------------------------
# Visit Mode Prediction
# -----------------------------
st.header("🧭 Predicted Visit Mode")

if st.button("Predict Visit Mode"):
    prediction = visit_model.predict(X_input)[0]
    st.success(f"Predicted Visit Mode ID: {prediction}")

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend_collaborative(user_id, n=5):

    user_id = int(user_id)

    # Check user exists
    if user_id not in predicted_ratings.index:
        return pd.Series(dtype=float)

    # Predicted ratings for user
    user_preds = predicted_ratings.loc[user_id]

    # Items already rated
    rated_items = recommender_df[
        recommender_df["UserId"] == user_id
    ]["AttractionId"].values

    # Keep unseen items
    unseen = user_preds.drop(rated_items, errors="ignore")

    # Return top N
    return unseen.sort_values(ascending=False).head(n)

# -----------------------------
# Show Recommendations
# -----------------------------
st.header("⭐ Recommended Attractions")

if st.button("Get Recommendations"):

    recs = recommend_collaborative(user_id)

    # ---------- Fallback if empty ----------
    if recs.empty:
        st.warning("No personalized recommendations found. Showing popular attractions instead.")

        fallback = (
            recommender_df.groupby("AttractionId")["Rating"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
        )

        for i, (attr, score) in enumerate(fallback.items(), 1):
            st.write(f"**{i}. Attraction {attr}** — Avg Rating: {score:.2f}")

    # ---------- Normal personalized ----------
    else:
        for i, (attr, score) in enumerate(recs.items(), 1):
            st.write(f"**{i}. Attraction {attr}** — Predicted Rating: {score:.2f}")

# -----------------------------
# Popular Attractions Chart
# -----------------------------
st.header("📊 Top Popular Attractions")

top_popular = (
    recommender_df.groupby("AttractionId")["Rating"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

st.bar_chart(top_popular)
