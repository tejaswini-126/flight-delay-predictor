import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Flight Delay Predictor", layout="centered")

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load("models/best_model.pkl")
columns = joblib.load("models/columns.pkl")

# ==============================
# TITLE
# ==============================
st.title("✈ Flight Delay Prediction System")
st.markdown("Predict whether a flight will be delayed using Machine Learning")

st.markdown("---")

# ==============================
# INPUTS
# ==============================
st.subheader("Enter Flight Details")

# Airlines from model
airlines = [col.replace("AIRLINE_", "") for col in columns if "AIRLINE_" in col]
origins = [col.replace("ORIGIN_AIRPORT_", "") for col in columns if "ORIGIN_AIRPORT_" in col]
destinations = [col.replace("DESTINATION_AIRPORT_", "") for col in columns if "DESTINATION_AIRPORT_" in col]

airline = st.selectbox("Airline", sorted(airlines))
origin = st.selectbox("Origin Airport", sorted(origins))
destination = st.selectbox("Destination Airport", sorted(destinations))

distance = st.number_input("Distance (miles)", min_value=100, max_value=3000, value=500)
hour = st.slider("Departure Hour", 0, 23, 12)

# Day selection
day_name = st.selectbox("Day of Week",
                       ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

day_map = {
    "Monday": 1, "Tuesday": 2, "Wednesday": 3,
    "Thursday": 4, "Friday": 5,
    "Saturday": 6, "Sunday": 7
}

day = day_map[day_name]

st.markdown("---")

# ==============================
# FEATURE ENGINEERING
# ==============================
def get_time_slot(hour):
    if hour < 12:
        return "Morning"
    elif hour < 18:
        return "Afternoon"
    else:
        return "Night"

time_slot = get_time_slot(hour)
is_weekend = 1 if day >= 6 else 0

# Create full feature dataframe
input_df = pd.DataFrame(columns=columns)
input_df.loc[0] = 0

# Base values
input_df["DISTANCE"] = distance
input_df["HOUR"] = hour
input_df["DAY_OF_WEEK"] = day
input_df["IS_WEEKEND"] = is_weekend

# One-hot encoding
def set_column(name):
    if name in input_df.columns:
        input_df[name] = 1

set_column(f"AIRLINE_{airline}")
set_column(f"ORIGIN_AIRPORT_{origin}")
set_column(f"DESTINATION_AIRPORT_{destination}")
set_column(f"TIME_SLOT_{time_slot}")

# ==============================
# PREDICTION
# ==============================
if st.button("Predict"):

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("❌ Flight is likely to be DELAYED")
    else:
        st.success("✅ Flight is likely to be ON TIME")

    # Probability
    st.write(f"🧠 Delay Probability: {prob:.2f}")

    # Risk Level
    if prob < 0.3:
        risk = "Low Risk 🟢"
    elif prob < 0.6:
        risk = "Medium Risk 🟡"
    else:
        risk = "High Risk 🔴"

    st.write(f"⚠ Risk Level: {risk}")

    st.markdown("---")

    # ==============================
    # FLIGHT SUMMARY
    # ==============================
    st.subheader("✈ Flight Summary")
    st.write(f"**Airline:** {airline}")
    st.write(f"**Route:** {origin} → {destination}")
    st.write(f"**Distance:** {distance} miles")
    st.write(f"**Departure Hour:** {hour}")
    st.write(f"**Day:** {day_name}")

    st.markdown("---")

    # ==============================
    # EXPLANATION
    # ==============================
    st.info("Prediction is influenced by factors like departure time, airport traffic, airline history, and distance.")

    # ==============================
    # WARNING
    # ==============================
    if prob > 0.6:
        st.warning("⚠ High chance of delay. Consider alternative flights.")

    # ==============================
    # PIE CHART
    # ==============================
    st.subheader("📈 Delay Probability Visualization")

    labels = ["On Time", "Delayed"]
    values = [1 - prob, prob]

    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%')
    ax.set_title("Prediction Breakdown")

    st.pyplot(fig)