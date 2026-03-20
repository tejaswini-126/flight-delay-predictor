# ==========================================
# SMART FLIGHT DELAY PREDICTION SYSTEM
# ==========================================

import pandas as pd
import numpy as np

# ==============================
# STEP 1: LOAD DATA
# ==============================

df = pd.read_csv("data/flights.csv", nrows=100000)

print("Initial Shape:", df.shape)

# ==============================
# STEP 2: SELECT IMPORTANT COLUMNS
# ==============================

df = df[[
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "DEPARTURE_TIME",
    "ARRIVAL_DELAY",
    "DISTANCE",
    "DAY_OF_WEEK"
]]

# ==============================
# STEP 3: HANDLE MISSING VALUES
# ==============================

df = df.dropna()

print("After dropping NA:", df.shape)

# ==============================
# STEP 4: TARGET VARIABLE
# ==============================

df["DELAYED"] = df["ARRIVAL_DELAY"].apply(lambda x: 1 if x > 10 else 0)

# ==============================
# STEP 5: FEATURE ENGINEERING
# ==============================

# Convert time to hour
df["HOUR"] = df["DEPARTURE_TIME"] // 100

# Time slot
def get_time_slot(hour):
    if hour < 12:
        return "Morning"
    elif hour < 18:
        return "Afternoon"
    else:
        return "Night"

df["TIME_SLOT"] = df["HOUR"].apply(get_time_slot)

# Weekend
df["IS_WEEKEND"] = df["DAY_OF_WEEK"].apply(lambda x: 1 if x >= 6 else 0)

# Distance category
def distance_category(d):
    if d < 500:
        return "Short"
    elif d < 1500:
        return "Medium"
    else:
        return "Long"

df["DIST_CAT"] = df["DISTANCE"].apply(distance_category)

# ==============================
# STEP 6: REDUCE FEATURE SIZE (IMPORTANT)
# ==============================

top_airports = df['ORIGIN_AIRPORT'].value_counts().nlargest(20).index
df = df[df['ORIGIN_AIRPORT'].isin(top_airports)]

# ==============================
# STEP 7: DROP UNUSED COLUMNS
# ==============================

df = df.drop(["ARRIVAL_DELAY", "DEPARTURE_TIME"], axis=1)

# ==============================
# STEP 8: ENCODING
# ==============================

df = pd.get_dummies(df, drop_first=True)

print("Final Shape after encoding:", df.shape)

# ==============================
# STEP 9: SPLIT FEATURES & TARGET
# ==============================

X = df.drop("DELAYED", axis=1)
y = df["DELAYED"]

# ==============================
# STEP 10: HANDLE IMBALANCE (SMOTE)
# ==============================

from imblearn.over_sampling import SMOTE

smote = SMOTE()

X_res, y_res = smote.fit_resample(X, y)

print("After SMOTE:", X_res.shape)

# ==============================
# STEP 11: TRAIN TEST SPLIT
# ==============================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# ==============================
# STEP 12: TRAIN MODELS
# ==============================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = {}

print("\nTraining Models...\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    results[name] = score
    print(f"{name}: {score:.4f}")

# ==============================
# STEP 13: SELECT BEST MODEL
# ==============================

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\nBest Model:", best_model_name)

# ==============================
# STEP 14: EVALUATION
# ==============================

from sklearn.metrics import classification_report, confusion_matrix

y_pred = best_model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ==============================
# STEP 15: FEATURE IMPORTANCE
# ==============================

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    feature_names = X.columns

    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print("\nTop 10 Important Features:\n")
    print(feat_imp.head(10))

# ==============================
# STEP 16: SAVE MODEL
# ==============================
import joblib

# Save model
joblib.dump(best_model, "models/best_model.pkl")

print("\nModel saved successfully in models/best_model.pkl ✅")

# Save feature columns
joblib.dump(X.columns, "models/columns.pkl")

print("Feature columns saved in models/columns.pkl ✅")