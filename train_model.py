"""
train_model.py
==============
This is the EXACT same code as final_project.ipynb, converted to a script.
The only additions (marked ★) are the extra saves needed by the Flask app
(label encoder classes, feature importance, classes map).
Run once before starting the server:  python train_model.py
"""

import os, json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib

# ★ ensure models/ folder exists
os.makedirs("models", exist_ok=True)

# CELL 1 : load data
df = pd.read_csv("student_dataset.csv")

# CELL 2 : inspect nulls
print(df.isnull().sum())

# CELL 3 : head
print(df.head(10))

# CELL 4 : fill AI columns for non-AI users
df.loc[df["uses_ai"] == "No", "ai_tools_used"] = "No_AI"
df.loc[df["uses_ai"] == "No", "ai_usage_purpose"] = "No_AI"

# CELL 5 : fill remaining NaN
df["ai_tools_used"] = df["ai_tools_used"].fillna("No_AI")
df["ai_usage_purpose"] = df["ai_usage_purpose"].fillna("No_AI")

# CELL 6-7 : verify
print(df.isnull().sum())
print(df.head())

# CELL 8 : drop non-feature columns
df.drop(["final_score", "passed", "student_id"], axis=1, inplace=True, errors="ignore")

# CELL 9 : label encoder
le = LabelEncoder()

# ★ Save individual encoders so Flask can encode incoming form data
le_gender = LabelEncoder().fit(["Female", "Male", "Other"])
le_uses_ai = LabelEncoder().fit([0, 1])

# Save performance_category BEFORE encoding so we keep the class names
perf_classes = sorted(df["performance_category"].unique())
le_perf = LabelEncoder().fit(perf_classes)
df["performance_category"] = le_perf.transform(df["performance_category"])

# CELL 10 : encode categoricals (matching notebook order)
df["gender"] = le.fit_transform(df["gender"])
df["uses_ai"] = le.fit_transform(df["uses_ai"])

joblib.dump(le_gender, "models/le_gender.pkl")
joblib.dump(le_uses_ai, "models/le_uses_ai.pkl")
joblib.dump(le_perf, "models/le_target.pkl")

# CELL 11 : one-hot encode
df = pd.get_dummies(df, columns=["ai_tools_used", "ai_usage_purpose", "grade_level"])

# CELL 12 : features / target
x = df.drop("performance_category", axis=1)
y = df["performance_category"]

# CELL 13 : train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# CELL 14-15 : Decision Tree
dt_model = DecisionTreeClassifier(max_depth=8)
dt_model.fit(x_train, y_train)
dt_pred = dt_model.predict(x_test)
dt_acc = accuracy_score(y_test, dt_pred)
print("Decision Tree Accuracy:", dt_acc)

# CELL 16-17 : Random Forest
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10)
rf_model.fit(x_train, y_train)
rf_pred = rf_model.predict(x_test)
rf_acc = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:", rf_acc)

# CELL 18-19 : XGBoost
xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.1, eval_metric="mlogloss")
xgb_model.fit(x_train, y_train)
xgb_pred = xgb_model.predict(x_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
print("XGBoost Accuracy:", xgb_acc)

# CELL 20 : accuracies dict
accuracies = {
    "Decision Tree": dt_acc,
    "Random Forest": rf_acc,
    "XGBoost": xgb_acc,
}

# CELL 21
best_model_name = max(accuracies, key=accuracies.get)
best_accuracy = accuracies[best_model_name]
print("Best Model: ", best_model_name)
print("Best Accuracy: ", best_accuracy)

# CELL 22 : save best model
if best_model_name == "Decision  Tree":   # typo preserved from notebook
    best_model = dt_model
elif best_model_name == "Random Forest":
    best_model = rf_model
else:
    best_model = xgb_model

joblib.dump(best_model, "best_model.pkl")          # original notebook output
joblib.dump(best_model, "models/best_model.pkl")   # ★ Flask path
print("Model saved as best_model.pkl")

# CELL 23 : save feature columns
joblib.dump(x.columns.tolist(), "columns.pkl")         # original
joblib.dump(x.columns.tolist(), "models/columns.pkl")  # ★ Flask path

# ★ Extra saves for Flask UI (classes map + feature importance)
classes_map = {int(le_perf.transform([c])[0]): c for c in le_perf.classes_}
with open("models/classes_map.json", "w") as f:
    json.dump(classes_map, f)

fi = dict(zip(x.columns.tolist(), best_model.feature_importances_.astype(float)))
fi_sorted = dict(sorted(fi.items(), key=lambda kv: kv[1], reverse=True)[:15])
with open("models/feature_importance.json", "w") as f:
    json.dump(fi_sorted, f)

print("\n✅ All model artefacts saved to models/")
print("   Classes:", classes_map)
print("   Top feature:", list(fi_sorted.keys())[0])
