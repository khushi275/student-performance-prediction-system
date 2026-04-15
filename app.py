"""
AI-Based Student Performance Predictor — Flask Application
Render-ready | SQLite | AJAX | PDF | Comparison | Simulation
"""
import os, json, sqlite3, datetime
from flask import (Flask, render_template, request, jsonify,
                   g, redirect, url_for, session)
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "studentai-secret-2024")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DB_PATH   = os.path.join(BASE_DIR, "predictions.db")

# ── Load model artefacts ──────────────────────────────────────────────────
model        = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "columns.pkl"))
le_gender    = joblib.load(os.path.join(MODEL_DIR, "le_gender.pkl"))
le_target    = joblib.load(os.path.join(MODEL_DIR, "le_target.pkl"))

with open(os.path.join(MODEL_DIR, "classes_map.json")) as f:
    classes_map = {int(k): v for k, v in json.load(f).items()}

with open(os.path.join(MODEL_DIR, "feature_importance.json")) as f:
    feature_importance = json.load(f)

# ── Notebook: uses_ai is already int (0/1) in the dataset ─────────────────
# LabelEncoder on [0,1] → 0→0, 1→1 (identity), so no encoder needed.
# gender: Female→0, Male→1, Other→2 (sorted alphabetically by LabelEncoder)
GENDER_MAP = {"Female": 0, "Male": 1, "Other": 2}

# ── DB helpers ────────────────────────────────────────────────────────────
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db:
        db.close()

def init_db():
    db = sqlite3.connect(DB_PATH)
    db.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name        TEXT,
            age                 INTEGER,
            gender              TEXT,
            grade_level         TEXT,
            study_hours         REAL,
            attendance          REAL,
            last_exam_score     INTEGER,
            sleep_hours         REAL,
            social_media_hours  REAL,
            predicted_grade     TEXT,
            performance         TEXT,
            prob_high           REAL,
            prob_medium         REAL,
            prob_low            REAL,
            pass_probability    REAL,
            risk_level          TEXT,
            score_estimate      REAL,
            created_at          TEXT
        )
    """)
    db.commit()
    db.close()

init_db()

# ── Mapping helpers ───────────────────────────────────────────────────────
GRADE_MAP     = {"High": "A",           "Medium": "B",       "Low": "C/D"}
RISK_LABEL    = {"High": "Low Risk",    "Medium": "Medium Risk", "Low": "High Risk"}
RISK_COLOR    = {"High": "success",     "Medium": "warning",    "Low": "danger"}

# ── Rule-based AI insights ────────────────────────────────────────────────
INSIGHT_RULES = [
    (lambda d: d["attendance_percentage"] < 60,
     "⚠️ Attendance below 60% — aim for 80%+ to significantly boost your grade."),
    (lambda d: d["study_hours_per_day"] < 2,
     "📚 Study time is very low. Aim for at least 3–4 focused hours daily."),
    (lambda d: d["social_media_hours"] > 4,
     "📱 High social media usage (>4 hrs/day) correlates with lower academic scores."),
    (lambda d: d["sleep_hours"] < 6,
     "😴 Sleep deprivation impairs memory. Target 7–8 hours per night."),
    (lambda d: d["sleep_hours"] > 9,
     "🛌 Excess sleep (>9 hrs) may be cutting into productive study time."),
    (lambda d: d["ai_dependency_score"] > 70,
     "🤖 Very high AI dependency — build independent problem-solving skills."),
    (lambda d: d["class_participation_score"] < 40,
     "🙋 Low class participation reduces concept retention. Engage more actively."),
    (lambda d: d["concept_understanding_score"] < 50,
     "🧠 Weak concept understanding — revisit fundamentals and seek tutoring support."),
    (lambda d: d["assignment_scores_avg"] < 50,
     "📝 Low assignment scores — consistent homework completion lifts final grades."),
    (lambda d: d["tutoring_hours"] == 0,
     "👨‍🏫 Consider enrolling in tutoring sessions to strengthen weak areas."),
]

def get_insights(data: dict) -> list:
    msgs = [msg for check, msg in INSIGHT_RULES if check(data)]
    return msgs if msgs else ["✅ Student profile looks well-balanced. Keep it up!"]

def estimate_score(ph, pm, pl):
    s = ph * 85 + pm * 65 + pl * 42
    return round(s, 2), round(max(0, s - 8), 2), round(min(100, s + 8), 2)

# ── Preprocess form → DataFrame ───────────────────────────────────────────
def preprocess(form: dict) -> pd.DataFrame:
    uses_ai_int = int(form.get("uses_ai", 0))
    gender_enc  = GENDER_MAP.get(form.get("gender", "Male"), 1)
    grade_level = form.get("grade_level", "12th")
    ai_tools    = form.get("ai_tools_used", "No_AI") if uses_ai_int else "No_AI"
    ai_purpose  = form.get("ai_usage_purpose", "No_AI") if uses_ai_int else "No_AI"

    row = {
        "age":                              int(form.get("age", 18)),
        "gender":                           gender_enc,
        "study_hours_per_day":              float(form.get("study_hours_per_day", 3)),
        "uses_ai":                          uses_ai_int,
        "ai_usage_time_minutes":            int(form.get("ai_usage_time_minutes") or 0),
        "ai_dependency_score":              int(form.get("ai_dependency_score") or 0),
        "ai_generated_content_percentage":  int(form.get("ai_generated_content_percentage") or 0),
        "ai_prompts_per_week":              int(form.get("ai_prompts_per_week") or 0),
        "ai_ethics_score":                  int(form.get("ai_ethics_score") or 50),
        "last_exam_score":                  int(form.get("last_exam_score") or 60),
        "assignment_scores_avg":            float(form.get("assignment_scores_avg", 65)),
        "attendance_percentage":            float(form.get("attendance_percentage", 75)),
        "concept_understanding_score":      int(form.get("concept_understanding_score") or 60),
        "study_consistency_index":          float(form.get("study_consistency_index", 0.5)),
        "improvement_rate":                 float(form.get("improvement_rate", 0.0)),
        "sleep_hours":                      float(form.get("sleep_hours", 7)),
        "social_media_hours":               float(form.get("social_media_hours", 2)),
        "tutoring_hours":                   float(form.get("tutoring_hours", 0)),
        "class_participation_score":        int(form.get("class_participation_score") or 50),
        # ── OHE : ai_tools_used ──────────────────────────────────────────
        "ai_tools_used_ChatGPT":            1 if ai_tools == "ChatGPT" else 0,
        "ai_tools_used_ChatGPT+Gemini":     1 if ai_tools == "ChatGPT+Gemini" else 0,
        "ai_tools_used_Claude":             1 if ai_tools == "Claude" else 0,
        "ai_tools_used_Copilot":            1 if ai_tools == "Copilot" else 0,
        "ai_tools_used_Gemini":             1 if ai_tools == "Gemini" else 0,
        "ai_tools_used_No_AI":              1 if ai_tools == "No_AI" else 0,
        # ── OHE : ai_usage_purpose ───────────────────────────────────────
        "ai_usage_purpose_Coding":          1 if ai_purpose == "Coding" else 0,
        "ai_usage_purpose_Doubt Solving":   1 if ai_purpose == "Doubt Solving" else 0,
        "ai_usage_purpose_Exam Prep":       1 if ai_purpose == "Exam Prep" else 0,
        "ai_usage_purpose_Homework":        1 if ai_purpose == "Homework" else 0,
        "ai_usage_purpose_No_AI":           1 if ai_purpose == "No_AI" else 0,
        "ai_usage_purpose_Notes":           1 if ai_purpose == "Notes" else 0,
        # ── OHE : grade_level ────────────────────────────────────────────
        "grade_level_10th":                 1 if grade_level == "10th" else 0,
        "grade_level_11th":                 1 if grade_level == "11th" else 0,
        "grade_level_12th":                 1 if grade_level == "12th" else 0,
        "grade_level_1st Year":             1 if grade_level == "1st Year" else 0,
        "grade_level_2nd Year":             1 if grade_level == "2nd Year" else 0,
        "grade_level_3rd Year":             1 if grade_level == "3rd Year" else 0,
    }

    df_row = pd.DataFrame([row])
    df_row = df_row.reindex(columns=feature_cols, fill_value=0)
    return df_row

# ── Core prediction logic ─────────────────────────────────────────────────
def run_prediction(form: dict) -> dict:
    df_row = preprocess(form)
    proba  = model.predict_proba(df_row)[0]
    proba_native = proba.astype(float).tolist()  # Fix float32
    labels = [classes_map[i] for i in range(len(proba))]

    prob_dict      = dict(zip(labels, proba_native))
    predicted_label = labels[np.argmax(proba)]

    ph = prob_dict.get("High",   0.0)
    pm = prob_dict.get("Medium", 0.0)
    pl = prob_dict.get("Low",    0.0)
    pass_prob = ph + pm

    score_est, score_lo, score_hi = estimate_score(ph, pm, pl)

    insight_data = {
        "attendance_percentage":       float(form.get("attendance_percentage", 75)),
        "study_hours_per_day":         float(form.get("study_hours_per_day", 3)),
        "social_media_hours":          float(form.get("social_media_hours", 2)),
        "sleep_hours":                 float(form.get("sleep_hours", 7)),
        "ai_dependency_score":         int(form.get("ai_dependency_score", 0)),
        "class_participation_score":   int(form.get("class_participation_score", 50)),
        "concept_understanding_score": int(form.get("concept_understanding_score", 60)),
        "assignment_scores_avg":       float(form.get("assignment_scores_avg", 65)),
        "tutoring_hours":              float(form.get("tutoring_hours", 0)),
    }

    radar_native = {
        "study_time":    float(min(100, float(form.get("study_hours_per_day", 3)) / 8 * 100)),
        "attendance":    float(form.get("attendance_percentage", 75)),
        "performance":   float(form.get("last_exam_score", 60)),
        "participation": float(form.get("class_participation_score", 50)),
        "consistency":   float(form.get("study_consistency_index", 0.5)) * 100,
    }

    return {
        "predicted_grade":   GRADE_MAP.get(predicted_label, predicted_label),
        "performance":       predicted_label,
        "prob_high":         float(round(ph * 100, 2)),
        "prob_medium":       float(round(pm * 100, 2)),
        "prob_low":          float(round(pl * 100, 2)),
        "pass_probability":  float(round((ph + pm) * 100, 2)),
        "risk_level":        RISK_LABEL.get(predicted_label, "Medium Risk"),
        "risk_color":        RISK_COLOR.get(predicted_label, "warning"),
        "score_estimate":    float(score_est),
        "score_range_low":   float(score_lo),
        "score_range_high":  float(score_hi),
        "insights":          get_insights(insight_data),
        "feature_importance": feature_importance,
        "radar":             radar_native,
    }

# ── Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    form = request.form.to_dict()
    
    # ✅ FIX 1: Student name validation & logging
    student_name = form.get("student_name", "").strip()
    if not student_name or len(student_name) < 2:
        student_name = "Anonymous Student"
        print("WARNING: Invalid/missing student_name, using default")
    
    form["student_name"] = student_name  # Ensure set
    print(f"Predict for student: {student_name}")  # Debug log
    
    result = run_prediction(form)

    # Recursive conversion for numpy → Python types (DB + session)
    def to_native(obj):
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_native(v) for v in obj]
        elif hasattr(obj, '__float__'):
            return float(obj)
        elif hasattr(obj, 'item'):
            return obj.item()
        elif np.issubdtype(type(obj), np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'numpy'):
            return obj.numpy()
        return obj
    
    result_native = to_native(result)
    result_native["student_name"] = student_name  # Ensure set

    # Persist to SQLite — store full form + result JSON so history detail shows exact original data
    db = get_db()
    db.execute("""
        INSERT INTO predictions
        (student_name, age, gender, grade_level, study_hours, attendance,
         last_exam_score, sleep_hours, social_media_hours,
         predicted_grade, performance,
         prob_high, prob_medium, prob_low,
         pass_probability, risk_level, score_estimate, created_at,
         full_form_json, full_result_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        student_name,
        int(form.get("age", 0)),
        form.get("gender", ""),
        form.get("grade_level", ""),
        float(form.get("study_hours_per_day", 0)),
        float(form.get("attendance_percentage", 0)),
        int(form.get("last_exam_score", 0)),
        float(form.get("sleep_hours", 0)),
        float(form.get("social_media_hours", 0)),
        result_native["predicted_grade"],
        result_native["performance"],
        result_native["prob_high"],
        result_native["prob_medium"],
        result_native["prob_low"],
        result_native["pass_probability"],
        result_native["risk_level"],
        result_native["score_estimate"],
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        json.dumps({k: str(v) for k, v in form.items()}),
        json.dumps(result_native),
    ))
    db.commit()

    session["last_result"] = result_native
    session["last_form"] = {k: str(v) for k, v in form.items()}
    return render_template("result.html", result=result_native, form=form)

@app.route("/result")
def result():
    result = session.get("last_result")
    form = session.get("last_form", {})
    if not result:
        return redirect(url_for("predict"))
    # Ensure student_name always present
    if not result.get("student_name"):
        result["student_name"] = "Anonymous Student"
    return render_template("result.html", result=result, form=form)

# ── Debug endpoint ─────────────────────────────────────────────────────────
@app.route("/debug")
def debug():
    return jsonify({
        "session_last_result": session.get("last_result", {}),
        "session_last_form": session.get("last_form", {}),
        "session_keys": list(session.keys())
    })


@app.route("/history")
def history():
    db = get_db()
    rows = db.execute("SELECT * FROM predictions ORDER BY id ASC LIMIT 50").fetchall()
    return render_template("history.html", rows=rows)


@app.route("/history/<int:prediction_id>")
def history_detail(prediction_id):
    """Show the original result for a stored prediction — uses saved JSON if available,
       otherwise falls back to re-running prediction from stored fields."""
    db = get_db()
    row = db.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
    if not row:
        return redirect(url_for("history"))

    # --- Prefer stored JSON (new predictions after schema update) ---
    if row["full_result_json"] and row["full_form_json"]:
        result_native = json.loads(row["full_result_json"])
        form = json.loads(row["full_form_json"])
        result_native.setdefault("student_name", row["student_name"] or "Anonymous Student")
        return render_template("result.html", result=result_native, form=form,
                               from_history=True, history_id=prediction_id)

    # --- Fallback: reconstruct from individual columns (old rows) ---
    form = {
        "student_name":              row["student_name"] or "Anonymous Student",
        "age":                       str(row["age"] or 18),
        "gender":                    row["gender"] or "Male",
        "grade_level":               row["grade_level"] or "12th",
        "study_hours_per_day":       str(row["study_hours"] or 3),
        "attendance_percentage":     str(row["attendance"] or 75),
        "last_exam_score":           str(row["last_exam_score"] or 60),
        "sleep_hours":               str(row["sleep_hours"] or 7),
        "social_media_hours":        str(row["social_media_hours"] or 2),
        "assignment_scores_avg":     "65",
        "concept_understanding_score": "60",
        "class_participation_score": "50",
        "tutoring_hours":            "0",
        "study_consistency_index":   "0.5",
        "improvement_rate":          "0.0",
        "ai_dependency_score":       "0",
        "ai_generated_content_percentage": "0",
        "ai_prompts_per_week":       "0",
        "ai_ethics_score":           "50",
        "ai_usage_time_minutes":     "0",
        "uses_ai":                   "0",
        "ai_tools_used":             "No_AI",
        "ai_usage_purpose":          "No_AI",
    }

    # If stored result values are non-zero use them directly, else re-run
    if row["pass_probability"] and row["pass_probability"] > 0:
        result_native = {
            "student_name":    row["student_name"] or "Anonymous Student",
            "predicted_grade": row["predicted_grade"],
            "performance":     row["performance"],
            "prob_high":       float(row["prob_high"] or 0),
            "prob_medium":     float(row["prob_medium"] or 0),
            "prob_low":        float(row["prob_low"] or 0),
            "pass_probability":float(row["pass_probability"] or 0),
            "risk_level":      row["risk_level"],
            "risk_color":      {"Low Risk": "success", "Medium Risk": "warning", "High Risk": "danger"}.get(row["risk_level"], "warning"),
            "score_estimate":  float(row["score_estimate"] or 0),
            "score_range_low": max(0, float(row["score_estimate"] or 0) - 8),
            "score_range_high":min(100, float(row["score_estimate"] or 0) + 8),
            "insights":        get_insights({k: float(form.get(k, 0)) for k in [
                "attendance_percentage", "study_hours_per_day", "social_media_hours",
                "sleep_hours", "ai_dependency_score", "class_participation_score",
                "concept_understanding_score", "assignment_scores_avg", "tutoring_hours"
            ]}),
            "feature_importance": feature_importance,
            "radar": {
                "study_time":    min(100, float(form["study_hours_per_day"]) / 8 * 100),
                "attendance":    float(form["attendance_percentage"]),
                "performance":   float(form["last_exam_score"]),
                "participation": 50.0,
                "consistency":   50.0,
            },
        }
    else:
        def to_native(obj):
            if isinstance(obj, dict): return {k: to_native(v) for k, v in obj.items()}
            if isinstance(obj, list): return [to_native(v) for v in obj]
            if hasattr(obj, 'item'): return obj.item()
            return obj
        result_native = to_native(run_prediction(form))
        result_native["student_name"] = form["student_name"]

    return render_template("result.html", result=result_native, form=form,
                           from_history=True, history_id=prediction_id)

@app.route("/insights")
def insights():
    db = get_db()
    stats = db.execute("""
        SELECT predicted_grade, COUNT(*) as cnt,
               AVG(pass_probability) as avg_pass,
               AVG(score_estimate) as avg_score
        FROM predictions GROUP BY predicted_grade
    """).fetchall()
    return render_template("insights.html", stats=stats, feature_importance=feature_importance)

@app.route("/compare")
def compare():
    return render_template("compare.html")

# ── JSON API ──────────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    try:
        return jsonify({"status": "success", "result": run_prediction(data)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/compare", methods=["POST"])
def api_compare():
    data = request.get_json(force=True)
    try:
        r1 = run_prediction(data["student1"])
        r2 = run_prediction(data["student2"])
        return jsonify({"status": "success", "student1": r1, "student2": r2})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    data = request.get_json(force=True)
    try:
        param = data.get("param")
        base = data.get("base_form", {})
        values = data.get("values", [])
        if not param or not values:
            return jsonify({"status": "error", "message": "Missing param or values"}), 400
        
        results = []
        for v in values:
            form_copy = {**base, param: v}
            r = run_prediction(form_copy)
            results.append({
                "value": v,
                "pass_probability": r["pass_probability"],
                "predicted_grade": r["predicted_grade"],
                "performance": r["performance"],
            })
        return jsonify({"status": "success", "simulations": results})
    except Exception as e:
        print(f"Simulation API error: {str(e)}")  # Server log
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/model-performance")
def model_performance():
    import os
    metrics_path = os.path.join(MODEL_DIR, "eval_metrics.json")
    with open(metrics_path) as f:
        metrics = json.load(f)
    return render_template("model_performance.html", metrics=metrics)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

