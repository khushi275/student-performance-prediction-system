<<<<<<< HEAD
# student-performance-prediction-system
=======
# 🎓 AI-Based Student Performance Predictor

A full-stack machine learning web application built with **Flask**, **SQLite**, and deployed on **Render**.
Predicts student academic performance using an ensemble of ML models.

---

## 🚀 Features

### Core Predictions
- Final Grade (A / B / C/D)
- Pass / Fail Probability (%)
- Estimated Score Range (e.g. 65–75)

### Advanced Predictions
- Risk Level (Low 🟢 / Medium 🟡 / High 🔴)
- AI-Generated Personalised Insights
- Feature Importance Analysis

### UI Highlights
- ✨ Animated gradient landing page with floating particles
- 📊 Interactive charts (Bar, Doughnut, Radar, Feature Importance)
- 🔬 "What If?" Performance Simulation
- 🆚 Side-by-side Student Comparison with Radar overlay
- 📄 PDF Report Download
- 🗄️ Prediction History (SQLite)
- 🌙 Dark Mode (always-on, professional)
- ⚡ AJAX-powered simulation (no page reload)

---

## 📁 Project Structure

```
student_predictor/
├── app.py                  # Flask application (all routes, APIs)
├── train_model.py          # Model training script (run once)
├── student_dataset.csv     # Training dataset (8000 records)
├── requirements.txt        # Python dependencies
├── render.yaml             # Render deployment config
├── models/                 # Auto-created by train_model.py
│   ├── best_model.pkl
│   ├── columns.pkl
│   ├── le_gender.pkl
│   ├── le_uses_ai.pkl
│   ├── le_target.pkl
│   ├── classes_map.json
│   └── feature_importance.json
├── templates/
│   ├── index.html          # Landing page
│   ├── predict.html        # Input form
│   ├── result.html         # Output dashboard
│   ├── history.html        # Prediction history
│   ├── insights.html       # Analytics dashboard
│   └── compare.html        # Student comparison
└── predictions.db          # SQLite database (auto-created)
```

---

## ⚙️ Local Setup

```bash
# 1. Clone or unzip the project
cd student_predictor

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (creates models/ folder)
python train_model.py

# 5. Run the app
python app.py
# → http://localhost:5000
```

---

## 🌐 Deploying to Render

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/student-predictor.git
git push -u origin main
```

### Step 2: Deploy on Render
1. Go to [render.com](https://render.com) → **New Web Service**
2. Connect your GitHub repo
3. Render auto-detects `render.yaml`
4. It will:
   - Install requirements
   - Run `python train_model.py` (train + save model)
   - Start with `gunicorn app:app`
5. Your app is live at `https://your-app.onrender.com`

> ⚠️ **Note:** SQLite `predictions.db` resets on each Render deploy (ephemeral disk).
> For production, upgrade to PostgreSQL via Render's managed database.

---

## 🔗 API Endpoints

| Route | Method | Description |
|---|---|---|
| `/` | GET | Landing page |
| `/predict` | GET/POST | Form + prediction |
| `/result` | GET | Last result (session) |
| `/history` | GET | Prediction history UI |
| `/insights` | GET | Analytics dashboard |
| `/compare` | GET | Compare two students |
| `/api/predict` | POST | JSON prediction API |
| `/api/compare` | POST | JSON compare API |
| `/api/simulate` | POST | What-if simulation |
| `/api/history` | GET | JSON history |

### Example API call

```bash
curl -X POST https://your-app.onrender.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 18, "gender": "Male", "grade_level": "12th",
    "study_hours_per_day": 5, "attendance_percentage": 85,
    "last_exam_score": 72, "assignment_scores_avg": 75,
    "concept_understanding_score": 68, "class_participation_score": 60,
    "sleep_hours": 7, "social_media_hours": 2,
    "tutoring_hours": 1, "study_consistency_index": 0.8,
    "improvement_rate": 0.1, "ai_dependency_score": 30,
    "uses_ai": "0"
  }'
```

---

## 🧠 ML Models Used

| Model | Algorithm |
|---|---|
| Decision Tree | `max_depth=8` |
| Random Forest | `200 estimators, max_depth=10` |
| Gradient Boosting | `200 estimators, lr=0.1` |

The best-performing model is automatically saved and used for inference.

---

## 📊 Dataset Features (26 columns)

`student_id`, `age`, `gender`, `grade_level`, `study_hours_per_day`, `uses_ai`,
`ai_usage_time_minutes`, `ai_tools_used`, `ai_usage_purpose`, `ai_dependency_score`,
`ai_generated_content_percentage`, `ai_prompts_per_week`, `ai_ethics_score`,
`last_exam_score`, `assignment_scores_avg`, `attendance_percentage`,
`concept_understanding_score`, `study_consistency_index`, `improvement_rate`,
`sleep_hours`, `social_media_hours`, `tutoring_hours`, `class_participation_score`,
`final_score`, `passed`, `performance_category`

**Target:** `performance_category` → `Low | Medium | High`

---

## 👨‍💻 Tech Stack

- **Backend:** Python 3.11, Flask 3.0, SQLite
- **ML:** Scikit-learn, XGBoost, Pandas, NumPy, Joblib
- **Frontend:** Bootstrap 5.3, Chart.js 4.4, FontAwesome 6.5
- **PDF:** jsPDF + html2canvas
- **Deployment:** Render (gunicorn)
>>>>>>> 32488be (Initial Commit)
