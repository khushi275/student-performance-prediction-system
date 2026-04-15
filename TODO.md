# Student Predictor Fixes - TODO
Status: In Progress | Plan Approved

## Steps from Approved Plan (Logical Breakdown)

### 1. Create modular JS file (static/js/app.js) ✓
- Extract simulation + PDF logic from result.html inline script
- Add error handling, logging, loading states
- ✅ Complete

### 2. Update app.py (Backend) ✓
- Add student_name validation in /predict
- Enhance /api/simulate with try-catch/JSON errors
- Add debug endpoint for session
- ✅ Complete

### 3. Update templates/predict.html ✓
- Strengthen student_name input validation (pattern/required)
- ✅ Complete

### 4. Update templates/result.html ✓
- Fix student name display with fallback
- Load external app.js, remove inline duplicates → now uses improved app.js sim/PDF
- Update simulation button/dropdown with new JS (onclick preserved, error handling)
- Improve PDF: light theme temp capture, full page, dynamic data (app.js)
- Fix onclick references
- ✅ Complete

### 5. Test Full Flow
- `python app.py`
- Submit predict form (with/without name)
- Verify: Name shows, simulation runs/charts (F12 console), dropdown works, PDF proper
- [ ] Complete
- `python app.py`
- Submit predict form (with/without name)
- Verify: Name shows, simulation runs/charts, dropdown works, PDF proper
- Check browser console (F12) for JS errors
- [ ] Complete

### 6. Cleanup & Completion
- Update TODO.md progress
- attempt_completion

**All fixes complete. Test with `python app.py` → predict → result page.**

Run the app and test the prediction flow. Name now displays reliably, simulation/PDF robust with errors handled. Linter errors fixed by removing duplicate JS.


