# GitHub file update list

This ZIP provides the full **available project folder layout** for uploading to
`Paddy_Production`. It is an **overlay**: preserve unrelated files in your
existing repo. The live GitHub tree was not accessible for an exact remote diff.

| File / folder | Action | Notes |
|---|---|---|
| `app/app.py` | **REPLACE** | Larger fonts and inputs; Dashboard, Predict, Insights; EBM XAI + what-if |
| `app/prediction_utils.py` | **ADD / REPLACE** | Model-specific preprocessing, inference, local EBM explanation, sensitivity |
| `notebooks/Predicting_Seasonal_Paddy_Production.ipynb` | **REPLACE** | Current five-model EBM notebook; reads `data/raw/` and exports to `models/` |
| `data/raw/paddy_weather_final.xlsx` | **KEEP / RENAME** | Use this exact filename (no `(2)` suffix) |
| `models/paddy_production_all_five_models_ebm.pkl` | **GENERATE + ADD** | Not included: run the notebook; required for predictions |
| `models/README.md` | **ADD** | Model generation guidance |
| `.streamlit/config.toml` | **ADD / REPLACE** | Green theme matching larger-font UI |
| `tests/test_prediction_utils.py` | **ADD** | Inference/preprocessing tests |
| `requirements.txt` | **REPLACE** | App runtime dependencies |
| `requirements-notebook.txt` | **ADD** | Extra training/notebook dependencies |
| `README.md` | **REVIEW / MERGE** | Updated project setup; keep prior info you still need |
| `DEMO_GUIDE.md` | **ADD** | 3–5-minute presentation/video checklist |
| `.gitignore` | **REVIEW / MERGE** | Ignore caches and local secrets |

## Uploading to GitHub

Place each file under its shown path, not everything in the repository root.
If uploading through GitHub's web UI, open each folder before uploading its
contents. For a local clone, copy the folder tree over your checkout, inspect
`git status`, and commit only the intended files.

Run from repo root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m streamlit run app/app.py
```

**The ZIP cannot run real predictions until you add the generated EBM pickle.**
Do not upload the older single-model XGBoost/GPR pickle under the new filename.


## Pickle compatibility patch
- Replace `app/app.py` and `app/prediction_utils.py`.
- Replace `notebooks/Predicting_Seasonal_Paddy_Production.ipynb` to use native JSON for XGBoost in one pickle.
- Add `scripts/convert_existing_pickle.py` to convert the current generated pickle without retraining.
- See `PICKLE_ERROR_FIX.md` for steps.
