# Sri Lankan Seasonal Paddy Production Prediction

Interactive Streamlit dashboard for the MSc Applied Machine Learning assignment.
This version follows `Predicting_Seasonal_Paddy_Production.ipynb` and compares:

- Random Forest
- XGBoost
- CatBoost
- LightGBM
- Explainable Boosting Machine (EBM, the algorithm investigated for explainability)

## Directory structure

```text
Paddy_Production/
├── app/
│   ├── app.py
│   └── prediction_utils.py
├── notebooks/
│   └── Predicting_Seasonal_Paddy_Production.ipynb
├── models/
│   ├── README.md
│   └── paddy_production_all_five_models_ebm.pkl  <-- create from notebook
├── data/
│   └── raw/
│       └── paddy_weather_final.xlsx
├── tests/
│   └── test_prediction_utils.py
├── .streamlit/
│   └── config.toml
├── requirements.txt
├── requirements-notebook.txt
├── DEMO_GUIDE.md
└── .gitignore
```

## First-time setup

1. Open a terminal in the **repository root** (`Paddy_Production/`).
2. Optionally create and activate a virtual environment:

   Windows PowerShell:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the app dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. **Generate your trained model file.** Run the notebook from its first cell to
   the final pickle export. For notebook dependencies, install with
   `python -m pip install -r requirements-notebook.txt`.

   It should produce:
   `models/paddy_production_all_five_models_ebm.pkl`.

   If using Google Colab, download the generated .pkl file and upload it to
   `models/` yourself (Colab does not save directly to your local GitHub repo).

5. Launch Streamlit:

   ```powershell
   python -m streamlit run app/app.py
   ```

6. Open the URL displayed in the terminal, usually http://localhost:8501.

## What the front-end does

**Dashboard:** charts for production by year, district and season, filters,
plus a downloadable filtered dataset.

**Predict Production:** enter district, season, year, agricultural and weather
measurements; select any of five models; view the estimated harvest in tonnes,
compare all model estimates, and view explanations. EBM's actual additive
feature contributions are shown. For any chosen model, an additional one-at-a-
time +10% input sensitivity comparison is available; this is a *what-if* tool,
not SHAP, LIME, or proof of causality.

**Model Insights:** review the original notebook's unseen-test MAE/RMSE/R²,
validation-selected model, EBM global importance and an individual EBM
explanation. SHAP and LIME are already demonstrated in the notebook.

**Important scope:** these are estimates *after seasonal measurements are
available*. They are not weather forecasts or advance planting predictions.

## Dataset and model file locations

- Dataset: `data/raw/paddy_weather_final.xlsx`
- Trained bundle: `models/paddy_production_all_five_models_ebm.pkl`

The app discovers both paths relative to its own script, so the paths do not
change if the terminal is opened elsewhere.

## Check preprocessing and predictions

Run:

```powershell
python -m unittest discover -s tests -v
```

The tests use a small dummy model to check preprocessing and app utilities.
They do not prove the actual trained EBM model has loaded correctly. Run the
live application after creating the pickle.

## Differences from previous application

See `GITHUB_DELTA.md` for the files to add, replace, or keep in the repository.
This package includes only the project files available in the uploaded work;
it may not contain unrelated files from the existing remote repository.

**Data sources identified in the notebook:** Department of Census and Statistics
(Sri Lanka) for paddy statistics and Open-Meteo for weather. Retain the source
and data compilation description in the written assignment report.
