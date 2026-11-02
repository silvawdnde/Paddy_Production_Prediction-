# Fix: `input stream corrupted` when loading the five-model pickle

Your 16.9 MB uploaded pickle has a complete pickle structure. The error may
come from the XGBoost model's internal binary memory snapshot or a dependency
version difference; a full traceback is needed to confirm the failing library.

## Easiest fix without retraining

1. In **Google Colab**, use the same working environment where you created
   `paddy_production_all_five_models_ebm.pkl`. Upload the pickle file.
2. Run the `convert_existing_pickle.py` script (provided under `scripts/`)
   with the path to the uploaded pickle. Example from the repository root:
   `python scripts/convert_existing_pickle.py models/paddy_production_all_five_models_ebm.pkl`
   In Colab, use `/content/paddy_production_all_five_models_ebm.pkl` if that's
   where the uploaded file is. The script replaces it with a compatible
   one-pickle bundle after checking the conversion.
3. Download the converted pickle and replace the file at
   `Paddy_Production/models/paddy_production_all_five_models_ebm.pkl`.
4. Update **both** `app/app.py` and `app/prediction_utils.py` from this package.
5. Restart Streamlit / reboot your deployment to clear the cached old pickle.

## Alternative if you have the fitted models still in the notebook

Run the **updated final two export/verification cells** in the new notebook.
They automatically save the portable one-pickle bundle and verify predictions.
You do **not** need to retrain if your notebook still has `trained_models`.

## If it still fails

Look at the Streamlit console/server logs; the updated app prints the FULL
traceback showing which package fails. Compare the versions recorded in
`bundle['library_versions']` with the deployed package versions.

- Match the producing notebook's Python, scikit-learn, CatBoost, LightGBM,
  InterpretML and XGBoost versions as closely as possible.
- Never `pickle.load` models from an untrusted source.

`models/paddy_production_all_five_models_ebm.pkl` is **not provided** in this
ZIP. The old model has not been silently relabeled as fixed, and you must
convert or re-export the real trained models.
