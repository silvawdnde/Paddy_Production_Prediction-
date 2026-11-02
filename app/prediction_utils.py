"""Inference-only helpers matching Predicting_Seasonal_Paddy_Production.ipynb.

The notebook trains one-hot encoded trees and an EBM using native categories.
No fitting, new feature engineering, or normalization is performed at inference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EXPECTED_MODELS = ("Random Forest", "XGBoost", "CatBoost", "LightGBM", "EBM")
REQUIRED_BUNDLE_KEYS = (
    "models", "features", "categorical_columns", "numeric_columns",
    "encoded_columns", "train_medians", "ebm_numeric_medians", "districts",
    "seasons", "numeric_defaults", "trained_through_year", "selected_by_validation",
    "validation_results", "test_results", "ebm_global_importance",
)


def validate_bundle(bundle: dict) -> None:
    """Fail with actionable information if an outdated pickle was supplied."""
    if not isinstance(bundle, dict):
        raise ValueError("The pickle must contain the notebook's model-bundle dictionary.")
    absent = [k for k in REQUIRED_BUNDLE_KEYS if k not in bundle]
    if absent:
        raise ValueError("Pickle does not match the updated EBM notebook. Missing: " + ", ".join(absent))
    models = bundle["models"]
    if not isinstance(models, dict) or any(n not in models for n in EXPECTED_MODELS):
        raise ValueError("Pickle must contain Random Forest, XGBoost, CatBoost, LightGBM and EBM.")
    if not bundle["encoded_columns"] or not bundle["features"]:
        raise ValueError("Missing trained feature names in the pickle.")


def prepare_input(values: dict, bundle: dict, model_name: str) -> pd.DataFrame:
    """Replicate the final notebook's prepare_ensemble/prepare_ebm at inference.

    Inputs must include all raw features. Values may be missing, but feature names
    must not be missing; the trained medians are used for missing numeric values.
    """
    validate_bundle(bundle)
    if model_name not in bundle["models"]:
        raise ValueError(f"Unknown model: {model_name}")
    missing = [col for col in bundle["features"] if col not in values]
    if missing:
        raise ValueError("Missing input field(s): " + ", ".join(missing))

    raw = pd.DataFrame([{col: values[col] for col in bundle["features"]}], columns=bundle["features"])
    if model_name == "EBM":
        for col in bundle["numeric_columns"]:
            raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(
                float(bundle["ebm_numeric_medians"][col])
            )
        for col in bundle["categorical_columns"]:
            raw[col] = raw[col].fillna("Unknown").astype(str)
        return raw

    one_hot = pd.get_dummies(raw, columns=bundle["categorical_columns"], dtype=int)
    one_hot = one_hot.reindex(columns=bundle["encoded_columns"], fill_value=0)
    for col, median in bundle["train_medians"].items():
        if col in one_hot.columns:
            one_hot[col] = pd.to_numeric(one_hot[col], errors="coerce").fillna(float(median))
    if one_hot.isnull().values.any():
        raise ValueError("One or more input values could not be processed.")
    return one_hot


def predict_one(values: dict, bundle: dict, model_name: str) -> tuple[float, pd.DataFrame]:
    X = prepare_input(values, bundle, model_name)
    prediction = float(bundle["models"][model_name].predict(X)[0])
    if not np.isfinite(prediction):
        raise ValueError(f"{model_name} returned a non-finite prediction.")
    return prediction, X


def predict_all(values: dict, bundle: dict) -> pd.DataFrame:
    rows = []
    for name in EXPECTED_MODELS:
        estimate, _ = predict_one(values, bundle, name)
        rows.append({"Model": name, "Predicted production (tonnes)": estimate})
    return pd.DataFrame(rows)


def ebm_explanation(values: dict, bundle: dict) -> tuple[float, float, pd.DataFrame]:
    """Return baseline, prediction and additive contributions for one record."""
    estimate, X = predict_one(values, bundle, "EBM")
    model = bundle["models"]["EBM"]
    baseline = float(np.asarray(model.intercept_).ravel()[0])
    effects = np.asarray(model.eval_terms(X), dtype=float)[0]
    explanation = pd.DataFrame({
        "Feature / interaction": model.term_names_,
        "Contribution (tonnes)": effects,
    }).sort_values("Contribution (tonnes)", key=abs, ascending=False)
    return baseline, estimate, explanation


def scenario_sensitivity(values: dict, bundle: dict, model_name: str,
                         change_fraction: float = 0.10) -> pd.DataFrame:
    """One-at-a-time numeric input changes and their effect on this model.

    This is a simple what-if analysis, NOT a feature attribution method or a
    causal result. Inputs are changed only in memory; the model is never refit.
    """
    if not (0 < change_fraction <= 1):
        raise ValueError("change_fraction must be between 0 and 1.")
    baseline, _ = predict_one(values, bundle, model_name)
    changes = []
    for feature in bundle["numeric_columns"]:
        if feature == "Year":
            continue
        old = values.get(feature)
        if old is None or pd.isna(old):
            continue
        try:
            old = float(old)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(old) or old <= 0:
            continue
        new_input = dict(values)
        new_input[feature] = old * (1 + change_fraction)
        new_prediction, _ = predict_one(new_input, bundle, model_name)
        changes.append({"Feature": feature, "Prediction change (tonnes)": new_prediction - baseline})
    return pd.DataFrame(changes, columns=["Feature", "Prediction change (tonnes)"])


def restore_portable_xgboost(bundle: dict) -> dict:
    """Restore XGBoost from its native JSON inside the single model pickle.

    This avoids unpickling XGBoost's native in-memory training snapshot, which
    can fail when exporting in Linux/Colab and importing in Windows or when
    XGBoost versions differ. The other four models remain regular objects.
    """
    if bundle.get("models", {}).get("XGBoost") is not None:
        return bundle  # Compatible with the old bundle if it already loaded.
    raw = bundle.get("xgboost_native_json")
    if not isinstance(raw, bytes) or not raw:
        raise ValueError("XGBoost missing: regenerate or convert the pickle with the updated notebook.")

    import tempfile
    from pathlib import Path
    from xgboost import XGBRegressor
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "xgboost.json"
        path.write_bytes(raw)
        model = XGBRegressor()
        model.load_model(str(path))
    bundle["models"]["XGBoost"] = model
    return bundle
