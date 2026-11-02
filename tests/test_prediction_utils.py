"""Run: python -m unittest discover -s tests -v
Tests preprocessing without needing Streamlit or a trained EBM pickle.
"""
import unittest
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tempfile
from xgboost import XGBRegressor
# Utilities live inside app/; tests are launched from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from prediction_utils import prepare_input, predict_all, predict_one, ebm_explanation, restore_portable_xgboost


class ToyRegressor:
    """Pickle-free substitute that checks correct columns and categorical features."""
    term_names_ = ["Year", "District"]
    intercept_ = np.array([1500.0])
    def predict(self, X):
        if "District" in X.columns:
            return np.asarray([1500.0 + float(X.iloc[0]["Year"]) + 10.0])
        return np.asarray([float(X.iloc[0]["Year"]) + float(X.iloc[0]["SownArea_ha"])])
    def eval_terms(self, X):
        return np.asarray([[float(X.iloc[0]["Year"]),10.0]])


class ModelInferenceTests(unittest.TestCase):
    def setUp(self):
        model = ToyRegressor()
        self.bundle = {
            "models": {key: model for key in ["Random Forest","XGBoost","CatBoost","LightGBM","EBM"]},
            "features": ["District", "Year", "Season", "SownArea_ha", "PreviousYearProduction_t"],
            "categorical_columns": ["District", "Season"],
            "numeric_columns": ["Year", "SownArea_ha", "PreviousYearProduction_t"],
            "encoded_columns": ["Year", "SownArea_ha", "PreviousYearProduction_t", "District_Ampara", "District_Galle", "Season_Maha", "Season_Yala"],
            "train_medians": {"Year": 2020, "SownArea_ha": 5000, "PreviousYearProduction_t": 9000},
            "ebm_numeric_medians": {"Year": 2020, "SownArea_ha": 5000, "PreviousYearProduction_t": 9000},
            "districts": ["Ampara", "Galle"], "seasons": ["Maha", "Yala"],
            "numeric_defaults": {"Year": 2020, "SownArea_ha": 5000, "PreviousYearProduction_t": 9000},
            "trained_through_year": 2021,
            "selected_by_validation": "XGBoost", "validation_results": [], "test_results": [], "ebm_global_importance": [],
        }
        self.row = {"District":"Ampara","Year":2022,"Season":"Maha","SownArea_ha":10000,"PreviousYearProduction_t":np.nan}

    def test_ensemble_preprocessing_and_median(self):
        x = prepare_input(self.row, self.bundle, "Random Forest")
        self.assertEqual(list(x.columns), self.bundle["encoded_columns"])
        self.assertEqual(float(x.iloc[0]["PreviousYearProduction_t"]),9000.0)
        self.assertEqual(int(x.iloc[0]["District_Ampara"]),1)
        self.assertEqual(int(x.iloc[0]["Season_Maha"]),1)

    def test_ebm_preserves_native_categories(self):
        x = prepare_input(self.row, self.bundle, "EBM")
        self.assertEqual(list(x.columns),self.bundle["features"])
        self.assertEqual(x.iloc[0]["District"],"Ampara")
        self.assertEqual(x.iloc[0]["Season"],"Maha")
        self.assertEqual(float(x.iloc[0]["PreviousYearProduction_t"]),9000.0)

    def test_all_five_predict_and_contributions_sum(self):
        output = predict_all(self.row, self.bundle)
        self.assertEqual(len(output),5)
        base, value, effects = ebm_explanation(self.row, self.bundle)
        self.assertAlmostEqual(base+effects["Contribution (tonnes)"].sum(),value)

    def test_what_if_for_all_five_models(self):
        from prediction_utils import scenario_sensitivity
        for model_name in self.bundle['models']:
            output = scenario_sensitivity(self.row, self.bundle, model_name)
            self.assertIn('Prediction change (tonnes)', output.columns)
            self.assertNotIn('Year', output['Feature'].tolist())
            self.assertEqual(len(output), 1)  # sown area; previous-year production missing
            self.assertTrue(np.isfinite(output['Prediction change (tonnes)']).all())

    def test_invalid_sensitivity_percentage(self):
        from prediction_utils import scenario_sensitivity
        with self.assertRaises(ValueError):
            scenario_sensitivity(self.row, self.bundle, 'EBM', -0.2)

    def test_portable_xgboost_json_restore(self):
        # Exercise native JSON reconstruction without touching the EBM library.
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.uniform(size=(20, 2)), columns=["a", "b"])
        y = 10 * X["a"] + X["b"]
        trained = XGBRegressor(n_estimators=4, n_jobs=1, random_state=42)
        trained.fit(X, y)
        with tempfile.TemporaryDirectory() as folder:
            native_file = Path(folder) / "xgboost.json"
            trained.save_model(str(native_file))
            bundle = {"models": {"XGBoost": None},
                      "xgboost_native_json": native_file.read_bytes()}
            restored = restore_portable_xgboost(bundle)["models"]["XGBoost"]
        self.assertTrue(np.allclose(trained.predict(X), restored.predict(X)))

    def test_missing_feature_rejected(self):
        incomplete={"Year":2023}
        with self.assertRaisesRegex(ValueError,"Missing input field"):
            predict_one(incomplete,self.bundle,"EBM")

if __name__ == "__main__":
    unittest.main()

