"""Convert an old five-model pickle to portable XGBoost inside the SAME pickle.

Run in the original environment where the old pickle loads (e.g. Google Colab).
No model retraining is needed. Only unpickle your own trusted model file.
Usage:
  python scripts/convert_existing_pickle.py models/paddy_production_all_five_models_ebm.pkl
"""
from pathlib import Path
import argparse
import pickle
import tempfile
import os
import importlib.metadata


def convert(path: Path):
    with path.open("rb") as f:
        bundle = pickle.load(f)
    models = bundle.get("models", {})
    if "XGBoost" not in models:
        raise ValueError("This pickle does not contain XGBoost.")
    if models["XGBoost"] is None and bundle.get("xgboost_native_json"):
        print("Already converted; no changes made.")
        return
    with tempfile.TemporaryDirectory() as folder:
        xgb_file = Path(folder) / "xgboost.json"
        models["XGBoost"].save_model(str(xgb_file))
        native_json = xgb_file.read_bytes()
    new_bundle = dict(bundle)
    new_bundle["models"] = dict(models)
    new_bundle["models"]["XGBoost"] = None
    new_bundle["xgboost_native_json"] = native_json
    versions = {}
    for name in ("scikit-learn", "xgboost", "catboost", "lightgbm", "interpret-core"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    new_bundle["library_versions"] = versions
    # Create and validate a new pickle before replacing the original.
    temporary_output = path.with_name(path.name + ".new")
    try:
        with temporary_output.open("wb") as f:
            pickle.dump(new_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
        with temporary_output.open("rb") as f:
            check = pickle.load(f)
        if check.get("xgboost_native_json") != native_json:
            raise ValueError("Verification failed; original file was NOT replaced.")
        os.replace(temporary_output, path)
        print(f"Converted: {path} ({path.stat().st_size:,} bytes).")
        print("XGBoost is now stored as portable native JSON inside one pickle file.")
    finally:
        if temporary_output.exists(): temporary_output.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pickle_path", type=Path, help="Path to the existing five-model pickle")
    args = parser.parse_args()
    convert(args.pickle_path)
