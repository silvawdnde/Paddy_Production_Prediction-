"""Sri Lankan paddy production — Streamlit dashboard for the 5-model EBM notebook.

Run from the repository root:
    python -m streamlit run app/app.py

Input data: data/raw/paddy_weather_final.xlsx
Trained bundle: models/paddy_production_all_five_models_ebm.pkl

The app performs inference only; all five models are trained and exported by
notebooks/Predicting_Seasonal_Paddy_Production.ipynb. Only unpickle trusted files.
"""
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from prediction_utils import (
    EXPECTED_MODELS,
    ebm_explanation,
    predict_all,
    predict_one,
    scenario_sensitivity,
    restore_portable_xgboost,
    validate_bundle,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "paddy_production_all_five_models_ebm.pkl"
DATA_PATH = ROOT / "data" / "raw" / "paddy_weather_final.xlsx"

st.set_page_config(
    page_title="Paddy Production | Sri Lanka",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Deliberately use larger typography for projector/demo and everyday readability.
st.markdown("""
<style>
:root { --green:#18553a; --forest:#11402c; --muted:#486a58; --line:#d6e5d9; }
html, body, [class*="css"], .stApp { font-size:18px; }
.stApp { color:#20392b; background:#f7faf7; }
[data-testid="stMainBlockContainer"] { max-width:1340px; padding-top:1.6rem; padding-bottom:3rem; }
[data-testid="stSidebar"] { background:#e8f2e9; border-right:1px solid #d5e5d8; }
[data-testid="stSidebar"] * { font-size:1.06rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { font-size:1.04rem; }
h1 { color:var(--forest) !important; font-size:2.45rem !important; line-height:1.14 !important; font-weight:760 !important; }
h2 { color:var(--green) !important; font-size:1.72rem !important; }
h3 { color:var(--green) !important; font-size:1.35rem !important; }
p, li, [data-testid="stMarkdownContainer"] p { font-size:1.08rem; line-height:1.63; }
.stCaption, [data-testid="stCaptionContainer"] { font-size:0.93rem !important; line-height:1.45 !important; }
[data-testid="stWidgetLabel"] label, label[data-testid="stWidgetLabel"] { font-size:1.05rem !important; font-weight:650; color:#244d35 !important; }
[data-baseweb="input"] input, [data-baseweb="select"] div, [data-baseweb="select"] input,
[data-testid="stNumberInput"] input { font-size:1.1rem !important; min-height:2.6rem; }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:15px; padding:1.05rem 1.1rem; box-shadow:0 2px 8px rgba(17,64,44,.04); }
[data-testid="stMetricLabel"] { font-size:1.02rem !important; font-weight:650; }
[data-testid="stMetricValue"] { font-size:2.0rem !important; font-weight:760; color:var(--forest); }
[data-testid="stMetricDelta"] { font-size:1rem; }
.stButton>button, [data-testid="stFormSubmitButton"] button,
[data-testid="stDownloadButton"] button { font-size:1.12rem !important; min-height:3.0rem; font-weight:650; border-radius:11px; }
button[data-baseweb="tab"] { font-size:1.13rem !important; font-weight:650; }
[data-testid="stDataFrame"] { font-size:1.05rem; }
[data-testid="stAlert"] { font-size:1.05rem; border-radius:12px; }
.hero { background:linear-gradient(118deg,#1b6344,#2e8061); padding:1.4rem 1.6rem; border-radius:18px; color:#fff; margin-bottom:1.3rem; }
.hero h2 { color:#fff !important; margin:0 0 .5rem 0; font-size:1.85rem !important; }
.hero p { color:#f0fff5; margin:0; font-size:1.07rem; line-height:1.5; }
.chip { display:inline-block; color:#195b3d; background:#e1f1e6; padding:.35rem .7rem; border-radius:1.4rem; font-size:.94rem; font-weight:700; margin-bottom:.5rem; }
@media (max-width:760px) {
 h1 {font-size:1.85rem !important;} h2 {font-size:1.45rem !important;}
 [data-testid="stMetricValue"] {font-size:1.65rem !important;}
 [data-testid="stMainBlockContainer"] {padding:1.1rem 1rem 2rem;}
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    if not MODEL_PATH.is_file():
        return None, "The trained five-model pickle has not been added to models/."
    try:
        with MODEL_PATH.open("rb") as fh:
            bundle = pickle.load(fh)  # Only load a pickle you trained or trust.
        bundle = restore_portable_xgboost(bundle)
        validate_bundle(bundle)
        return bundle, None
    except Exception as exc:
        import traceback
        # Keep the complete traceback in server logs to identify which model failed.
        print("MODEL LOADING ERROR\n" + traceback.format_exc(), flush=True)
        reason = str(exc)
        if "input stream corrupted" in reason.lower():
            reason += (". This often comes from XGBoost's internal pickle format. "
                       "Run the provided conversion script in the notebook environment "
                       "and replace the pickle in models/.")
        return None, f"The saved model cannot be loaded: {reason}"


@st.cache_data(show_spinner=False)
def load_dataset():
    if not DATA_PATH.is_file():
        return None
    try:
        return pd.read_excel(DATA_PATH)
    except Exception as exc:
        st.warning(f"The Excel dataset cannot be read: {exc}")
        return None


def show_plot(fig, height=390):
    fig.update_layout(
        height=height, margin=dict(l=30, r=24, t=65, b=36),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color="#254335", size=15),
        title_font=dict(size=20, color="#164b36"),
        legend=dict(font=dict(size=14)),
    )
    fig.update_xaxes(tickfont=dict(size=14), title_font=dict(size=16), gridcolor="#e3eee5")
    fig.update_yaxes(tickfont=dict(size=14), title_font=dict(size=16), gridcolor="#e3eee5")
    st.plotly_chart(fig, use_container_width=True)


def starting_values(bundle, dataset, sample_choice):
    row = {key: float(value) for key, value in bundle["numeric_defaults"].items()}
    row.update({"District": bundle["districts"][0], "Season": bundle["seasons"][0]})
    row["Year"] = int(bundle["trained_through_year"]) + 1
    actual = None
    if sample_choice != "Enter my own values" and dataset is not None:
        sample_id = int(sample_choice.split(" | ", 1)[0])
        chosen = dataset.loc[sample_id]
        for feature in bundle["features"]:
            row[feature] = chosen[feature]
        actual = float(chosen["Production_t"])
    return row, actual


INPUT_LABELS = {
    "SownArea_ha": "Sown area (hectares)",
    "PreviousYearProduction_t": "Previous-year production (tonnes)",
    "SeasonRainfall_mm": "Season rainfall (mm)",
    "SeasonMeanTemp_C": "Mean temperature (°C)",
    "SeasonMeanMaxTemp_C": "Mean maximum temperature (°C)",
    "SeasonMeanMinTemp_C": "Mean minimum temperature (°C)",
    "SeasonET0_mm": "Season evapotranspiration (mm)",
    "SeasonMeanRadiation_MJ_m2_day": "Solar radiation (MJ/m²/day)",
    "SeasonSunshineHours": "Sunshine hours (season)",
    "RainyDays": "Number of rainy days",
    "HotDays35C": "Number of hot days (≥35°C)",
}

bundle, bundle_error = load_model_bundle()
dataset = load_dataset()

st.sidebar.title("🌾 Paddy Production")
st.sidebar.markdown("**Sri Lanka · Maha & Yala**")
page = st.sidebar.radio(
    "Go to", ["📊 Dashboard", "🌾 Predict Production", "🔍 Model Insights"],
    label_visibility="visible",
)
st.sidebar.divider()
st.sidebar.markdown("**Available models**")
st.sidebar.write("Random Forest · XGBoost · CatBoost · LightGBM · EBM")
st.sidebar.caption("Completed-season harvest estimates using observed seasonal weather, not advance forecasts.")
if bundle is None:
    st.sidebar.warning("Model not yet loaded")
else:
    st.sidebar.success("Five trained models ready")

if page == "📊 Dashboard":
    st.title("Sri Lankan Paddy Production")
    st.markdown("""<div class="hero"><h2>Explore paddy harvest patterns</h2>
    <p>View historical production across districts, seasons and years, and explore how cultivated area relates to harvest volume.</p></div>""", unsafe_allow_html=True)
    if dataset is None:
        st.error("Dataset missing. Place paddy_weather_final.xlsx in data/raw/.")
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Records", f"{len(dataset):,}")
    m2.metric("Districts", f"{dataset['District'].nunique()}")
    m3.metric("Period", f"{int(dataset.Year.min())}–{int(dataset.Year.max())}")
    m4.metric("Seasons", f"{dataset.Season.nunique()}")
    st.divider()
    st.subheader("Filter the dataset")
    f1, f2, f3 = st.columns([1, 1.4, 2])
    seasons = f1.multiselect("Cultivation season", sorted(dataset.Season.unique()), default=sorted(dataset.Season.unique()))
    years = f2.slider("Year range", int(dataset.Year.min()), int(dataset.Year.max()),
                      (int(dataset.Year.min()), int(dataset.Year.max())))
    districts = f3.multiselect("Districts (empty = all)", sorted(dataset.District.unique()))
    df = dataset[dataset.Year.between(*years) & dataset.Season.isin(seasons)]
    if districts:
        df = df[df.District.isin(districts)]
    if df.empty:
        st.info("No records match your filters. Change the year, season, or district selection.")
        st.stop()

    left, right = st.columns(2, gap="large")
    with left:
        totals = df.groupby("Year", as_index=False).Production_t.sum()
        show_plot(px.line(totals, x="Year", y="Production_t", markers=True,
                          title="Harvest volume by year", labels={"Production_t":"Production (tonnes)"}))
    with right:
        totals = df.groupby("Season", as_index=False).Production_t.sum()
        show_plot(px.bar(totals, x="Season", y="Production_t", color="Season",
                         title="Recorded production by season", labels={"Production_t":"Production (tonnes)"}))
    left, right = st.columns(2, gap="large")
    with left:
        top = df.groupby("District", as_index=False).Production_t.sum().nlargest(12, "Production_t")
        show_plot(px.bar(top.sort_values("Production_t"), x="Production_t", y="District", orientation="h",
                         title="Highest-production districts", labels={"Production_t":"Production (tonnes)"}), 470)
    with right:
        show_plot(px.scatter(df, x="SownArea_ha", y="Production_t", color="Season",
                             hover_data=["District", "Year"], title="Sown area versus harvest",
                             labels={"SownArea_ha":"Sown area (hectares)", "Production_t":"Production (tonnes)"}), 470)
    with st.expander("View filtered records"):
        st.dataframe(df.head(100), hide_index=True, use_container_width=True)
        st.download_button("Download filtered records (.csv)",
                           df.to_csv(index=False).encode("utf-8"), file_name="filtered_paddy_records.csv", mime="text/csv")
    st.info("Totals reflect the selected dataset records; they are not automatically national production estimates.")

elif page == "🌾 Predict Production":
    st.title("Predict Seasonal Paddy Production")
    st.markdown("""<div class="hero"><h2>Estimate a district’s paddy harvest</h2>
    <p>Enter agricultural and completed-season weather observations, choose a model, and see the prediction and its explanation.</p></div>""", unsafe_allow_html=True)
    if bundle is None:
        st.error(bundle_error)
        st.markdown("**How to enable predictions:** Run the notebook through its last ‘Save to Pickle File’ cell, then upload the generated `paddy_production_all_five_models_ebm.pkl` to `models/`.")
        st.stop()

    samples = ["Enter my own values"]
    if dataset is not None:
        test_samples = dataset[dataset.Year.between(2022, 2023)]
        samples.extend(f"{idx} | {row.District} · {row.Season} {row.Year}"
                       for idx, row in test_samples.iterrows())
    example = st.selectbox("Input method", samples, help="Choose a test record to demonstrate a known actual result, or enter your own values.")
    defaults, actual = starting_values(bundle, dataset, example)
    st.caption("Try a 2022–2023 record to compare an estimated harvest with the observed harvest. Actual production is not used as an input.")

    with st.form("prediction_inputs"):
        choose1, choose2 = st.columns(2)
        selected = choose1.selectbox("Prediction model", list(EXPECTED_MODELS), index=4,
                                     help="EBM has built-in feature explanations. All five models can make predictions.")
        compare = choose2.checkbox("Compare all five model predictions", value=True)
        st.subheader("1. District and cultivation season")
        c1, c2, c3 = st.columns(3)
        district = c1.selectbox("District", bundle["districts"],
                                index=bundle["districts"].index(defaults["District"]) if defaults["District"] in bundle["districts"] else 0)
        season = c2.selectbox("Season", bundle["seasons"],
                              index=bundle["seasons"].index(defaults["Season"]) if defaults["Season"] in bundle["seasons"] else 0)
        year = c3.number_input("Year", min_value=2000, max_value=2050, value=int(defaults["Year"]), step=1)
        values = {"District": district, "Season": season, "Year": int(year)}
        st.subheader("2. Cultivation and weather details")
        st.caption("Use measurements for the completed cultivation season. Values are not automatically forecast by this app.")
        columns = st.columns(2, gap="large")
        for idx, feature in enumerate(x for x in bundle["numeric_columns"] if x != "Year"):
            default = defaults.get(feature)
            default = bundle["numeric_defaults"].get(feature, 0.0) if pd.isna(default) else default
            values[feature] = float(columns[idx % 2].number_input(
                INPUT_LABELS.get(feature, feature),
                value=float(default), min_value=0.0, format="%.2f", key=f"input_{example}_{feature}"))
        previous_missing = st.checkbox("Previous-year harvest unknown — use training-data median", value=False)
        submitted = st.form_submit_button("🌾 Predict paddy production", type="primary", use_container_width=True)

    if submitted:
        if previous_missing:
            values["PreviousYearProduction_t"] = float("nan")
        try:
            estimate, _ = predict_one(values, bundle, selected)
            comparisons = predict_all(values, bundle) if compare else None
            st.session_state["prediction_result"] = dict(
                values=values, model=selected, estimate=estimate, comparisons=comparisons, actual=actual,
                source=example,
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

    outcome = st.session_state.get("prediction_result")
    if outcome is not None:
        st.divider()
        st.markdown('<span class="chip">Prediction result</span>', unsafe_allow_html=True)
        st.subheader("Estimated paddy production")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{outcome['model']} prediction", f"{outcome['estimate']:,.0f} tonnes")
        if outcome["actual"] is not None:
            c2.metric("Observed harvest", f"{outcome['actual']:,.0f} tonnes")
            c3.metric("Prediction difference", f"{outcome['estimate']-outcome['actual']:+,.0f} tonnes")
        else:
            c2.metric("Cultivation season", outcome["values"]["Season"])
            c3.metric("District", outcome["values"]["District"])
        st.caption(f"Showing the latest submitted result for {outcome['values']['District']} · {outcome['values']['Season']} · {outcome['values']['Year']}.")
        if outcome["values"]["Year"] > int(bundle["trained_through_year"]):
            st.warning("This year comes after the training period. Predictions for new years may be less accurate.")
        if outcome["estimate"] < 0:
            st.warning("This model estimated a negative harvest, which is not physically meaningful.")

        if outcome["comparisons"] is not None:
            st.subheader("Compare all five models")
            table = outcome["comparisons"]
            st.dataframe(table.style.format({"Predicted production (tonnes)": "{:,.0f}"}),
                         hide_index=True, use_container_width=True)
            show_plot(px.bar(table, x="Model", y="Predicted production (tonnes)", color="Model",
                             title="Five-model harvest predictions", labels={"Predicted production (tonnes)":"Tonnes"}), 380)

        st.subheader("Understand the prediction")
        how1, how2 = st.tabs(["🌱 EBM feature explanation", "🔄 What-if changes for the selected model"])
        with how1:
            st.write("EBM explains a prediction as a baseline plus individual feature contributions. This explanation is **for EBM**, even if another model is selected above.")
            try:
                baseline, ebm_estimate, effects = ebm_explanation(outcome["values"], bundle)
                b1, b2 = st.columns(2)
                b1.metric("EBM prediction", f"{ebm_estimate:,.0f} tonnes")
                b2.metric("EBM baseline", f"{baseline:,.0f} tonnes")
                st.caption("Positive feature effects raise the EBM estimate; negative feature effects lower it. They do not establish cause and effect.")
                top = effects.head(12).sort_values("Contribution (tonnes)")
                show_plot(px.bar(top, x="Contribution (tonnes)", y="Feature / interaction",
                                 orientation="h", color="Contribution (tonnes)",
                                 color_continuous_scale="RdYlGn",
                                 title="Most important terms for this district-season"), 500)
                with st.expander("View all EBM term contributions"):
                    st.dataframe(effects.style.format({"Contribution (tonnes)":"{:+,.1f}"}),
                                 hide_index=True, use_container_width=True)
                if not np.isclose(baseline + effects["Contribution (tonnes)"].sum(), ebm_estimate, atol=1.0):
                    st.warning("EBM contributions do not exactly sum to this prediction; review the model link function.")
            except Exception as exc:
                st.warning(f"EBM explanation is unavailable: {exc}")
        with how2:
            st.write(f"See how **{outcome['model']}** changes its estimate when one numeric input is increased by 10%, keeping the others the same.")
            st.caption("This is a what-if sensitivity check, not SHAP/LIME or evidence of causality. Results depend on the chosen example and changes may be unrealistic.")
            try:
                sensitivity = scenario_sensitivity(outcome["values"], bundle, outcome["model"])
                if sensitivity.empty:
                    st.info("No suitable numeric inputs were available for this comparison.")
                else:
                    st.dataframe(sensitivity.style.format({"Prediction change (tonnes)":"{:+,.0f}"}),
                                 hide_index=True, use_container_width=True)
                    ranked = sensitivity.sort_values("Prediction change (tonnes)")
                    show_plot(px.bar(ranked, x="Prediction change (tonnes)", y="Feature", orientation="h",
                                     color="Prediction change (tonnes)", color_continuous_scale="RdYlGn",
                                     title="Predicted change after a +10% input adjustment"), 470)
            except Exception as exc:
                st.warning(f"Sensitivity comparison is unavailable: {exc}")
        st.info("Educational estimates using complete-season measurements. Do not use model results alone for planting or resource decisions.")
    else:
        st.info("Fill in the form and click **Predict paddy production** to see model predictions and explanations.")

else:
    st.title("Model Insights & Explainability")
    st.markdown("""<div class="hero"><h2>How accurate and explainable are the five models?</h2>
    <p>Explore the notebook's model evaluation, EBM global feature importance, and explanations for individual predictions.</p></div>""", unsafe_allow_html=True)
    if bundle is None:
        st.error(bundle_error)
        st.stop()
    evaluation, importance, local = st.tabs(["📊 Model results", "🌿 EBM feature importance", "🔎 Individual explanation"])
    with evaluation:
        test_results = pd.DataFrame(bundle["test_results"]).sort_values("Test MAE (tonnes)")
        best_test = test_results.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Chosen by validation", bundle["selected_by_validation"])
        c2.metric("Lowest test error", str(best_test["Model"]))
        c3.metric("Lowest test MAE", f"{best_test['Test MAE (tonnes)']:,.0f} tonnes")
        st.caption("The 2022–2023 test results are reported for comparison, not used to reselect the model.")
        st.dataframe(test_results.style.format({"Test MAE (tonnes)":"{:,.1f}",
                                                "Test RMSE (tonnes)":"{:,.1f}", "Test R²":"{:.3f}"}),
                     hide_index=True, use_container_width=True)
        show_plot(px.bar(test_results, x="Model", y="Test MAE (tonnes)", color="Model",
                         title="Unseen test error (2022–2023) — lower is better"))
        ebm = test_results[test_results.Model == "EBM"]
        if not ebm.empty:
            r = ebm.iloc[0]
            st.write(f"**EBM test result:** MAE {r['Test MAE (tonnes)']:,.1f} tonnes, RMSE {r['Test RMSE (tonnes)']:,.1f} tonnes, and R² {r['Test R²']:.3f}.")
        with st.expander("Validation and hyperparameter tuning table"):
            st.dataframe(pd.DataFrame(bundle["validation_results"]), hide_index=True, use_container_width=True)
        st.caption("MAE is average absolute error, RMSE gives larger errors greater weight, and R² describes the variation in production explained on this test set.")
    with importance:
        ranking = pd.DataFrame(bundle["ebm_global_importance"]).sort_values("Importance (model units)", ascending=False)
        top = ranking.head(12).sort_values("Importance (model units)")
        st.write("The EBM ranking shows which features or interactions most affect its predictions on average.")
        show_plot(px.bar(top, x="Importance (model units)", y="Term", orientation="h",
                         title="EBM global importance"), 540)
        st.dataframe(ranking.head(15).style.format({"Importance (model units)":"{:,.1f}"}),
                     hide_index=True, use_container_width=True)
        st.info("Importance is an association within the trained EBM model. It does not mean a feature causes production to change.")
    with local:
        previous = st.session_state.get("prediction_result")
        if previous is None:
            st.info("Visit **Predict Production**, enter values and click Predict. Your example's EBM explanation will appear here.")
        else:
            base, estimate, effects = ebm_explanation(previous["values"], bundle)
            c1, c2 = st.columns(2)
            c1.metric("EBM harvest estimate", f"{estimate:,.0f} tonnes")
            c2.metric("Baseline harvest", f"{base:,.0f} tonnes")
            st.caption(f"Explanation for {previous['values']['District']} · {previous['values']['Season']} · {previous['values']['Year']}")
            top = effects.head(10).sort_values("Contribution (tonnes)")
            show_plot(px.bar(top, x="Contribution (tonnes)", y="Feature / interaction",
                             orientation="h", color="Contribution (tonnes)", color_continuous_scale="RdYlGn",
                             title="Feature effects for this EBM estimate"), 470)
            st.dataframe(effects.style.format({"Contribution (tonnes)":"{:+,.1f}"}),
                         hide_index=True, use_container_width=True)
            st.caption("Positive values increase EBM's prediction; negative values reduce it. They are model explanations, not causal relationships.")

st.divider()
st.caption("Data sources identified in the project notebook: Department of Census and Statistics, Sri Lanka (paddy statistics), and Open-Meteo (weather data). Educational use only.")
