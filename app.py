"""
Child Stunting Risk Predictor — Kenya
Runs the same way online or offline: `streamlit run app.py`
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "stunting_classifier.joblib"
METADATA_PATH = APP_DIR / "model_metadata.json"

st.set_page_config(page_title="Child Stunting Risk Predictor", page_icon="🧒", layout="centered")


@st.cache_resource
def load_model():
    pipeline = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return pipeline, metadata


pipeline, metadata = load_model()
CATS = metadata["category_values"]

# ---------------------------------------------------------------------------
# Dietary / care guidance shown alongside each prediction. Not a diagnosis —
# a starting point for a conversation with a health worker.
# ---------------------------------------------------------------------------
GUIDANCE = {
    "Severe stunting": {
        "headline": "This suggests a high level of concern.",
        "advice": [
            "See a health worker or clinic as soon as possible for a full growth assessment.",
            "Ask about therapeutic or supplementary feeding programmes available locally "
            "(e.g. RUTF/RUSF where indicated).",
            "Increase meal frequency and protein/energy density (e.g. add oil, groundnut "
            "paste, eggs, animal-source foods where accessible) under a health worker's guidance.",
            "Rule out and treat any underlying illness or infection with a clinician's help.",
        ],
    },
    "Moderate stunting": {
        "headline": "This suggests some concern that's worth following up on.",
        "advice": [
            "Schedule a growth-monitoring visit at the nearest health facility.",
            "Increase dietary diversity: aim for foods from multiple groups each day "
            "(grains, legumes, dairy, fruits/vegetables, animal-source protein where possible).",
            "If the child is under 6 months, prioritise exclusive breastfeeding; if older, "
            "continue breastfeeding alongside nutritious complementary foods.",
            "Re-check growth again in 4-6 weeks to see whether the trend is improving.",
        ],
    },
    "Normal growth": {
        "headline": "Growth looks on track.",
        "advice": [
            "Continue the current feeding routine — it appears to be working.",
            "Keep up regular growth-monitoring visits so any change is caught early.",
            "Maintain dietary diversity and continued breastfeeding if the child is under 2.",
        ],
    },
}

st.title("🧒 Child Stunting Risk Predictor")
st.caption(
    "Estimates a Kenyan child's stunting risk category (WHO height-for-age z-score bands) "
    "from household and health-history information. **This is a screening aid, not a "
    "diagnosis** — always confirm with an actual height/age measurement and a health worker."
)

mode = st.radio("Who's filling this in?", ["Parent / caregiver", "Health worker"], horizontal=True)
is_medic = mode == "Health worker"

st.divider()

with st.form("predict_form"):
    st.subheader("About the household")

    county = st.selectbox("County", options=[c.title() for c in CATS["v024_label"]])

    if is_medic:
        residence = st.radio("Type of place of residence", ["Urban", "Rural"], horizontal=True)
    else:
        residence = st.radio("Do you live in a town/city or a village?", ["Town / city", "Village"], horizontal=True)
        residence = "Urban" if residence == "Town / city" else "Rural"

    if is_medic:
        education = st.selectbox("Mother's highest education level",
                                  ["No education", "Primary", "Secondary", "Higher"])
    else:
        education = st.selectbox("How far did the child's mother go in school?",
                                  ["Never went to school", "Primary school", "Secondary school",
                                   "College or university"])
        education = {"Never went to school": "No education", "Primary school": "Primary",
                     "Secondary school": "Secondary", "College or university": "Higher"}[education]

    if is_medic:
        wealth = st.selectbox("Household wealth quintile", ["Poorest", "Poorer", "Middle", "Richer", "Richest"])
    else:
        wealth = st.select_slider(
            "Roughly how would you describe your household's financial situation?",
            options=["Struggling a lot", "Struggling somewhat", "Getting by", "Comfortable", "Well-off"],
        )
        wealth = {"Struggling a lot": "Poorest", "Struggling somewhat": "Poorer", "Getting by": "Middle",
                  "Comfortable": "Richer", "Well-off": "Richest"}[wealth]

    if is_medic:
        water = st.selectbox("Source of drinking water", [c.title() for c in CATS["v113_label"]])
    else:
        water_simple = st.selectbox(
            "Where does your household mainly get drinking water?",
            ["Piped into the house or yard", "A protected well, borehole, or spring",
             "An unprotected well or spring", "A river, lake, or pond", "Rainwater", "Bottled or tanker water"],
        )
        water_map = {
            "Piped into the house or yard": "piped into dwelling",
            "A protected well, borehole, or spring": "protected well",
            "An unprotected well or spring": "unprotected well",
            "A river, lake, or pond": "river/dam/lake/ponds/stream/canal/irrigation channel",
            "Rainwater": "rainwater",
            "Bottled or tanker water": "bottled water",
        }
        water = water_map[water_simple]

    if is_medic:
        toilet = st.selectbox("Type of toilet facility", [c.title() for c in CATS["v116_label"]])
    else:
        toilet_simple = st.selectbox(
            "What kind of toilet does your household use?",
            ["Flush toilet", "Covered pit latrine", "Open pit latrine (no cover)", "No toilet / open ground"],
        )
        toilet_map = {
            "Flush toilet": "flush to septic tank",
            "Covered pit latrine": "pit latrine with slab",
            "Open pit latrine (no cover)": "pit latrine without slab/open pit",
            "No toilet / open ground": "no facility/bush/field",
        }
        toilet = toilet_map[toilet_simple]

    st.subheader("About this child")

    birth_order = st.number_input(
        "Birth order (1 = first child, 2 = second, etc.)" if is_medic
        else "Is this your 1st, 2nd, 3rd child, and so on?",
        min_value=1, max_value=20, value=1, step=1,
    )

    if birth_order == 1:
        birth_spacing = None
        st.caption("First child — no previous birth to space from.")
    else:
        birth_spacing = st.number_input(
            "Preceding birth interval, in months" if is_medic
            else "How many months between this child's birth and the one before?",
            min_value=0, max_value=240, value=24, step=1,
        )

    know_birthweight = st.checkbox("I know the child's birth weight", value=False)
    birthweight = None
    if know_birthweight:
        birthweight = st.number_input("Birth weight (kg)", min_value=0.5, max_value=7.0, value=3.0, step=0.1)

    breastfeeding_months = st.number_input(
        "Total months breastfed so far (0 if never breastfed / not yet started)"
        if is_medic else
        "How many months has the child been breastfed in total?",
        min_value=0, max_value=60, value=6, step=1,
    )

    know_anc = st.checkbox("I know how many antenatal (pregnancy) checkups the mother had", value=False)
    anc_visits = None
    if know_anc:
        anc_visits = st.number_input(
            "Number of antenatal care visits during this pregnancy",
            min_value=0, max_value=30, value=4, step=1,
        )

    bcg = polio_doses = dptih_doses = rota_doses = None
    if is_medic:
        st.subheader("Vaccination history")
        st.caption("From the child's vaccination card, if available.")

        bcg_choice = st.radio("BCG", ["Not given", "Given"], horizontal=True, index=0)
        bcg = 1 if bcg_choice == "Given" else 0

        polio_doses = st.slider("Polio doses received (of 3)", min_value=0, max_value=3, value=0)
        dptih_doses = st.slider("DPT-HepB-Hib / Pentavalent doses received (of 3)", min_value=0, max_value=3, value=0)
        rota_doses = st.slider("Rotavirus doses received (of 2)", min_value=0, max_value=2, value=0)

    submitted = st.form_submit_button("Predict stunting risk", use_container_width=True, type="primary")

if submitted:
    row = pd.DataFrame([{
        "bord": birth_order,
        "birth_spacing_months": birth_spacing if birth_spacing is not None else np.nan,
        "birthweight_kg": birthweight if birthweight is not None else np.nan,
        "breastfeeding_months": breastfeeding_months,
        "anc_visits": anc_visits if anc_visits is not None else np.nan,
        "bcg_received": bcg if bcg is not None else np.nan,
        "polio_doses": polio_doses if polio_doses is not None else np.nan,
        "dptih_doses": dptih_doses if dptih_doses is not None else np.nan,
        "rotavirus_doses": rota_doses if rota_doses is not None else np.nan,
        "v024_label": county.lower(),
        "v025_label": residence.lower(),
        "v106_label": education.lower(),
        "v190_label": wealth.lower(),
        "v113_label": water.lower(),
        "v116_label": toilet.lower(),
    }])

    proba = pipeline.predict_proba(row)[0]
    classes = pipeline.classes_
    pred = classes[np.argmax(proba)]
    prob_by_class = dict(zip(classes, proba))

    st.divider()
    st.subheader("Result")

    color = {"Severe stunting": "🔴", "Moderate stunting": "🟠", "Normal growth": "🟢"}[pred]
    st.markdown(f"### {color} {pred}")
    st.write(GUIDANCE[pred]["headline"])

    order = ["Severe stunting", "Moderate stunting", "Normal growth"]
    chart_df = pd.DataFrame({
        "Category": order,
        "Model confidence": [prob_by_class.get(c, 0) for c in order],
    }).set_index("Category")
    st.bar_chart(chart_df)

    st.markdown("**What this could mean:**")
    for item in GUIDANCE[pred]["advice"]:
        st.markdown(f"- {item}")

    with st.expander("How reliable is this prediction?"):
        st.write(
            f"On held-out test data, this model reaches a macro-F1 of "
            f"{metadata['test_macro_f1']:.2f} and ROC-AUC of {metadata['test_roc_auc']:.2f} "
            f"(a model that always guesses the most common category scores "
            f"{metadata['baseline_macro_f1']:.2f} macro-F1 for comparison). "
            "It performs noticeably better at recognising typical growth than at catching "
            "every case of severe stunting — a health worker's own assessment, including an "
            "actual height and weight measurement, should always take priority over this tool."
        )

    st.info(
        "This tool does not replace a clinical growth assessment. If you're worried about "
        "a child's growth, please visit the nearest health facility.",
        icon="ℹ️",
    )
