import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Earthquake Damage Predictor",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# Load Files
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "Earthquake_15Feature_Model.pkl")
)

scaler = joblib.load(
    os.path.join(BASE_DIR, "Earthquake_15Feature_Scaler.pkl")
)

columns = joblib.load(
    os.path.join(BASE_DIR, "Earthquake_15Feature_Columns.pkl")
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📊 Project Information")

st.sidebar.info(
    """
    **Earthquake Damage Prediction**

    Model: XGBoost Classifier

    Features Used: 15

    Dataset:
    DrivenData Nepal Earthquake Dataset

    Target:
    Predict building damage after an earthquake.
    """
)

st.sidebar.metric("Training Records", "260,601")

st.sidebar.subheader("🏆 Best XGBoost Parameters")

st.sidebar.code("""
n_estimators = 300
max_depth = 5
learning_rate = 0.1
subsample = 1.0
colsample_bytree = 0.8
""")

st.sidebar.subheader("📈 Model Performance")
st.sidebar.metric("F1 Macro Score", f"{0.6502889730008033:.4f}")

st.sidebar.metric("Accuracy", "69%")

# -----------------------------
# Main Title
# -----------------------------
st.title("🏠 Earthquake Damage Predictor")

st.markdown("""
### 📖 About This Project

This application predicts earthquake damage levels using building characteristics from the Nepal Earthquake dataset.

**Model:** XGBoost Classifier  
**Features Used:** 15 Selected Features  
**Target:** Damage Grade (1 = Low, 2 = Medium, 3 = High)
""")

st.write(
    "Predict the likely damage level of a building based on structural and location characteristics."
)

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns(2)

with col1:

    st.subheader("📍 Location & Building Details")

    geo_level_1_id = st.number_input(
        "Geo Level 1 ID",
        min_value=0,
        value=8
    )

    geo_level_2_id = st.number_input(
        "Geo Level 2 ID",
        min_value=0,
        value=396
    )

    count_floors_pre_eq = st.number_input(
        "Number of Floors",
        min_value=1,
        value=2
    )

    age = st.number_input(
        "Building Age (Years)",
        min_value=0,
        value=15
    )

    area_percentage = st.number_input(
        "Area Percentage",
        min_value=0,
        value=4
    )

    height_percentage = st.number_input(
        "Height Percentage",
        min_value=0,
        value=7
    )

    count_families = st.number_input(
        "Number of Families",
        min_value=1,
        value=1
    )

with col2:

    st.subheader("🏗️ Construction Details")

    foundation_options = {
        "Type H": "h",
        "Type I": "i",
        "Type R": "r",
        "Type U": "u",
        "Type W": "w"
    }

    foundation_label = st.selectbox(
        "Foundation Type",
        list(foundation_options.keys())
    )

    foundation_type = foundation_options[foundation_label]

    roof_type = st.selectbox(
        "Roof Type",
        ["n", "q", "x"]
    )

    ground_floor_type = st.selectbox(
        "Ground Floor Type",
        ["f", "m", "v", "x", "z"]
    )

    other_floor_type = st.selectbox(
        "Other Floor Type",
        ["j", "q", "s", "x"]
    )

    has_superstructure_mud_mortar_stone = st.selectbox(
        "Mud Mortar Stone",
        [0, 1]
    )

    has_superstructure_stone_flag = st.selectbox(
        "Stone Flag",
        [0, 1]
    )

    has_superstructure_cement_mortar_brick = st.selectbox(
        "Cement Mortar Brick",
        [0, 1]
    )

    has_secondary_use = st.selectbox(
        "Secondary Use",
        [0, 1]
    )

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Predict Damage"):

    input_df = pd.DataFrame({
        'geo_level_1_id': [geo_level_1_id],
        'geo_level_2_id': [geo_level_2_id],
        'count_floors_pre_eq': [count_floors_pre_eq],
        'age': [age],
        'area_percentage': [area_percentage],
        'height_percentage': [height_percentage],
        'foundation_type': [foundation_type],
        'roof_type': [roof_type],
        'ground_floor_type': [ground_floor_type],
        'other_floor_type': [other_floor_type],
        'has_superstructure_mud_mortar_stone': [has_superstructure_mud_mortar_stone],
        'has_superstructure_stone_flag': [has_superstructure_stone_flag],
        'has_superstructure_cement_mortar_brick': [has_superstructure_cement_mortar_brick],
        'count_families': [count_families],
        'has_secondary_use': [has_secondary_use]
    })

    # -----------------------------
    # One Hot Encoding
    # -----------------------------
    input_df = pd.get_dummies(
        input_df,
        columns=[
            'foundation_type',
            'roof_type',
            'ground_floor_type',
            'other_floor_type'
        ],
        drop_first=True
    )

    # -----------------------------
    # Align Columns
    # -----------------------------
    input_df = input_df.reindex(
        columns=columns,
        fill_value=0
    )

    # -----------------------------
    # Log Transform
    # -----------------------------
    input_df[['age', 'area_percentage']] = np.log1p(
        input_df[['age', 'area_percentage']]
    )

    # -----------------------------
    # Scale
    # -----------------------------
    input_scaled = scaler.transform(input_df)

    # -----------------------------
    # Predict
    # -----------------------------
    pred = model.predict(input_scaled)[0]

    proba = model.predict_proba(input_scaled)[0]

    st.divider()

    st.subheader("📈 Prediction Result")

    if pred == 0:

        st.success("🟢 Grade 1 : Low Damage")

        st.info(
            """
            Low structural damage expected.

            Recommendation:
            • Routine inspection
            • Continue normal occupancy
            """
        )

    elif pred == 1:

        st.warning("🟡 Grade 2 : Medium Damage")

        st.info(
            """
            Moderate damage risk.

            Recommendation:
            • Structural assessment advised
            • Monitor for cracks and instability
            """
        )

    else:

        st.error("🔴 Grade 3 : High Damage")

        st.info(
            """
            Severe structural damage likely.

            Recommendation:
            • Immediate inspection
            • Consider evacuation
            • Emergency response planning
            """
        )

    # -----------------------------
    # Prediction Probabilities
    # -----------------------------
    st.subheader("📊 Prediction Probabilities")

    st.write(f"🟢 Grade 1 (Low Damage): {proba[0]*100:.2f}%")
    st.progress(float(proba[0]))

    st.write(f"🟡 Grade 2 (Medium Damage): {proba[1]*100:.2f}%")
    st.progress(float(proba[1]))

    st.write(f"🔴 Grade 3 (High Damage): {proba[2]*100:.2f}%")
    st.progress(float(proba[2]))


    
st.markdown("---")
st.markdown(
    "Developed by **Jeyashri S A** | Data Science EarthQuake Project"
)