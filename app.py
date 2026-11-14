
import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(
    page_title="Heart Disease ML Project",
    page_icon="❤️",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("heart_disease_model.pkl")

@st.cache_data
def load_config():
    with open("feature_config.json", "r") as f:
        return json.load(f)

model = load_model()
config = load_config()

st.title("❤️ Heart Disease Prediction")
st.caption("Machine Learning Demonstration | Random Forest")

st.warning(
    "Educational demonstration only. This experimental model "
    "is not intended for medical diagnosis or treatment. "
    "Do not use its output to make health decisions."
)

st.markdown(
    "Enter the dataset features below to see how the trained "
    "machine-learning model classifies the supplied data."
)

with st.form("prediction_form"):
    st.subheader("Enter Feature Details")

    user_input = {}

    columns = st.columns(2)

    for index, (feature, details) in enumerate(config.items()):
        with columns[index % 2]:
            if details["type"] == "numeric":
                user_input[feature] = st.number_input(
                    feature,
                    min_value=float(details["min"]),
                    max_value=float(details["max"]),
                    value=float(details["default"]),
                    step=0.1,
                    key=feature
                )
            else:
                user_input[feature] = st.selectbox(
                    feature,
                    details["options"],
                    key=feature
                )

    submitted = st.form_submit_button(
        "Run ML Prediction",
        use_container_width=True
    )

if submitted:
    input_df = pd.DataFrame([user_input])

    prediction = int(model.predict(input_df)[0])
    scores = model.predict_proba(input_df)[0]

    st.divider()
    st.subheader("Model Output")

    if prediction == 1:
        st.error("Model class output: Yes (Class 1)")
    else:
        st.success("Model class output: No (Class 0)")

    st.write(
        "This is the model's classification, not a medical assessment."
    )

    st.subheader("Model Scores")

    score_df = pd.DataFrame({
        "Class": ["No (0)", "Yes (1)"],
        "Model score": scores
    })

    st.bar_chart(
        score_df.set_index("Class")
    )

    st.caption(
        "Scores are classifier outputs and are not validated "
        "medical probabilities or estimates of individual health risk."
    )

    with st.expander("View submitted input"):
        st.dataframe(input_df, use_container_width=True)

st.divider()
st.caption("Academic Machine Learning Project | Random Forest Classifier")
