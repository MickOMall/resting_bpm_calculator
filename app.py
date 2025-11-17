import streamlit as st
import pandas as pd
import numpy as np
import joblib
from treeinterpreter import treeinterpreter as ti
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import os

# -----------------------------
# Load model and training data
# -----------------------------
model = joblib.load("model/resting_bpm_model.joblib")
cols = pd.read_csv("model/columns.csv").iloc[:, 0].tolist()

# -----------------------------
# Explainers
# -----------------------------
def explain(pred, top_features, top_contributions):
    explanation = "### Crew-AI Analysis Report\n"
    explanation += (
        f"\nThe model predicts your **Resting BPM** to be **{pred:.1f}**. "
        "Below is a detailed breakdown of how the major input features influenced this prediction.\n\n"
        "#### Key Influencing Factors\n"
    )
    for f, c in zip(top_features, top_contributions):
        direction = "increased" if c > 0 else "decreased"
        explanation += f"- **{f}** {direction} the predicted BPM by **{c:+.2f}**\n"
    explanation += (
        "\nThis explanation highlights the internal decision-making of the random forest model, "
        "showing a transparent breakdown of contributing factors."
    )
    return explanation

def explain_model_performance(r2, mae, mape):
    explanation = "### Crew-AI Model Performance Analysis\n\n"
    explanation += f"- **R² = {r2:.3f}**: Closer to 1.0 is better.\n"
    explanation += f"- **MAE = {mae:.2f} BPM**: Average deviation from actual BPM.\n"
    explanation += f"- **MAPE = {mape*100:.2f}%**: Average percent error.\n\n"
    return explanation

# -----------------------------
# Prediction function
# -----------------------------
def predict_resting_bpm(input_dict):
    df_input = pd.DataFrame([input_dict])
    bool_cols = df_input.select_dtypes(include='bool').columns
    df_input[bool_cols] = df_input[bool_cols].astype(int)
    df_input = pd.get_dummies(df_input, drop_first=True)
    df_input = df_input.reindex(columns=cols, fill_value=0)
    
    pred = model.predict(df_input)[0]
    _, bias, contributions = ti.predict(model, df_input)
    contribs = contributions[0]
    top_idx = np.argsort(np.abs(contribs))[-5:][::-1]
    top_features = df_input.columns[top_idx]
    top_contributions = contribs[top_idx]

    short_explanation = ", ".join([f"{f} ({c:+.2f})" for f, c in zip(top_features, top_contributions)])
    verbose_explanation = explain(pred, top_features, top_contributions)

    return {"predicted_bpm": pred, "top_features": short_explanation, "verbose": verbose_explanation}

# -----------------------------
# UI - Main
# -----------------------------
st.title("Resting Heart Rate Predictor")
st.write("Enter your health variables to predict your resting heart rate.")

# -----------------------------
# Sidebar - Utilities
# -----------------------------
st.sidebar.header("Utility Calculators")
# Gallons → Liters
gallons = st.sidebar.number_input("Gallons → Liters", min_value=0.0, step=0.1)
st.sidebar.write(f"Liters: **{gallons * 3.785:.3f}**")
# Pounds → Kilograms
lbs = st.sidebar.number_input("Pounds → Kilograms", min_value=0.0, step=0.1)
st.sidebar.write(f"Kilograms: **{lbs * 0.453592:.3f}**")
# Inches → Meters
inches = st.sidebar.number_input("Inches → Meters", min_value=0.0, step=0.1)
st.sidebar.write(f"Meters: **{inches * 0.0254:.3f}**")
# BMI Calculator
st.sidebar.subheader("BMI Calculator")
kg = st.sidebar.number_input("Weight (kg)", min_value=0.0)
m = st.sidebar.number_input("Height (m)", min_value=0.0)
if m > 0:
    bmi = kg / (m ** 2)
    st.sidebar.write(f"**BMI:** {bmi:.2f}")

# Model Performance Metrics
st.sidebar.header("Model Performance")
if os.path.exists("test_data.csv") and os.path.exists("test_labels.csv"):
    X_test = pd.read_csv("test_data.csv")
    y_test = pd.read_csv("test_labels.csv").iloc[:, 0]
    X_test = pd.get_dummies(X_test, drop_first=True)
    X_test = X_test.reindex(columns=cols, fill_value=0)
    pred_test = model.predict(X_test)
    r2 = r2_score(y_test, pred_test)
    mae = mean_absolute_error(y_test, pred_test)
    mape = mean_absolute_percentage_error(y_test, pred_test)
    st.sidebar.write(f"**R²:** {r2:.3f}")
    st.sidebar.write(f"**MAE:** {mae:.2f}")
    st.sidebar.write(f"**MAPE:** {mape*100:.2f}%")
    st.sidebar.markdown(explain_model_performance(r2, mae, mape))
else:
    st.sidebar.write("Test data unavailable. Metrics cannot be computed.")

# -----------------------------
# User Input
# -----------------------------
user_input = {}
st.subheader("Input Your Variables")
for col in cols:
    if col.lower().startswith("is_") or "bool" in col.lower():
        user_input[col] = st.checkbox(col)
    else:
        user_input[col] = st.number_input(col, value=0.0)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict"):
    result = predict_resting_bpm(user_input)
    st.session_state["prediction_result"] = result
    st.success(f"### Predicted Resting BPM: **{result['predicted_bpm']:.1f}**")
    st.info(f"**Top Contributing Features:** {result['top_features']}")
    st.write(result["verbose"])

# -----------------------------
# Enhanced Feedback System
# -----------------------------
st.header("Help Improve the Model")
st.write("Provide feedback so we can improve future predictions.")

feedback_file = "model/feedback.csv"
os.makedirs("model", exist_ok=True)

result = st.session_state.get("prediction_result", None)

if result:
    with st.form("feedback_form"):
        thumbs = st.radio("Was the predicted BPM accurate?", ["Yes", "No"])
        actual_bpm = None
        if thumbs == "No":
            actual_bpm = st.number_input(
                "Please provide your actual resting BPM:",
                min_value=0.0, max_value=200.0, step=1.0
            )
        feedback_text = st.text_area("Additional feedback (optional):")
        submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            if thumbs == "No" and (actual_bpm is None or actual_bpm == 0.0):
                st.error("Please provide your actual BPM before submitting.")
            else:
                bpm_to_save = result['predicted_bpm'] if thumbs == "Yes" else actual_bpm
                feedback_entry = user_input.copy()
                feedback_entry.update({
                    "Resting_BPM": bpm_to_save,
                    "feedback_text": feedback_text,
                    "thumbs": thumbs,
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                feedback_df = pd.DataFrame([feedback_entry])

                # Drop mismatched columns if CSV exists
                if os.path.exists(feedback_file):
                    existing_df = pd.read_csv(feedback_file)
                    feedback_df = feedback_df[[c for c in feedback_df.columns if c in existing_df.columns]]
                    feedback_df.to_csv(feedback_file, mode="a", header=False, index=False)
                else:
                    feedback_df.to_csv(feedback_file, index=False)

                st.success("✅ Thank you! Your feedback has been recorded.")
else:
    st.info("Please make a prediction first to provide feedback.")
