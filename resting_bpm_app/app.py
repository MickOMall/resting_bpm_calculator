import streamlit as st
import pandas as pd
import numpy as np
import joblib
from treeinterpreter import treeinterpreter as ti
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import os

# Load model and data
model = joblib.load("model/resting_bpm_model.joblib")
cols = pd.read_csv("model/columns.csv").iloc[:, 0].tolist()

# Crew AI explanation 
def explain(pred, top_features, top_contributions):
    explanation = "### Crew-AI Analysis Report\n"
    explanation += (
        f"\nThe model predicts your **Resting BPM** to be **{pred:.1f}**.\n\n"
        "This value is produced by a Random Forest regression model. A Random Forest consists of many decision trees, "
        "each making its own prediction based on your inputs. The final BPM estimate is the average of all trees, "
        "making it more stable and resistant to noise than a single decision tree.\n\n"
        "The model’s prediction can be decomposed into contributions from each feature. These contributions represent "
        "how much each input pushed the prediction above or below the model’s baseline resting heart rate.\n\n"
        "#### Key Influencing Factors (detailed breakdown)\n"
    )

    for f, c in zip(top_features, top_contributions):
        direction = "increased" if c > 0 else "decreased"
        explanation += (
            f"- **{f}** {direction} the predicted BPM by **{c:+.2f}**.\n"
            f"  - This number reflects the aggregated influence of this feature across all decision trees.\n"
            f"  - A higher magnitude means the model relied more heavily on this feature.\n"
        )

    explanation += (
        "\n#### How to interpret these contributions\n"
        "- A **positive contribution** indicates that the feature pushed the predicted BPM higher.\n"
        "- A **negative contribution** indicates it lowered the prediction.\n"
        "- The **size** of the contribution reflects how strongly that feature shaped the model’s decision.\n\n"
        "These values come from TreeInterpreter, which breaks down the Random Forest’s internal decision paths into "
        "feature-level contributions. This gives a transparent, human-readable explanation of *why* the model reached "
        "its final prediction instead of functioning as a black-box.\n\n"
        "Overall, these insights reveal the mathematical reasoning behind your resting BPM prediction, enabling "
        "a clear understanding of which factors the model considered most important."
    )

    return explanation

#  model performance explanation
def explain_model_performance(r2, mae, mape):
    explanation = "### Crew-AI Model Performance Analysis\n\n"
    explanation += f"- **R² = {r2:.3f}**: This measures how well the model explains the variability in resting BPM.\n"
    if r2 < 0.5:
        explanation += "  - The model explains only a small portion of the variance. Predictions may be unreliable.\n"
    elif r2 < 0.8:
        explanation += "  - The model captures a moderate amount of the variance. Predictions are somewhat reliable.\n"
    else:
        explanation += "  - The model explains most of the variance. Predictions are highly reliable.\n"

    explanation += f"\n- **MAE = {mae:.2f} BPM**: On average, the model's predictions deviate from the actual BPM by this many beats per minute.\n"
    if mae > 5:
        explanation += "  - There can be noticeable deviations in individual predictions.\n"
    else:
        explanation += "  - Predictions are generally very close to actual BPM values.\n"

    explanation += f"\n- **MAPE = {mape*100:.2f}%**: This is the average percent error.\n"
    if mape > 10:
        explanation += "  - Percent errors are relatively high, meaning some predictions could be off by a significant fraction.\n"
    else:
        explanation += "  - Percent errors are low, indicating the model is precise across a wide range of BPM values.\n"

    explanation += "\nThese metrics provide a clear understanding of the model’s overall accuracy, reliability, and areas where it may need improvement."
    return explanation

# Prediction function
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

# UI
st.title("Resting Heart Rate Predictor")
st.write("Enter your health variables to predict your resting heart rate.")

# sidebar
st.sidebar.header("Utility Calculators")

# Weight converter
lbs = st.sidebar.number_input("Weight (lbs)", min_value=0.0, step=0.1)
kg = lbs * 0.453592
st.sidebar.write(f"Kilograms: **{kg:.3f}**")

# Height converter
inches = st.sidebar.number_input("Height (inches)", min_value=0.0, step=0.1)
meters = inches * 0.0254
st.sidebar.write(f"Meters: **{meters:.3f}**")

# BMI calculator automatically using converters
bmi = None
if kg > 0 and meters > 0:
    bmi = kg / (meters ** 2)
    st.sidebar.write(f"**BMI:** {bmi:.2f}")

# Other 
gallons = st.sidebar.number_input("Gallons → Liters", min_value=0.0, step=0.1)
st.sidebar.write(f"Liters: **{gallons * 3.785:.3f}**")

# performance
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

# main inputs
st.subheader("Input Your Variables")
user_input = {}
for col in cols:
    if "diet" in col.lower():
        continue
    if col.lower() == "bmi" and bmi is not None:
        user_input[col] = bmi
    elif col.lower().startswith("is_") or "bool" in col.lower():
        user_input[col] = st.checkbox(col)
    else:
        if col.lower() not in ["weight", "height"]:
            user_input[col] = st.number_input(col, value=0.0)

# pred button
if st.button("Predict"):
    result = predict_resting_bpm(user_input)
    st.session_state["prediction_result"] = result
    st.success(f"### Predicted Resting BPM: **{result['predicted_bpm']:.1f}**")
    st.info(f"**Top Contributing Features:** {result['top_features']}")
    st.write(result["verbose"])

# HITL
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
            actual_bpm = st.number_input("Please provide your actual resting BPM:", min_value=0.0, max_value=200.0, step=1.0)
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

                # feedback
                if os.path.exists(feedback_file) and os.path.getsize(feedback_file) > 0:
                    try:
                        existing_df = pd.read_csv(feedback_file)

                        # Add missing cols to existing
                        for col in feedback_df.columns:
                            if col not in existing_df.columns:
                                existing_df[col] = np.nan

                        # Add missing to new row
                        for col in existing_df.columns:
                            if col not in feedback_df.columns:
                                feedback_df[col] = np.nan

                        feedback_df = feedback_df[existing_df.columns]
                        feedback_df.to_csv(feedback_file, mode="a", header=False, index=False)

                    except pd.errors.EmptyDataError:
                        feedback_df.to_csv(feedback_file, index=False)

                else:
                    feedback_df.to_csv(feedback_file, index=False)

                st.success("Thank you! Your feedback has been recorded.")

else:
    st.info("Please make a prediction first to provide feedback.")
