import joblib
import numpy as np
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "rf_models.joblib"

st.set_page_config(page_title="GPA Prediction", layout="centered")
st.title(":blue[STUDENT GPA PREDICTION]")

# --- User input ---
max_semester = 6
current_semester = st.selectbox("Current semester", list(range(1, max_semester + 1)))

gpa_inputs = []
for i in range(1, current_semester + 1):
    gpa = st.number_input(
        f"GPA – Semester {i}",
        min_value=0.0,
        max_value=4.0,
        step=0.01,
        format="%.2f"
    )
    gpa_inputs.append(gpa)


# --- Predict ---
if any(g == 0.0 for g in gpa_inputs):
    st.warning("⚠️ Please enter GPA for all completed semesters.")
else:
    try:
        input_data = np.array(gpa_inputs).reshape(1, -1)

        if current_semester < max_semester:
            model_list = joblib.load(model_path)
            model_index = current_semester - 1
            model = model_list[model_index]

            predicted_gpa = model.predict(input_data)[0]

            st.subheader(f"📘 Predicted GPA – Semester {current_semester + 1}")
            st.success(f"Predicted GPA: {predicted_gpa:.2f}")
        else:
            st.info("You are already at the final semester.")

    except Exception as e:
        st.error(f"❌ An error has occurred: {e}. Please try again.")