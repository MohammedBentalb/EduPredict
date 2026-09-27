import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

st.title("EduPredict")

pipeline = joblib.load(ROOT / "models" / "best_model.joblib")

hours_studied = st.slider("Hours studied", 1, 44, 20)
attendance = st.slider("Attendance", 60, 100, 80)
sleep_hours = st.slider("Sleep hours", 4, 10, 7)
previous_scores = st.slider("Previous scores", 50, 100, 75)
tutoring_sessions = st.slider("Tutoring sessions", 0, 8, 1)
physical_activity = st.slider("Physical activity", 0, 6, 3)

parental_involvement = st.selectbox("Parental involvement", ["Low", "Medium", "High"])
access_to_resources = st.selectbox("Access to resources", ["Low", "Medium", "High"])
motivation_level = st.selectbox("Motivation level", ["Low", "Medium", "High"])
family_income = st.selectbox("Family income", ["Low", "Medium", "High"])
teacher_quality = st.selectbox("Teacher quality", ["Low", "Medium", "High"])
parental_education_level = st.selectbox("Parental education level", ["High School", "College", "Postgraduate"])
distance_from_home = st.selectbox("Distance from home", ["Near", "Moderate", "Far"])

gender = st.selectbox("Gender", ["Female", "Male"])
school_type = st.selectbox("School type", ["Private", "Public"])
extracurricular_activities = st.selectbox("Extracurricular activities", ["No", "Yes"])
internet_access = st.selectbox("Internet access", ["No", "Yes"])
learning_disabilities = st.selectbox("Learning disabilities", ["No", "Yes"])
peer_influence = st.selectbox("Peer influence", ["Negative", "Neutral", "Positive"])

if st.button("Predict exam score"):

    student = pd.DataFrame([{
        "Hours_Studied": hours_studied,
        "Attendance": attendance,
        "Sleep_Hours": sleep_hours,
        "Previous_Scores": previous_scores,
        "Tutoring_Sessions": tutoring_sessions,
        "Physical_Activity": physical_activity,
        "Parental_Involvement": parental_involvement,
        "Access_to_Resources": access_to_resources,
        "Motivation_Level": motivation_level,
        "Family_Income": family_income,
        "Teacher_Quality": teacher_quality,
        "Parental_Education_Level": parental_education_level,
        "Distance_from_Home": distance_from_home,
        "Gender": gender,
        "School_Type": school_type,
        "Extracurricular_Activities": extracurricular_activities,
        "Internet_Access": internet_access,
        "Learning_Disabilities": learning_disabilities,
        "Peer_Influence": peer_influence,
    }])

    score = pipeline.predict(student)[0]
    st.metric("Estimated exam score", f"{score:.1f} / 100")

    if score < 65:
        st.warning("At-risk student")
    else:
        st.success("Not at risk")









