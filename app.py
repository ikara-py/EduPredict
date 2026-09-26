import streamlit as st
import pandas as pd
import joblib

artifact = joblib.load("models/final_model.joblib")

model = artifact["model"]
feature_columns = artifact["feature_columns"]

st.title("EduPredict")
st.write("Predict a student's exam score")

hours_studied = st.number_input("Hours Studied", min_value=0.0, step=1.0)
attendance = st.number_input("Attendance", min_value=0.0, max_value=100.0, step=1.0)
sleep_hours = st.number_input("Sleep Hours", min_value=1, step=0.5)
previous_scores = st.number_input("Previous Scores", min_value=0.0, step=1.0)
tutoring_sessions = st.number_input("Tutoring Sessions", min_value=0, step=1.0)
physical_activity = st.number_input("Physical Activity", min_value=0, step=1.0)

parental_involvement = st.selectbox(
    "Parental Involvement",
    ["Low", "Medium", "High"]
)

access_to_resources = st.selectbox(
    "Access to Resources",
    ["Low", "Medium", "High"]
)

motivation_level = st.selectbox(
    "Motivation Level",
    ["Low", "Medium", "High"]
)

family_income = st.selectbox(
    "Family Income",
    ["Low", "Medium", "High"]
)

teacher_quality = st.selectbox(
    "Teacher Quality",
    ["Low", "Medium", "High"]
)

parental_education = st.selectbox(
    "Parental Education Level",
    ["High School", "College", "Postgraduate"]
)

extracurricular = st.selectbox(
    "Extracurricular Activities",
    ["No", "Yes"]
)

internet_access = st.selectbox(
    "Internet Access",
    ["No", "Yes"]
)

school_type = st.selectbox(
    "School Type",
    ["Private", "Public"]
)

peer_influence = st.selectbox(
    "Peer Influence",
    ["Negative", "Neutral", "Positive"]
)

learning_disabilities = st.selectbox(
    "Learning Disabilities",
    ["No", "Yes"]
)

distance_from_home = st.selectbox(
    "Distance from Home",
    ["Far", "Moderate", "Near"]
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

if st.button("Predict"):
    data = pd.DataFrame([{
        'Hours_Studied': hours_studied,
        'Attendance': attendance,
        'Parental_Involvement': parental_involvement,
        'Access_to_Resources': access_to_resources,
        'Sleep_Hours': sleep_hours,
        'Previous_Scores': previous_scores,
        'Motivation_Level': motivation_level,
        'Tutoring_Sessions': tutoring_sessions,
        'Family_Income': family_income,
        'Teacher_Quality': teacher_quality,
        'Physical_Activity': physical_activity,
        'Parental_Education_Level': parental_education,
        'Extracurricular_Activities': extracurricular,
        'Internet_Access': internet_access,
        'School_Type': school_type,
        'Peer_Influence': peer_influence,
        'Learning_Disabilities': learning_disabilities,
        'Distance_from_Home': distance_from_home,
        'Gender': gender
    }])

    level = {
        'Low': 0,
        'Medium': 1,
        'High': 2
    }

    ordinal_cols = [
        'Parental_Involvement',
        'Access_to_Resources',
        'Motivation_Level',
        'Family_Income',
        'Teacher_Quality'
    ]

    for col in ordinal_cols:
        data[col] = data[col].map(level)

    edu_mapping = {
        'High School': 0,
        'College': 1,
        'Postgraduate': 2
    }

    data['Parental_Education_Level'] = data['Parental_Education_Level'].map(
        edu_mapping
    )

    data = pd.get_dummies(data, dtype=int, drop_first=True)

    data = data.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(data)

    st.success(f"Predicted Exam Score: {prediction[0]:.2f}")