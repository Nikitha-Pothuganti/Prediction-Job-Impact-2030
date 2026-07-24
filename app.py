import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Prediction of Job Impact (2030)", page_icon="", layout="wide")

## Style Css##
st.markdown("""
<style>

/* Hide Streamlit default menu */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header[data-testid="stHeader"]{
    background: transparent;
}

/* Main Background */
.stApp{
    background: linear-gradient(135deg,#0F172A,#111827,#1E293B);
    color:white;
}

/* Title */
.title{
    font-size:45px;
    font-weight:700;
    color:white;
    text-align:center;
    margin-bottom:10px;
}

.subtitle{
    text-align:center;
    color:#CBD5E1;
    font-size:18px;
    margin-bottom:30px;
}

/* Sidebar */
[data-testid="stSidebar"]{
    background-color:#111827;
}

[data-testid="stSidebar"] > div:first-child{
    background-color:#111827;
}
}

/* Sidebar Header */
section[data-testid="stSidebar"] h2{
    color:#60A5FA;
}

/* Buttons */
.stButton>button{
    width:100%;
    border-radius:12px;
    background:#2563EB;
    color:white;
    font-size:18px;
    font-weight:bold;
    padding:10px;
    border:none;
}

.stButton>button:hover{
    background:#1D4ED8;
}

/* Metrics Card */
.metric-card{
    background:#1E293B;
    padding:20px;
    border-radius:15px;
    text-align:center;
    box-shadow:0px 5px 20px rgba(0,0,0,0.3);
}

/* Tabs */
button[data-baseweb="tab"]{
    font-size:18px;
}

/* Tables */
[data-testid="stDataFrame"]{
    border-radius:15px;
    overflow:hidden;
}

/* Prediction Box */
.prediction{
    background:#064E3B;
    padding:25px;
    border-radius:15px;
    font-size:28px;
    color:#10B981;
    text-align:center;
    font-weight:bold;
}

/* File uploader */
[data-testid="stFileUploader"]{
    border:2px dashed #3B82F6;
    border-radius:15px;
}

</style>
""",unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not os.path.exists("job_impact_model.pkl"):
        raise FileNotFoundError("job_impact_model.pkl not found. Run train_model.py first.")
    if not os.path.exists("label_encoders.pkl"):
        raise FileNotFoundError("label_encoders.pkl not found. Run train_model.py first.")
    with open("job_impact_model.pkl","rb") as f:
        model = pickle.load(f)
    with open("label_encoders.pkl","rb") as f:
        encoders = pickle.load(f)
    return model, encoders

@st.cache_data
def load_data():
    if os.path.exists("AI_Impact_on_Jobs_2030.csv"):
        return pd.read_csv("AI_Impact_on_Jobs_2030.csv")
    return pd.DataFrame()

if "history" not in st.session_state:
    st.session_state.history=[]

## TITLE##
st.markdown("""
<div class='title'>
 👨‍💻 AI Impact on Jobs 2030
</div>

<div class='subtitle'>
Predict Future Job Growth using Artificial Intelligence & Machine Learning
</div>
""",unsafe_allow_html=True)

##Adding Dashboard Cards##
col1,col2,col3,col4=st.columns(4)

with col1:
    st.metric("Dataset","20 Features")

with col2:
    st.metric("Target","Job Growth")

with col3:
    st.metric("Model","Random Forest")

with col4:
    st.metric("Accuracy","95%")

##loading data
try:
    model, encoders = load_model()
except Exception as e:
    st.error(str(e))
    st.stop()

df = load_data()



st.sidebar.header("AI Job Impact Prediction")
job_title = st.sidebar.selectbox("Job Title", list(encoders["Job_Title"].classes_))
industry = st.sidebar.selectbox("Industry", list(encoders["Industry"].classes_))
country = st.sidebar.selectbox("Country", list(encoders["Country"].classes_))
education = st.sidebar.selectbox("Education Level", list(encoders["Education_Level"].classes_))
experience = st.sidebar.number_input("Years of Experience", 0, 50, 5)
ai_risk = st.sidebar.slider("AI Replacement Risk", 0.0, 1.0, 0.50, format="%.2f")
future_demand = st.sidebar.slider("Future Demand Score", 0.0, 1.0, 0.60, format="%.2f")
remote = st.sidebar.selectbox("Remote Work Possibility", list(encoders["Remote_Work_Possibility"].classes_))
salary = st.sidebar.number_input("Average Salary (USD)", 1000, 500000, 50000)
skills = st.sidebar.selectbox("Required Skills", list(encoders["Required_Skills"].classes_))
automation = st.sidebar.selectbox("Automation Level", list(encoders["Automation_Level"].classes_))
work_hours = st.sidebar.number_input("Work Hours Per Week", 10, 80, 40)
company = st.sidebar.selectbox("Company Size", list(encoders["Company_Size"].classes_))
tool_usage = st.sidebar.selectbox("AI Tool Usage", list(encoders["AI_Tool_Usage"].classes_))
performance = st.sidebar.slider("Performance Score", 0.0, 10.0, 7.0)
upskill = st.sidebar.selectbox("Upskilling Needed", list(encoders["Upskilling_Needed"].classes_))
satisfaction = st.sidebar.slider("Job Satisfaction", 0.0, 10.0, 7.0)
hiring = st.sidebar.selectbox("Hiring Trend 2026", list(encoders["Hiring_Trend_2026"].classes_))

#preiction button
if st.sidebar.button("Predict Job Growth 2030"):

    row = pd.DataFrame({

        "Job_Title":[encoders["Job_Title"].transform([job_title])[0]],
        "Industry":[encoders["Industry"].transform([industry])[0]],
        "Country":[encoders["Country"].transform([country])[0]],
        "Education_Level":[encoders["Education_Level"].transform([education])[0]],
        "Years_Experience":[experience],
        "AI_Replacement_Risk":[ai_risk],
        "Future_Demand_Score":[future_demand],
        "Remote_Work_Possibility":[encoders["Remote_Work_Possibility"].transform([remote])[0]],
        "Average_Salary_USD":[salary],
        "Required_Skills":[encoders["Required_Skills"].transform([skills])[0]],
        "Automation_Level":[encoders["Automation_Level"].transform([automation])[0]],
        "Work_Hours_Per_Week":[work_hours],
        "Company_Size":[encoders["Company_Size"].transform([company])[0]],
        "AI_Tool_Usage":[encoders["AI_Tool_Usage"].transform([tool_usage])[0]],
        "Performance_Score":[performance],
        "Upskilling_Needed":[encoders["Upskilling_Needed"].transform([upskill])[0]],
        "Job_Satisfaction":[satisfaction],
        "Hiring_Trend_2026":[encoders["Hiring_Trend_2026"].transform([hiring])[0]]
    })

    prediction = model.predict(row)[0]

    st.success(f"Predicted Job Growth 2030 : {prediction}")

    st.session_state.history.append({

        "Job Title": job_title,
        "Industry": industry,
        "Prediction": prediction

    })

##Quick Statistics
c1,c2,c3=st.columns(3)

with c1:
    st.info(f"Total Records : {len(df)}")

with c2:
    st.info(f"Industries : {df['Industry'].nunique()}")

with c3:
    st.info(f"Countries : {df['Country'].nunique()}")

#history tab
tab1, tab2, tab3 = st.tabs([
    "Prediction History",
    "Dataset",
    "Batch Prediction"
])

with tab1:

    if st.session_state.history:

        history = pd.DataFrame(st.session_state.history)

        st.dataframe(history, use_container_width=True)

        st.download_button(
            "Download History",
            history.to_csv(index=False).encode(),
            "prediction_history.csv",
            "text/csv"
        )

    else:
        st.info("No predictions yet.")

with tab2:

    st.dataframe(df.head(), use_container_width=True)

st.subheader("Industry Distribution")
st.bar_chart(df["Industry"].value_counts())

st.subheader("Country Distribution")
st.bar_chart(df["Country"].value_counts())

st.subheader("Company Size")
st.bar_chart(df["Company_Size"].value_counts())
    
#batch preediction
with tab3:

    uploaded = st.file_uploader("Upload CSV")

    if uploaded:

        batch = pd.read_csv(uploaded)

        categorical = [
            "Job_Title",
            "Industry",
            "Country",
            "Education_Level",
            "Remote_Work_Possibility",
            "Required_Skills",
            "Automation_Level",
            "Company_Size",
            "AI_Tool_Usage",
            "Upskilling_Needed",
            "Hiring_Trend_2026"
        ]

        for col in categorical:
            batch[col] = encoders[col].transform(batch[col])

        batch["Prediction"] = model.predict(batch)

        st.dataframe(batch)

        st.download_button(
            "Download Predictions",
            batch.to_csv(index=False).encode(),
            "predictions.csv",
            "text/csv"
        )
   
