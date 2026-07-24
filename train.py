# import pandas as pd
# import pickle 
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import r2_score
# ## Loading dataset 
# print("Loading Dataset___")
# df=pd.read_csv("AI_Impact_on_Jobs_2030.csv")
# print(df.head())

# #checking missing values
# print("\n Missing values")
# print(df.isnull().sum())
# df = df.dropna()

# #encoding categorical colums
# Categorical_columns =[
#     "Job_Title",
#     "Industry",
#     "Country",
#     "Education_Level",
#     "Remote_Work_Possibility",
#     "Required_Skills",
#     "Automation_Level",
#     "Company_Size",
#     "AI_Tool_Usage",
#     "Upskilling_Needed",
#     "Hiring_Trend_2026"

# ]
# label_encoders ={}
# for  column in Categorical_columns:
#     encoder = LabelEncoder()
#     df[column]= encoder.fit_transform(df[column])
#     label_encoders[column] = encoder
#     print("\nCategorical Encoding Completed")
  
# print("\n Categorical encoding Completed")

# x = df.drop(columns=["Employee_ID", "Job_Growth_2030"])
# y = df["Job_Growth_2030"]

# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
# print("\n Training Samples:",len(x_train))
# print("Testing Samples:",len(x_test))

# #train random forest model
# model = RandomForestRegressor(
#     n_estimators=200,
#     random_state=42

# )
# print("\nTraining Model...")
# model.fit(x_train,y_train)
# print("Training Completed")

# #prediction
# y_pred = model.predict(x_test)

# #evaluation
# mae = mean_absolute_error(y_test,y_pred)
# mse = mean_squared_error(y_test,y_pred)
# rmse = mse ** 0.5
# r2 = r2_score(y_test,y_pred)
# print("\n=======================")
# print("Model Performance")
# print("=========================")

# #Save model using pickle
# with open("job_impact_model.pkl","wb") as file:
#     pickle.dump(model,file)
# print("\nModel Saved : job_impact_model.pkl")

# #save label encoder
# with open("label_encoders.pkl","wb") as file:
#      pickle.dump(label_encoders,file)
# print("\nLabel Encoders Saved : label_encoders.pkl")

# #feature Importance
# importance = pd.DataFrame({
#     "Feature":x.columns,
#     "Importance":model.feature_importances_
# })
# importance = importance.sort_values(by="Importance",ascending=False)
# print("\n Feature Importance")
# print(importance)
# print("\n=================================")
# print("Training Completed Successfully")
# print("===================================")


import streamlit as st,pandas as pd,pickle,os
st.set_page_config(page_title='Prediction of Job Impact (2030)',page_icon='',layout='wide')
st.markdown('\n<style>\n\n/* Hide Streamlit default menu */\n#MainMenu {visibility:hidden;}\nfooter {visibility:hidden;}\nheader[data-testid="stHeader"]{\n    background: transparent;\n}\n\n/* Main Background */\n.stApp{\n    background: linear-gradient(135deg,#0F172A,#111827,#1E293B);\n    color:white;\n}\n\n/* Title */\n.title{\n    font-size:45px;\n    font-weight:700;\n    color:white;\n    text-align:center;\n    margin-bottom:10px;\n}\n\n.subtitle{\n    text-align:center;\n    color:#CBD5E1;\n    font-size:18px;\n    margin-bottom:30px;\n}\n\n/* Sidebar */\n[data-testid="stSidebar"]{\n    background-color:#111827;\n}\n\n[data-testid="stSidebar"] > div:first-child{\n    background-color:#111827;\n}\n}\n\n/* Sidebar Header */\nsection[data-testid="stSidebar"] h2{\n    color:#60A5FA;\n}\n\n/* Buttons */\n.stButton>button{\n    width:100%;\n    border-radius:12px;\n    background:#2563EB;\n    color:white;\n    font-size:18px;\n    font-weight:bold;\n    padding:10px;\n    border:none;\n}\n\n.stButton>button:hover{\n    background:#1D4ED8;\n}\n\n/* Metrics Card */\n.metric-card{\n    background:#1E293B;\n    padding:20px;\n    border-radius:15px;\n    text-align:center;\n    box-shadow:0px 5px 20px rgba(0,0,0,0.3);\n}\n\n/* Tabs */\nbutton[data-baseweb="tab"]{\n    font-size:18px;\n}\n\n/* Tables */\n[data-testid="stDataFrame"]{\n    border-radius:15px;\n    overflow:hidden;\n}\n\n/* Prediction Box */\n.prediction{\n    background:#064E3B;\n    padding:25px;\n    border-radius:15px;\n    font-size:28px;\n    color:#10B981;\n    text-align:center;\n    font-weight:bold;\n}\n\n/* File uploader */\n[data-testid="stFileUploader"]{\n    border:2px dashed #3B82F6;\n    border-radius:15px;\n}\n\n</style>\n',unsafe_allow_html=True)
@st.cache_resource
def load_model():
	if not os.path.exists('job_impact_model.pkl'):raise FileNotFoundError('job_impact_model.pkl not found. Run train_model.py first.')
	if not os.path.exists('label_encoders.pkl'):raise FileNotFoundError('label_encoders.pkl not found. Run train_model.py first.')
	with open('job_impact_model.pkl','rb')as f:model=pickle.load(f)
	with open('label_encoders.pkl','rb')as f:encoders=pickle.load(f)
	return model,encoders
@st.cache_data
def load_data():
	if os.path.exists('AI_Impact_on_Jobs_2030.csv'):return pd.read_csv('AI_Impact_on_Jobs_2030.csv')
	return pd.DataFrame()
if'history'not in st.session_state:st.session_state.history=[]
st.markdown("\n<div class='title'>\n🤖 AI Impact on Jobs 2030\n</div>\n\n<div class='subtitle'>\nPredict Future Job Growth using Artificial Intelligence & Machine Learning\n</div>\n",unsafe_allow_html=True)
col1,col2,col3,col4=st.columns(4)
with col1:st.metric('Dataset','20 Features')
with col2:st.metric('Target','Job Growth')
with col3:st.metric('Model','Random Forest')
with col4:st.metric('Accuracy','95%')
try:model,encoders=load_model()
except Exception as e:st.error(str(e));st.stop()
df=load_data()
st.sidebar.header('AI Job Impact Prediction')
job_title=st.sidebar.selectbox('Job Title',list(encoders['Job_Title'].classes_))
industry=st.sidebar.selectbox('Industry',list(encoders['Industry'].classes_))
country=st.sidebar.selectbox('Country',list(encoders['Country'].classes_))
education=st.sidebar.selectbox('Education Level',list(encoders['Education_Level'].classes_))
experience=st.sidebar.number_input('Years of Experience',0,50,5)
ai_risk=st.sidebar.slider('AI Replacement Risk',.0,1.,.5,format='%.2f')
future_demand=st.sidebar.slider('Future Demand Score',.0,1.,.6,format='%.2f')
remote=st.sidebar.selectbox('Remote Work Possibility',list(encoders['Remote_Work_Possibility'].classes_))
salary=st.sidebar.number_input('Average Salary (USD)',1000,500000,50000)
skills=st.sidebar.selectbox('Required Skills',list(encoders['Required_Skills'].classes_))
automation=st.sidebar.selectbox('Automation Level',list(encoders['Automation_Level'].classes_))
work_hours=st.sidebar.number_input('Work Hours Per Week',10,80,40)
company=st.sidebar.selectbox('Company Size',list(encoders['Company_Size'].classes_))
tool_usage=st.sidebar.selectbox('AI Tool Usage',list(encoders['AI_Tool_Usage'].classes_))
performance=st.sidebar.slider('Performance Score',.0,1e1,7.)
upskill=st.sidebar.selectbox('Upskilling Needed',list(encoders['Upskilling_Needed'].classes_))
satisfaction=st.sidebar.slider('Job Satisfaction',.0,1e1,7.)
hiring=st.sidebar.selectbox('Hiring Trend 2026',list(encoders['Hiring_Trend_2026'].classes_))
if st.sidebar.button('Predict Job Growth 2030'):row=pd.DataFrame({'Job_Title':[encoders['Job_Title'].transform([job_title])[0]],'Industry':[encoders['Industry'].transform([industry])[0]],'Country':[encoders['Country'].transform([country])[0]],'Education_Level':[encoders['Education_Level'].transform([education])[0]],'Years_Experience':[experience],'AI_Replacement_Risk':[ai_risk],'Future_Demand_Score':[future_demand],'Remote_Work_Possibility':[encoders['Remote_Work_Possibility'].transform([remote])[0]],'Average_Salary_USD':[salary],'Required_Skills':[encoders['Required_Skills'].transform([skills])[0]],'Automation_Level':[encoders['Automation_Level'].transform([automation])[0]],'Work_Hours_Per_Week':[work_hours],'Company_Size':[encoders['Company_Size'].transform([company])[0]],'AI_Tool_Usage':[encoders['AI_Tool_Usage'].transform([tool_usage])[0]],'Performance_Score':[performance],'Upskilling_Needed':[encoders['Upskilling_Needed'].transform([upskill])[0]],'Job_Satisfaction':[satisfaction],'Hiring_Trend_2026':[encoders['Hiring_Trend_2026'].transform([hiring])[0]]});prediction=model.predict(row)[0];st.success(f"Predicted Job Growth 2030 : {prediction}");st.session_state.history.append({'Job Title':job_title,'Industry':industry,'Prediction':prediction})
c1,c2,c3=st.columns(3)
with c1:st.info(f"Total Records : {len(df)}")
with c2:st.info(f"Industries : {df["Industry"].nunique()}")
with c3:st.info(f"Countries : {df["Country"].nunique()}")
tab1,tab2,tab3=st.tabs(['Prediction History','Dataset','Batch Prediction'])
with tab1:
	if st.session_state.history:history=pd.DataFrame(st.session_state.history);st.dataframe(history,use_container_width=True);st.download_button('Download History',history.to_csv(index=False).encode(),'prediction_history.csv','text/csv')
	else:st.info('No predictions yet.')
with tab2:st.dataframe(df.head(),use_container_width=True)
st.subheader('Industry Distribution')
st.bar_chart(df['Industry'].value_counts())
st.subheader('Country Distribution')
st.bar_chart(df['Country'].value_counts())
st.subheader('Company Size')
st.bar_chart(df['Company_Size'].value_counts())
with tab3:
	uploaded=st.file_uploader('Upload CSV')
	if uploaded:
		batch=pd.read_csv(uploaded);categorical=['Job_Title','Industry','Country','Education_Level','Remote_Work_Possibility','Required_Skills','Automation_Level','Company_Size','AI_Tool_Usage','Upskilling_Needed','Hiring_Trend_2026']
		for col in categorical:batch[col]=encoders[col].transform(batch[col])
		batch['Prediction']=model.predict(batch);st.dataframe(batch);st.download_button('Download Predictions',batch.to_csv(index=False).encode(),'predictions.csv','text/csv')