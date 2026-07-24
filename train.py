import pandas as pd
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
## Loading dataset 
print("Loading Dataset___")
df=pd.read_csv("AI_Impact_on_Jobs_2030.csv")
print(df.head())

#checking missing values
print("\n Missing values")
print(df.isnull().sum())
df = df.dropna()

#encoding categorical colums
Categorical_columns =[
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
label_encoders ={}
for  column in Categorical_columns:
    encoder = LabelEncoder()
    df[column]= encoder.fit_transform(df[column])
    label_encoders[column] = encoder
    print("\nCategorical Encoding Completed")
  
print("\n Categorical encoding Completed")

x = df.drop(columns=["Employee_ID", "Job_Growth_2030"])
y = df["Job_Growth_2030"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print("\n Training Samples:",len(x_train))
print("Testing Samples:",len(x_test))

#train random forest model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42

)
print("\nTraining Model...")
model.fit(x_train,y_train)
print("Training Completed")

#prediction
y_pred = model.predict(x_test)

#evaluation
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test,y_pred)
print("\n=======================")
print("Model Performance")
print("=========================")

#Save model using pickle
with open("job_impact_model.pkl","wb") as file:
    pickle.dump(model,file)
print("\nModel Saved : job_impact_model.pkl")

#save label encoder
with open("label_encoders.pkl","wb") as file:
     pickle.dump(label_encoders,file)
print("\nLabel Encoders Saved : label_encoders.pkl")

#feature Importance
importance = pd.DataFrame({
    "Feature":x.columns,
    "Importance":model.feature_importances_
})
importance = importance.sort_values(by="Importance",ascending=False)
print("\n Feature Importance")
print(importance)
print("\n=================================")
print("Training Completed Successfully")
print("===================================")
