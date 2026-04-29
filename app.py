from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

        ####Load Model
model = joblib.load("model.pkl")

        ### FastAPI
app = FastAPI()

        ### Input schema
class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int
    ### Home route
@app.get("/")
def home():
    return{"message": "Diabetes Prediction API Running"}

        ###Prediction route
@app.post("/Predict")
def predict(data: PatientData):
    input_data = np.array([[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]])
    prediction = model.predict(input_data)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    return{"prediction": result}