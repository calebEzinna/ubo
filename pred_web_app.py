import warnings
import numpy as np
import pickle
import pandas as pd
from fastapi import FastAPI, Form, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder
import uvicorn

# Suppress all warnings
warnings.filterwarnings('ignore')

# Load the saved model
model = pickle.load(open('trained_model.sav', 'rb'))

# Load the drugs database
drug_df = pd.read_excel("Drugs_Database.xlsx")

# FastAPI app
app = FastAPI()

# Class to define the request body for prediction
class PredictionRequest(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Function to predict diabetes type
def predict_diabetes(input_data):
    input_data_as_nparray = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_data_as_nparray)
    
    label_encoder = LabelEncoder()
    diabetes_types = ['Gestational', 'No Diabetes', 'Type 1', 'Type 2']
    label_encoder.fit(diabetes_types)
    predicted_type = label_encoder.inverse_transform(prediction)[0]
    
    return predicted_type

# Function to get drugs for a given diabetes type
def get_drugs(diabetes_type):
    return drug_df[drug_df['Type of Diabetes'] == diabetes_type]['Drugs Name'].tolist()

@app.post('/predict')
def predict(request: PredictionRequest):
    input_data = [
        request.Pregnancies,
        request.Glucose,
        request.BloodPressure,
        request.SkinThickness,
        request.BMI,
        request.DiabetesPedigreeFunction,
        request.Age
    ]
    
    diagnosis = predict_diabetes(input_data)
    drugs = get_drugs(diagnosis)
    
    if diagnosis == 'No Diabetes':
        drugs = []

    return {
        'prediction': diagnosis,
        'drugs': drugs
    }

# Endpoint to render the form
@app.get('/')
def form():
    return {
        'message': 'Use POST /predict to get predictions and drug recommendations.'
    }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
