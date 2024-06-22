import warnings
import json
import numpy as np
import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# Load the saved model
with open('random_forest_balanced_bagging_model.sav', 'rb') as file:
    loaded_model = pickle.load(file)

# Set the page configuration
st.set_page_config(page_title='Diabetes Prediction', page_icon=':thermometer:', layout='wide')

data = pd.read_excel("Drugs_Database.xlsx")
drug_df = pd.DataFrame(data)

# Create a function for prediction
def diabetes_prediction(input_data):
    # Changing the data into a NumPy array
    input_data_as_nparray = np.asarray(input_data, dtype=np.float64)

    # Reshaping the data since there is only one instance
    input_data_reshaped = input_data_as_nparray.reshape(1, -1)

    # Predicting the diabetes type
    prediction = loaded_model.predict(input_data_reshaped)

    # Mapping the prediction to the diabetes type
    label_encoder = LabelEncoder()
    diabetes_types = ['Gestational', 'No Diabetes', 'Type 1', 'Type 2']
    label_encoder.fit(diabetes_types)
    predicted_type = label_encoder.inverse_transform(prediction)[0]

    return predicted_type

# Function to get drugs for a given diabetes type
def get_drugs(diabetes_type):
    return drug_df[drug_df['Type of Diabetes'] == diabetes_type]['Drugs Name'].tolist()

# Function to load selected drugs from JSON file
def load_selected_drugs():
    try:
        with open('selections.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Function to save selected drugs to JSON file
def save_selected_drugs(selected_drugs):
    with open('selections.json', 'w') as f:
        json.dump(selected_drugs, f)

def diagnosis_page():
    if 'diagnosis' in st.session_state and 'selected_drugs' in st.session_state and 'recommendation' in st.session_state:
        st.title('Diagnosis')
        st.write(f'### Diabetes Type: {st.session_state.diagnosis}')
        st.write(f'### Prescription:')
        for drug in st.session_state.selected_drugs:
            st.write(f'- {drug}')
        st.write(f'### Recommendation: {st.session_state.recommendation}')
    else:
        st.write('Please go back to the main page and make a prediction first.')

def main():
    # Initialize session state for selected drugs and recommendations
    if 'selected_drugs' not in st.session_state:
        st.session_state.selected_drugs = load_selected_drugs()
    if 'recommendation' not in st.session_state:
        st.session_state.recommendation = ""
    if 'diagnosis' not in st.session_state:
        st.session_state.diagnosis = ""
    if 'drugs' not in st.session_state:
        st.session_state.drugs = []

    # Giving a title
    st.title('Diabetes Prediction and Drug Recommendation Web App')

    # Getting input from the user
    Pregnancies = st.text_input('No. of Pregnancies:')
    Glucose = st.text_input('Glucose level:')
    BloodPressure = st.text_input('Blood Pressure value:')
    SkinThickness = st.text_input('Skin thickness value:')
    Insulin = st.text_input('Insulin value:')
    BMI = st.text_input('BMI value:')
    DiabetesPedigreeFunction = st.text_input('Diabetes pedigree function value:')
    Age = st.text_input('Age:')

    # Code for prediction
    if st.button('Predict'):
        input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]

        # Ensure all input data are valid floats
        try:
            input_data = [float(i) for i in input_data]
            st.session_state.diagnosis = diabetes_prediction(input_data)
            st.session_state.drugs = get_drugs(st.session_state.diagnosis)

            st.success(f'Prediction: {st.session_state.diagnosis}')
            st.experimental_rerun()
        except ValueError:
            st.error("Please enter valid numeric values for all fields.")

    if st.session_state.diagnosis and st.session_state.diagnosis != 'No Diabetes':
        st.write(f'Recommended Drugs for {st.session_state.diagnosis}:')

        # Filter selected drugs to ensure they are in the available options
        available_drugs = st.session_state.drugs
        selected_drugs = [drug for drug in st.session_state.selected_drugs if drug in available_drugs]

        # Ensure unique keys for each multiselect
        selected_drugs = st.multiselect(
            'Select drugs',
            available_drugs,
            default=selected_drugs,
            key='drug_selector'
        )
        st.session_state.selected_drugs = selected_drugs
        save_selected_drugs(selected_drugs)

        # Text area for recommendation
        st.session_state.recommendation = st.text_area(
            'Enter your recommendation:',
            value=st.session_state.recommendation,
            key='recommendation_area'
        )

        if st.button('Submit'):
            st.experimental_rerun()

# Define the Streamlit app
if __name__ == '__main__':
    selection = st.sidebar.radio('Go to', ['Main', 'Diagnosis'])
    if selection == 'Main':
        main()
    elif selection == 'Diagnosis':
        diagnosis_page()
