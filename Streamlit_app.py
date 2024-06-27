import streamlit as st
import xgboost as xgb
import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go

# Get the directory of the script
script_directory = os.path.dirname(__file__)

# Load the trained model
model = xgb.XGBRegressor()
model.load_model(os.path.join(script_directory, 'model.json'))

# Define the function for prediction
def predict(input_data):
    input_df = pd.DataFrame(input_data, index=[0])
    prediction = model.predict(input_df)
    return prediction[0]

# Streamlit app
st.title('Estimation of the UCS of cement-treated soils using XGBoost model')

# Step 1: Calculate η/Civ
st.header('Step 1: Calculate η/Civ')

# Display the equations
st.markdown(r"""
### Equations:
1. $\eta = 100 - 100 \left[ \frac{\rho_i}{1 + \frac{C}{100}} \left( \frac{1}{\gamma_{ss}} + \frac{C}{100 \cdot \gamma_{SC}} \right) \right]$
2. $Civ = \frac{100 \left[ \frac{\rho_i}{1 + \frac{C}{100}} \cdot \frac{C}{100} \right]}{\gamma_{SC}}$
""")

# Input fields for parameters required to calculate nCiv
ρi = st.number_input('Initial dry density of the soil (ρi) [g/cm³]', min_value=0.0, value=1.5)
γss = st.number_input('Specific gravity of the soil grains (γss)', min_value=0.0, value=2.65)
γSC = st.number_input('Specific gravity of the cement (γsc)', min_value=0.0, value=3.15)
C = st.number_input('Cement dosage (%) - Range [1-10]', min_value=0.0, max_value=10.0, value=6.0)

# Calculate n and Civ using the provided equations
η = 100 - 100 * ((ρi / (1 + (C / 100))) * (1 / γss + (C / 100) / γSC))
Civ = (100 * ((ρi / (1 + (C / 100))) * (C / 100))) / γSC

# Calculate nCiv
nCiv = η / Civ

# Display the calculated nCiv value
st.write(f'Calculated η/Civ value: {nCiv:.2f}')

# Step 2: Estimate UCS
st.header('Step 2: Estimate UCS')

# Create two columns for inputs
col1, col2 = st.columns(2)

# Input fields for continuous features in the first column
with col1:
    LL = st.number_input('Liquid Limit (%) - Range [0-54]', min_value=0.0, max_value=54.0, value=40.0)
    FC = st.number_input('Fine Contents (%) - Range [0-99]', min_value=0.0, max_value=99.0, value=30.0)
    ρnorm = st.number_input('Normalized dry Density - Range [0.82-1.06]', min_value=0.82, max_value=1.06, value=0.85)
    ωnorm = st.number_input('Normalized water Content - Range [0.32-1.82]', min_value=0.32, max_value=1.82, value=0.9)

# Input fields for continuous features in the second column
with col2:
    cem_type = st.selectbox('Select Cement Type', ['None', 'CEM-I', 'CEM-II', 'CEM-III'])

# Initialize CEM values
CEM_I = 0
CEM_II = 0
CEM_III = 0

# Set the selected CEM type
if cem_type == 'CEM_I':
    CEM_I = 1
elif cem_type == 'CEM_II':
    CEM_II = 1
elif cem_type == 'CEM_III':
    CEM_III = 1

# Prepare base input data without curing time
base_input_data = {
    'LL': LL,
    'FC': FC,
    'CEM_I': CEM_I,
    'CEM_II': CEM_II,
    'CEM_III': CEM_III,
    'ρnorm': ρnorm,
    'ωnorm': ωnorm,
    'C': C,
    'nCiv': nCiv,
}

# Define curing times for prediction
curing_times = [7, 28, 90]

# Make predictions when button is clicked
if st.button('Predict UCS'):
    if cem_type == 'None':
        st.write('Please select a cement type to make a prediction.')
    else:
        ucs_values = []
        for T in curing_times:
            input_data = base_input_data.copy()
            input_data['T'] = T
            prediction = predict(input_data)
            prediction = round(prediction, 1)  # Round the prediction to 1 decimal place
            ucs_values.append(prediction)
            st.write(f'The predicted UCS value at T={T} days is: {prediction:.1f} kPa')  # Format the output to 1 decimal place
        
        # Plotting the results
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=curing_times, 
            y=ucs_values, 
            mode='lines+markers', 
            name='',
            hovertemplate='T: %{x}<br>UCS: %{y:.1f} kPa'
        ))
        fig.update_layout(
            title='Predicted UCS vs Curing Time (T)',
            xaxis_title='Curing Time (T) [days]',
            yaxis_title='Predicted UCS [kPa]',
            template='plotly_dark'
        )
        st.plotly_chart(fig)
