import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import joblib

# Display raw data in the app
st.title('Telecom Data Dashboard')

st.subheader("Top 10 Handsets Used by Customers")
image = Image.open('Top 10 Handset.png')
st.image(image)

st.subheader("Top 10 Handsets Used by Customers")
image = Image.open('Top 10 handset value counts.png')
st.image(image)

st.subheader("Top 3 most used application")
image = Image.open('Top 3 most used application.png')
st.image(image)

st.subheader("Top 10 handset value counts")
image = Image.open('Top 10 handset value counts.png')
st.image(image)

st.subheader("Top 10 Handset")
image = Image.open('Top 10 Handset.png')
st.image(image)

st.subheader("Top 3 Handset Manufacturers.png")
image = Image.open('Top 3 Handset Manufacturers.png.png')
st.image(image)

st.subheader("Top 3 Handset Manufacturers value counts")
image = Image.open('Top 3 Handset Manufacturers value counts.png')
st.image(image)

st.subheader("Total Data Usage by Application")
image = Image.open('Total Data Usage by Application.png')
st.image(image)

st.subheader("Heatmap")
image = Image.open('Heatmap.png')
st.image(image)

st.subheader("K-means clustering")
image = Image.open('K-means clustering.png')
st.image(image)

st.subheader("pairplot")
image = Image.open('pairplot.png')
st.image(image)

st.subheader("PCA-1,PCA-2")
image = Image.open('PCA-1,PCA-2.png')
st.image(image)

st.subheader("plot distribution for each cluster boxplot")
image = Image.open('plot distribution for each cluster boxplot.png')
st.image(image)

st.subheader("boxplot")
image = Image.open('boxplot.png')
st.image(image)

st.subheader("Top 5 Handset for Apple")
image = Image.open('Top 5 Handset for Apple.png')
st.image(image)

st.subheader("Top 5 Handset for sumsung")
image = Image.open('Top 5 Handset for sumsung.png')
st.image(image)

st.subheader("Top 5 Handset for Huawei")
image = Image.open('Top 5 Handset for Huawei.png')
st.image(image)


import streamlit as st
import numpy as np
import pickle

# Load the saved regression model
with open('regression_model.pkl', 'rb') as file:
    regression_model = pickle.load(file)

# Streamlit app title
st.title("Satisfaction Score Prediction")

# Create input form for the user to enter feature values
st.header("Enter Feature Values")

# Input fields for Avg TCP Retransmission Volume, Avg RTT, and Avg Throughput
tcp_retrans_vol = st.number_input('Avg TCP Retrans. Vol (Bytes)', min_value=0)
rtt = st.number_input('Avg RTT (ms)', min_value=0.0)
throughput = st.number_input('Avg Throughput (kbps)', min_value=0.0)

# Button for prediction
if st.button('Predict Satisfaction Score'):
    # Prepare the input as a 2D array for the model
    input_features = np.array([[tcp_retrans_vol, rtt, throughput]])
    
    # Make the prediction using the loaded model
    satisfaction_score = regression_model.predict(input_features)
    
    # Display the predicted satisfaction score
    st.write(f"Predicted Satisfaction Score: {satisfaction_score[0]}")





