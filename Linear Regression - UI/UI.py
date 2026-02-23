import pickle

import numpy as np
import streamlit as st
import pickle
import numpy as np

st.title("House Price Prediction")

user_input = st.text_input("Enter the Area in Sqft")

model = pickle.load(open("siva.pkl","rb"))

if st.button("Predict Price"):
    if user_input:
        a = float(user_input)
        price = model.predict(np.array([[a]]))
        st.success("The predicted price is"+str(price))


