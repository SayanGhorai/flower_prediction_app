import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)

# Set the title
st.markdown("<h1 style='text-align: center;'>🌸 Welcome to Flower Prediction app</h1>", unsafe_allow_html=True)

# Input sliders for flower features
sepal_length = st.slider("sepal length in cm", 4.0, 8.0, 5.1)
sepal_width = st.slider("sepal width in cm", 2.0, 4.5, 3.5)
petal_length = st.slider("petal length in cm", 1.0, 7.0, 1.4)
petal_width = st.slider("petal width in cm", 0.1, 2.5, 0.2)

# When button is clicked
if st.button("predict"):
    # Prepare input
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    # Predict flower type
    prediction = model.predict(input_data)[0]

    # Display result
    st.markdown("## Prediction is :")
    st.markdown(f"### **{prediction}**")

    # Show image based on prediction
    if prediction == "Iris-setosa":
        st.image("iris_setosa.jpg", caption="Iris-setosa", use_container_width=True)
    elif prediction == "Iris-versicolor":
        st.image("iris_versicolor.jpg", caption="Iris-versicolor", use_container_width=True)
    elif prediction == "Iris-virginica":
        st.image("iris_virginica.jpg", caption="Iris-virginica", use_container_width=True)

