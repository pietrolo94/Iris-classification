import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset from URL
dataset_path = 'https://frenzy86.s3.eu-west-2.amazonaws.com/fav/iris.data'
df = pd.read_csv(dataset_path, header=None)
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
df['class'] = df['class'].astype('category')  # Ensure 'class' is categorical

# Load the model
model = joblib.load("iris_model.pkl")

# Streamlit App
st.title("Iris Classification 🌸")

# Navigation between pages
menu = st.sidebar.selectbox("Navigation", 
                             ["Home", 
                              "Prediction", 
                              "Pairplot", 
                              "Histograms", 
                              "Boxplot", 
                              "Correlation Heatmap", 
                              "Violin Plot"])

# Home Page
if menu == "Home":
    st.header("EDA on Iris dataset and Logistic Regression ML!")
    st.write("This app allows you to:")
    st.markdown("""
    - Predict the species of Iris based on measured features.
    - Explore the dataset through visualizations:
        - Pairplot
        - Histograms
        - Boxplot
        - Correlation Heatmap
        - Violin Plot
    """)
    st.subheader("Iris Dataset")
    st.write(df)  # Display the DataFrame on the home page

# Prediction Page
elif menu == "Prediction":
    st.header("Predict the Iris Species")
    
    # User input
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)
    
    if st.button("Predict"):
        # Prediction
        sample = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
        prediction = model.predict(sample)  # Returns an array
        st.success(f"The predicted species is: **{prediction[0]}** 🌼")

# Pairplot Page
elif menu == "Pairplot":
    st.header("Pairplot of Features")
    fig = sns.pairplot(df, hue='class', height=3, aspect=1)
    st.pyplot(fig)

# Histograms Page
elif menu == "Histograms":
    st.header("Histograms of Features")
    features = ['sepal length', 'sepal width', 'petal length', 'petal width']
    for feature in features:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=df, x=feature, hue='class', kde=True, ax=ax, palette='pastel', multiple='stack')
        ax.set_title(f"Distribution of {feature}")
        st.pyplot(fig)

# Boxplot Page
elif menu == "Boxplot":
    st.header("Boxplot of Features")
    features = ['sepal length', 'sepal width', 'petal length', 'petal width']
    for feature in features:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x='class', y=feature, palette='pastel', ax=ax)
        ax.set_title(f"Boxplot of {feature} by Class")
        st.pyplot(fig)

# Correlation Heatmap Page
elif menu == "Correlation Heatmap":
    st.header("Correlation Heatmap")
    correlation_matrix = df.iloc[:, :-1].corr()  # Exclude the 'class' column
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)

# Violin Plot Page
elif menu == "Violin Plot":
    st.header("Violin Plot of Features")
    features = ['sepal length', 'sepal width', 'petal length', 'petal width']
    for feature in features:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.violinplot(data=df, x='class', y=feature, palette='pastel', ax=ax)
        ax.set_title(f"Violin Plot of {feature} by Class")
        st.pyplot(fig)

