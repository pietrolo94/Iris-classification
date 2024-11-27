import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Caricare il dataset Iris da un URL
dataset_path = 'https://frenzy86.s3.eu-west-2.amazonaws.com/fav/iris.data'
df = pd.read_csv(dataset_path, header=None)
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
df['class'] = df['class'].astype('category')  # Assicurarsi che 'class' sia categoriale

# Caricare il modello
model = joblib.load("iris_model.pkl")

# Streamlit App
st.title("Classificazione delle Iris 🌸")

# Navigazione tra pagine
menu = st.sidebar.selectbox("Navigazione", 
                             ["Home", 
                              "Predizione", 
                              "Pairplot", 
                              "Istogrammi", 
                              "Boxplot", 
                              "Heatmap delle Correlazioni", 
                              "Violin Plot"])

# Pagina Home
if menu == "Home":
    st.header("EDA Iris dataset and Logistic regression ML!")
    st.write("Questa applicazione consente di:")
    st.markdown("""
    - Predire la specie di Iris basandosi su caratteristiche misurate.
    - Esplorare il dataset tramite grafici:
        - Pairplot
        - Istogrammi
        - Boxplot
        - Heatmap delle correlazioni
        - Violin Plot
    """)

# Pagina Predizione
elif menu == "Predizione":
    st.header("Predizione della specie di Iris")
    
    # Input utente
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)
    
    if st.button("Predici"):
        # Predizione
        sample = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
        prediction = model.predict(sample)  # Restituisce un array
        st.success(f"La specie predetta è: **{prediction[0]}** 🌼")

# Pagina Pairplot
elif menu == "Pairplot":
    st.header("Pairplot delle Caratteristiche")
    fig = sns.pairplot(df, hue='class', height=3, aspect=1)
    st.pyplot(fig)

# Pagina Istogrammi
elif menu == "Istogrammi":
    st.header("Istogrammi delle Caratteristiche")
    features = ['sepal length', 'sepal width', 'petal length', 'petal width']
    for feature in features:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=df, x=feature, hue='class', kde=True, ax=ax, palette='pastel', multiple='stack')
        ax.set_title(f"Distribuzione di {feature}")
        st.pyplot(fig)

# Pagina Boxplot
elif menu == "Boxplot":
    st.header("Boxplot delle Caratteristiche")
    features = ['sepal length', 'sepal width', 'petal length', 'petal width']
    for feature in features:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x='class', y=feature, palette='pastel', ax=ax)
        ax.set_title(f"Boxplot di {feature} per Classe")
        st.pyplot(fig)

# Pagina Heatmap delle Correlazioni
elif menu == "Heatmap delle Correlazioni":
    st.header("Heatmap delle Correlazioni")
    correlation_matrix = df.iloc[:, :-1].corr()  # Escludere la colonna 'class'
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Matrice delle Correlazioni")
    st.pyplot(fig)

# Pagina Violin Plot
elif menu == "Violin Plot":
    st.header("Violin Plot delle Caratteristiche")
    features = ['sepal length', 'sepal width', 'petal length', 'petal width']
    for feature in features:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.violinplot(data=df, x='class', y=feature, palette='pastel', ax=ax)
        ax.set_title(f"Violin Plot di {feature} per Classe")
        st.pyplot(fig)
