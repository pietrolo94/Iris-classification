import streamlit as st
import joblib

@st.cache_resource
def load_model():
    model = joblib.load("NLPEs2.pkl")  
    return model

# Carica il modello
model = load_model()

# Interfaccia Streamlit
st.title("Sentiment Analysis App")
user_input = st.text_area("Inserisci un commento:")
if st.button("Predici Sentiment"):
    if user_input:
        prediction = model.predict([user_input])[0]  
        st.write(f"Il sentiment previsto è: **{prediction}**")
    else:
        st.write("Per favore, inserisci un commento.")


# (Opzionale) Footer o note sull'app
st.write("---")
st.write("Modello creato da Pietro Zoffoli.")