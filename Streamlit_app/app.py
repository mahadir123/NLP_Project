import streamlit as st
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")

# Define custom labels
labels = {
    'LABEL_0': "Non-anti-LGBT+ content",
    'LABEL_1': "Homophobia",
    'LABEL_2': "Transphobia"
}

# Load the models from Hugging Face
@st.cache_resource
def load_model(language):
    if language == "Malayalam":
        model_name = "Kishorekumarponnusamy/hom_tran_mal_mbert"  # Replace with your Malayalam model
        nlp_pipeline = pipeline("text-classification", model=model_name)
    elif language == "Tamil":
        model_name = "Kishorekumarponnusamy/hom_tran_tam_mbert"  # Replace with your Tamil model
        nlp_pipeline = pipeline("text-classification", model=model_name)
    return nlp_pipeline

# Create the Streamlit app
st.title("Homophobia and Transphobia Identification for Social Media Texts")

st.write("Select a language and enter some text to analyze the sentiment:")

# Add a selection box for language choice
language = st.selectbox("Choose a language", ("Malayalam", "Tamil"))

user_input = st.text_area("Text Input", "Enter text here...")

if st.button("Analyze"):
    nlp_pipeline = load_model(language)
    result = nlp_pipeline(user_input)
    label = result[0]['label']
    # confidence = result[0]['score']
    
    # Map the model output label to the custom label
    custom_label = labels.get(label, "Unknown")
    
    st.write("Sentiment:", custom_label)
    # st.write("Confidence Score:", confidence)
