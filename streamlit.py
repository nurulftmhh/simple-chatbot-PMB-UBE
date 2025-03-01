import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import joblib
import string
import html
from tensorflow.keras.models import load_model

# Cache the model loading
@st.cache_resource
def load_resources():
    try:
        # Load the LSTM model (using the correct file format)
        model = load_model("model_pmb_lstm.h5")
        
        # Load the label encoder
        with open("label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
            
        # Load the text vectorization layer
        with open("text_vectorization.pkl", "rb") as f:
            text_vectorizer = pickle.load(f)
            
        # Load the training data for response mapping
        train_df = pd.read_csv('Data Train.csv')
        intent_response_mapping = dict(zip(train_df['Intent'], train_df['Respon']))
        
        # Load the slangwords dictionary
        slangwords_dict = load_slangwords('Slangword-indonesian.csv')
        
        # Add manual slangwords that were used during training
        manual_slang_dict = {
            "mhs": "mahasiswa",
            "maba": "mahasiswa baru",
            "pkkmb": "pengenalan kehidupan kampus bagi mahasiswa baru"
        }
        slangwords_dict.update(manual_slang_dict)
        
        return model, label_encoder, text_vectorizer, intent_response_mapping, slangwords_dict
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        return None, None, None, None, None

# Load slangwords from CSV file
def load_slangwords(file_path):
    try:
        slangwords = {}
        import csv
        with open(file_path, mode='r', encoding='utf-8', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2:  # Ensure there are at least two columns
                    slang = row[0].strip()  # Slang word
                    correct = row[1].strip()  # Replacement word
                    slangwords[slang] = correct
        return slangwords
    except Exception as e:
        st.warning(f"Could not load slangwords file: {str(e)}. Proceeding with empty dictionary.")
        return {}

# Fix slangwords in text (to match training preprocessing)
def fix_slangwords(text, slangwords_dict):
    words = text.split()
    fixed_words = [slangwords_dict[word.lower()] if word.lower() in slangwords_dict else word for word in words]
    return ' '.join(fixed_words)

# Text preprocessing function (to match training preprocessing)
def preprocess_text(text, slangwords_dict):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = fix_slangwords(text, slangwords_dict)  # Fix slangwords
    return text

# Prediction function
def predict_intent_and_response(user_input, model, label_encoder, text_vectorizer, intent_response_mapping, slangwords_dict):
    try:
        # Preprocess the input (using same preprocessing as training)
        processed_input = preprocess_text(user_input, slangwords_dict)
        
        # Vectorize the text
        input_seq = text_vectorizer([processed_input])
        
        # Make prediction
        prediction = model.predict(input_seq)
        predicted_class_index = np.argmax(prediction)
        
        # Get the predicted intent
        predicted_intent = label_encoder.inverse_transform([predicted_class_index])[0]
        
        # Get the corresponding response
        response = intent_response_mapping.get(predicted_intent, 
            "Maaf, saya tidak dapat memahami pertanyaan Anda. Mohon ajukan pertanyaan dengan cara yang berbeda.")
        
        return predicted_intent, response, prediction[0][predicted_class_index]
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, "Terjadi kesalahan dalam memproses pertanyaan Anda. Silakan coba lagi.", 0.0

def local_css():
    st.markdown("""
    <style>
    .chat-container {
        padding: 20px;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 10px;
        display: flex;
        flex-direction: column;
        color: black;
    }
    
    .message-content {
        display: flex;
        align-items: flex-start;
        gap: 10px;
    }
    
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
    }
    
    .bot-name {
        font-weight: bold;
        margin-bottom: 5px;
        color: #2E4053;
    }
    
    .message-bubble {
        padding: 12px 16px;
        border-radius: 15px;
        max-width: 80%;
        line-height: 1.4;
    }
    
    .user-message {
        background-color: #E9ECEF;
        margin-left: auto;
        margin-right: 2%;
    }
    
    .bot-message {
        background-color: #007AFF;
        color: white;
        margin-right: auto;
        margin-left: 2%;
    }
    
    .chat-input {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 20px;
        background-color: white;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    }
    
    .header-container {
        padding: 20px;
        text-align: center;
        border-bottom: 1px solid #E9ECEF;
        margin-bottom: 30px;
    }
    
    .stButton button {
        background-color: blue;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: white;
    }
    
    .stTextInput input {
        padding: 0.8rem 1rem;
        border-radius: 5px
        border: 1px solid #E9ECEF;
        font-size: 16px;
    }

    [data-testid="InputInstructions"], 
    div.st-emotion-cache-ysi923,
    div.st-emotion-cache-1qg05tj,
    div.st-emotion-cache-16j3ejq,
    div.st-emotion-cache-e1nzilvr,
    .st-emotion-cache-16idsys p,
    .stTextInput div small {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }   
    
    .st-emotion-cache-acwcvw {
        display: none !important;
    }  
    
    .main-content {
        margin-bottom: 100px;
        padding: 0 20px;
    }
    </style>
    """, unsafe_allow_html=True)

def display_message(message, is_user=True):
    bot_avatar = "https://miro.medium.com/v2/resize:fit:828/format:webp/1*I9KrlBSL9cZmpQU3T2nq-A.jpeg"
    
    if is_user:
        st.markdown(f"""
        <div class="chat-message">
            <div class="message-bubble user-message">
                {message}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message">
            <div class="message-content">
                <img src="{bot_avatar}" class="avatar" alt="EduBot">
                <div style="flex-grow: 1;">
                    <div class="bot-name" style="color: white;">EduBot</div>
                    <div class="message-bubble bot-message">
                        {message}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="EduBot - Asisten Pendaftaran Mahasiswa",
        page_icon="🎓",
        layout="wide"
    )
    
    local_css()
    
    # Initialize session state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
        # Add initial greeting message
        st.session_state.conversation.append({
            'text': "Halo! Saya adalah asisten pendaftaran mahasiswa baru UBE. Saya siap membantu Anda dengan informasi terkait program studi, jadwal, syarat, cara pendaftaran, biaya kuliah, persiapan, proses, pengumuman seleksi, cara daftar ulang, dan informasi pkkmb bagi calon Mahasiswa Baru UBE. Apa yang ingin Anda tanyakan?",
            'is_user': False
        })
    
    # Load resources
    model, label_encoder, text_vectorizer, intent_response_mapping, slangwords_dict = load_resources()
    
    if not all([model, label_encoder, text_vectorizer, intent_response_mapping]):
        st.error("Gagal memuat sumber daya yang diperlukan. Silakan refresh halaman.")
        return
    
    # Header
    st.markdown("""
    <div class="header-container">
        <img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*I9KrlBSL9cZmpQU3T2nq-A.jpeg" 
             style="width: 80px; height: 80px; border-radius: 50%; margin-bottom: 10px;">
        <h1 style="margin: 10px 0; font-size: 28px;">EduBot</h1>
        <p style="color: white; font-size: 16px;">Asisten Pendaftaran Mahasiswa Baru</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # Display conversation history
    for message in st.session_state.conversation:
        display_message(message['text'], message['is_user'])
    
    # Chat input
    with st.container():
        # Add a JavaScript snippet to hide the input instructions 
        st.markdown("""
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Hide all elements that might contain instructions
            const smallElements = document.querySelectorAll('small');
            smallElements.forEach(el => {
                el.style.display = 'none';
            });
            
            // Find and hide elements with specific data-testid
            const inputInstructions = document.querySelectorAll('[data-testid="InputInstructions"]');
            inputInstructions.forEach(el => {
                el.style.display = 'none';
            });
        });
        </script>
        """, unsafe_allow_html=True)
        
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input(
                "",
                placeholder="Ketik pertanyaan Anda di sini...",
                key="user_input",
                label_visibility="collapsed"  # This hides the label
            )
            
            col1, col2, col3 = st.columns([4, 1, 4])
            with col2:
                submit_button = st.form_submit_button("Kirim")
    
    # Handle user input
    if submit_button and user_input:
        # Add user message to conversation
        st.session_state.conversation.append({
            'text': user_input,
            'is_user': True
        })
        
        # Get bot response
        intent, response, confidence = predict_intent_and_response(
            user_input, model, label_encoder, text_vectorizer, intent_response_mapping, slangwords_dict
        )
        
        # Add bot response to conversation
        st.session_state.conversation.append({
            'text': response,
            'is_user': False
        })
        
        # Rerun to update the display
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
