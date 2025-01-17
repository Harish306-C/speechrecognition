import streamlit as st
import speech_recognition as sr
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import pandas as pd
import numpy as np

# Initialize components
sentiment_analyzer = pipeline("sentiment-analysis")
recommendation_data = ["Book A", "Book B", "Course A", "Course B", "Movie A", "Movie B"]
interaction_logs = []

# Function: Sentiment Analysis
def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result["label"], result["score"]

# Function: Real-time Speech Recognition
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now!")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand the audio."
        except sr.RequestError:
            return "Could not request results, check your internet connection."

# Function: Real-time Recommendations
def get_recommendations(user_input, data):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data + [user_input])
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    recommendations = [data[i] for i in cosine_similarities.argsort()[0][-3:][::-1]]
    return recommendations

# Function: Object Handling with RAG
def handle_query_with_rag(query):
    return f"Processed query: {query} (RAG simulation)"

# Function: Dashboard
def update_dashboard(logs):
    df = pd.DataFrame(logs, columns=["Timestamp", "Feature Used", "Input", "Output"])
    st.dataframe(df)

# Streamlit App Layout
st.title("Multi-Feature Real-Time App")
st.sidebar.header("Choose a Feature")
feature = st.sidebar.radio(
    "Select an option:", 
    ("Sentiment Analysis", "Real-Time Speech Recognition", "Recommendations", "Object Handling with RAG", "Dashboard")
)

# Sentiment Analysis Feature
if feature == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    user_text = st.text_input("Enter text for sentiment analysis:")
    if st.button("Analyze"):
        if user_text:
            label, score = analyze_sentiment(user_text)
            st.success(f"Sentiment: {label}, Confidence: {score:.2f}")
            interaction_logs.append([time.ctime(), "Sentiment Analysis", user_text, f"{label}, {score:.2f}"])
        else:
            st.error("Please enter text.")

# Real-Time Speech Recognition Feature
elif feature == "Real-Time Speech Recognition":
    st.header("Real-Time Speech Recognition")
    if st.button("Start Listening"):
        speech_text = speech_to_text()
        st.success(f"Recognized Text: {speech_text}")
        interaction_logs.append([time.ctime(), "Speech Recognition", "Audio Input", speech_text])

# Real-Time Recommendations Feature
elif feature == "Recommendations":
    st.header("Real-Time Recommendations")
    user_input = st.text_input("Enter your preference or keyword:")
    if st.button("Get Recommendations"):
        if user_input:
            recommendations = get_recommendations(user_input, recommendation_data)
            st.success(f"Recommendations: {', '.join(recommendations)}")
            interaction_logs.append([time.ctime(), "Recommendations", user_input, ", ".join(recommendations)])
        else:
            st.error("Please enter your preference or keyword.")

# Object Handling with RAG Feature
elif feature == "Object Handling with RAG":
    st.header("Object Handling with RAG")
    query = st.text_input("Enter your query:")
    if st.button("Handle Query"):
        if query:
            response = handle_query_with_rag(query)
            st.success(f"Response: {response}")
            interaction_logs.append([time.ctime(), "Object Handling with RAG", query, response])
        else:
            st.error("Please enter a query.")

# Dashboard Feature
elif feature == "Dashboard":
    st.header("Interaction Dashboard")
    update_dashboard(interaction_logs)
