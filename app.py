import numpy as np 
from tensorflow.keras.datasets import imdb
import tensorflow as tf 
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Step-01: Load the imdb ataset word index 
word_index=imdb.get_word_index()
reverse_word_index={value: key for key , value in word_index.items()}

#Load the pre-trained model with relu activation function
model = load_model('SimpleRNN_IMDB_1.h5')

# Step-02: Helper function 
# function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

# Function to preprocess user input 
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word , 2) + 3 for word in words]
    padded_review= sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review 

# Step-03: Prediction function 
def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)
    sentiment= 'positive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment,prediction[0][0]


# Streamlit web app design 
import streamlit as st
st.title("IMDB moview Review Sentiment Analysis")
st.write("Enter a moview review to classifiy it as positive or negative")

# User input 
user_input=st.text_area("Moview Review")

if st.button("Classify"):

    preprocess_input=preprocess_text(user_input)

    #Make prediction 
    prediction=model.predict(preprocess_input)
    sentiment="Positive" if prediction[0][0]>0.5 else "Negative"

    #Display the result 
    st.write(f'Sentiment:{sentiment}')
    st.write(f"Prediction score:{prediction[0][0]}")
else:
    st.write("please enter a movie review")



    




