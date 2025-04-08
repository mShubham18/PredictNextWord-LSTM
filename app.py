import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
# loading the model
model = load_model("models/new_model.keras")

#loading the tokenizer
with open("models/new_tokenizer.pkl","rb") as file:
    tokenizer = pickle.load(file)

#Predicting Function
## Function to predict the next word
def predict_next_word(model,tokenizer,text,max_sequence_length):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_length:
        #Ensuring the sequence length matches the max_sequbece
        token_list = token_list[-(max_sequence_length-1):]
    token_list = pad_sequences([token_list],maxlen=max_sequence_length-1,padding="pre")
    predicted = model.predict(token_list,verbose=0)
    predict_word_index = np.argmax(predicted,axis=1)
    for word, index in tokenizer.word_index.items():
        if index==predict_word_index:
            return word
    return None

#Streamlit App Interface

st.title("Next Word Prediction using LSTM Architecture Model")
input_text = st.text_input("Enter the sequence of words","To be or not to be")
if st.button("Predict the next word"):
    #retrieving the maximum sequence length from the model
    max_sequence_len = model.input_shape[1]+1
    next_word = predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f"Next word: {next_word}")