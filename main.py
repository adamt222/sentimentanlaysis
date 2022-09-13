import streamlit as st
import pandas as pd

header = st.beta_container()
dataset = st.beta_container()
model_training = st.beta_container()

with header:
    st.title("Welcome to this sentiment analysis app !")
    st.text("In this project I will train a model to try and figure the sentiment of some text out")
