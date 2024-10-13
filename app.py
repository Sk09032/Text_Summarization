import streamlit as st

# Must be the first Streamlit command
st.set_page_config(page_title="Text Summarization", layout="wide", initial_sidebar_state="collapsed")

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from heapq import nlargest
import string

# Function to download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

# Download required NLTK data
download_nltk_data()

# Set up stopwords and punctuation
stop_words = set(stopwords.words('english'))
punctuation = string.punctuation + '\n'

st.title("This is an Extractive Text Summarization Streamlit App.")
st.subheader("Using NLTK")

text = st.text_area("Enter your text here", height=200)
percent = st.number_input("Enter the ratio of summary (0-1)", min_value=0.0, max_value=1.0, value=0.3)

def generate_summary():
    if text:
        try:
            # Tokenize the text into sentences and words
            sentence_tokens = sent_tokenize(text)
            word_tokens = word_tokenize(text.lower())

            # Remove stopwords and punctuation
            word_tokens = [word for word in word_tokens if word not in stop_words and word not in punctuation]

            # Calculate word frequencies
            word_freq = FreqDist(word_tokens)

            # Normalize frequencies
            max_freq = max(word_freq.values())
            for word in word_freq.keys():
                word_freq[word] = word_freq[word] / max_freq

            # Calculate sentence scores
            sent_score = {}
            for sent in sentence_tokens:
                for word in word_tokenize(sent.lower()):
                    if word in word_freq.keys():
                        sent_score[sent] = sent_score.get(sent, 0) + word_freq[word]

            # Select top sentences
            select_length = max(1, int(len(sentence_tokens) * percent))
            summary = nlargest(select_length, sent_score, key=sent_score.get)

            # Join the summary sentences
            summary_text = ' '.join(summary)

            st.write("Summary:")
            st.write(summary_text)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if st.button("Generate Summary"):
    generate_summary()