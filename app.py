import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from heapq import nlargest
import string
import re

# Must be the first Streamlit command
st.set_page_config(page_title="Text Summarization", layout="wide", initial_sidebar_state="collapsed")

# Fallback tokenization functions
def simple_sentence_tokenize(text):
    return re.split(r'(?<=[.!?]) +', text)

def simple_word_tokenize(text):
    return re.findall(r'\w+', text.lower())

# Try to download NLTK data, use fallback if it fails
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    from nltk.tokenize import sent_tokenize, word_tokenize
    stop_words = set(stopwords.words('english'))
    st.success("NLTK data loaded successfully!")
except LookupError:
    st.warning("NLTK data not found. Using simple tokenization.")
    sent_tokenize = simple_sentence_tokenize
    word_tokenize = simple_word_tokenize
    stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])

punctuation = string.punctuation + '\n'

st.title("This is an Extractive Text Summarization Streamlit App.")
st.subheader("Using NLTK or Fallback Tokenization")

text = st.text_area("Enter your text here", height=200)
percent = st.number_input("Enter the ratio of summary (0-1)", min_value=0.0, max_value=1.0, value=0.3)

def generate_summary():
    if text:
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

if st.button("Generate Summary"):
    generate_summary()