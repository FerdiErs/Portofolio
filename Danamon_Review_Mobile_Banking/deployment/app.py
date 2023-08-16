import streamlit as st
import re
import pandas as pd
from transformers import pipeline
from gensim.models import LdaModel
from gensim.corpora import Dictionary

# Function to preprocess text
def text_preprocess(teks):
    teks = teks.lower()
    teks = re.sub("@[A-Za-z0-9_]+", " ", teks)
    teks = re.sub("#[A-Za-z0-9_]+", " ", teks)
    teks = re.sub(r"\\n", " ", teks)
    teks = teks.strip()
    teks = re.sub(r"http\S+", " ", teks)
    teks = re.sub(r"www.\S+", " ", teks)
    teks = re.sub("[^A-Za-z\s']", " ", teks)
    return teks

# Function to perform inference and get the topic with the highest probability
def get_highest_probability_topic(lda_model, dictionary, new_document, topic_names):
    new_bow = dictionary.doc2bow(new_document.split())
    topic_distribution = lda_model.get_document_topics(new_bow, minimum_probability=0)

    highest_probability_topic = max(topic_distribution, key=lambda x: x[1])
    topic_id, probability = highest_probability_topic
    topic_name = topic_names.get(topic_id, f"Topic {topic_id}")

    return topic_name, probability

# Load sentiment analysis model
pretrained_name = "w11wo/indonesian-roberta-base-sentiment-classifier"
nlp = pipeline("sentiment-analysis", model=pretrained_name, tokenizer=pretrained_name)

# Streamlit app
def main():
    st.title("Sentiment Analysis and Topic Inference App")
    st.write("Enter your text below:")
    input_text = st.text_area("Input Text")

    if st.button("Analyze Sentiment"):
        processed_text = text_preprocess(input_text)
        result = nlp(processed_text)
        sentiment = result[0]['label']
        probability = result[0]['score']
        st.write("Sentiment:", sentiment)
        st.write("Probability:", probability)

    if st.button("Infer Topic"):
        lda_model = LdaModel.load("lda.model")
        dictionary = Dictionary.load("dictionary.dict")
        topic_names = {0: 'User Experience',
               1: 'App Features',
               2: 'Questions and Engagement',
               3: 'Opinion on Banking App',
               4: 'Mixed Feedback and Technical Issues',
                }

        inferred_topic, inferred_probability = get_highest_probability_topic(lda_model, dictionary, input_text, topic_names)
        st.write("Inferred Topic:", inferred_topic)
        st.write("Inference Probability:", inferred_probability)

if __name__ == "__main__":
    main()
