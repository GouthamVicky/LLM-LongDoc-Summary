import nltk
import pandas as pd
from nltk.cluster import KMeansClusterer
import numpy as np

import numpy as np
import re
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from scipy.spatial import distance
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer,util



def clean_and_tokenize(sentence,stop_words):
    sentence = re.sub('[^a-zA-Z0-9.]', ' ', sentence)
    sentence = sentence.lower()
    tokens = sentence.split()
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def extractive_summary_generator(input_article_text,model,stop_words):
    sentences = sent_tokenize(input_article_text)
    corpus = [clean_and_tokenize(sentence,stop_words) for sentence in sentences]

    # Encode sentences using Sentence Transformers
    sentence_embeddings = model.encode(corpus)

    n_clusters = 30
    if len(sentence_embeddings) >= n_clusters:
        kmeans = KMeans(n_clusters, init='k-means++', random_state=42)
        y_kmeans = kmeans.fit_predict(sentence_embeddings)

        # Finding and printing the nearest sentence vector from cluster centroid
        my_list = []
        for i in range(n_clusters):
            my_dict = {}

            for j in range(len(y_kmeans)):
                if y_kmeans[j] == i:
                    my_dict[j] = distance.euclidean(kmeans.cluster_centers_[i], sentence_embeddings[j])

            if my_dict:  # Check if my_dict is not empty
                min_distance = min(my_dict.values())
                my_list.append(min(my_dict, key=my_dict.get))


        sentences = [sentences[i] for i in sorted(my_list) if len(sentences[i].split()) >= 4]
        extractive_summary = "\n".join(sentences)
        return extractive_summary
    else:
        return "\n".join(corpus)


def preprocess_text(text,model,stop_words):
    #Remove special characters and extra whitespace
    text = re.sub(r'[^a-zA-Z0-9.\s]', '', text)
    text = ' '.join(text.split())

    # Remove extra spaces
    text = ' '.join(text.split())

    # Remove extra multi-line breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)

    return extractive_summary_generator(text,model)


def remove_pattern_words(text):
    # Define patterns to match 'xmath' and 'xcite' and their variations
    pattern_xmath = r'xmath\d+'
    pattern_xcite = r'xcite'

    # Replace the pattern words with an empty string
    text_without_pattern_words = re.sub(pattern_xmath, '', text)
    text_without_pattern_words = re.sub(pattern_xcite, '', text_without_pattern_words)

    return text_without_pattern_words


def generate_prompt(article_text,model,stop_words):

    formatted_prompt = remove_pattern_words(f"### Please give me a brief summary of this research paper\n" \
                              f"### Paper : {preprocess_text(str(article_text),model)}\n\n" \
                              f"### Summary :")
    return formatted_prompt
