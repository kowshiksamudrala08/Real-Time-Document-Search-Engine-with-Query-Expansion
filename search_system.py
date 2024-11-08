import os
import re
import math
import time
import nltk
import streamlit as st
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# Download the stopwords if not already downloaded
nltk.download('stopwords')
nltk.download('wordnet')


# Initialize Stemmer, Stopwords, IDF, and other variables
ps = PorterStemmer()
# Load stopwords from nltk
stopwords = set(stopwords.words('english'))
idf = defaultdict(int)
vectors = {}
relevant_num = defaultdict(int)
N = 0
doc_query_relevance = {}



# Extract content between tags
def extract_section(text, start_tag, end_tag):
    return re.search(f'{start_tag}(.*?){end_tag}', text, re.DOTALL).group(1).strip()

# Tokenize and preprocess text
def tokenise(text, stopwords, ps):
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return [ps.stem(token) for token in tokens if token not in stopwords]

# Query Expansion using WordNet
def expand_query(query_tokens):
    expanded_tokens = set(query_tokens)
    for token in query_tokens:
        # Find synonyms for the token using WordNet
        synonyms = wn.synsets(token)
        for syn in synonyms:
            for lemma in syn.lemmas():
                expanded_tokens.add(lemma.name())
    return list(expanded_tokens)

# Compute TF-IDF for a document
def calculate_tf_idf(document, idf):
    vector = {}
    for token in document:
        vector[token] = vector.get(token, 0) + 1
    for key in vector:
        if idf.get(key) is not None:
            vector[key] = (1 + math.log10(vector[key])) * idf[key]
    return vector

# Calculate cosine similarity between two vectors
def cosine_similarity(vector1, vector2):
    dot_product = sum(vector1.get(key, 0) * vector2.get(key, 0) for key in set(vector1) & set(vector2))
    query_norm = math.sqrt(sum(value ** 2 for value in vector1.values()))
    document_norm = math.sqrt(sum(value ** 2 for value in vector2.values()))
    if query_norm == 0 or document_norm == 0:
        return 0
    return dot_product / (query_norm * document_norm)


# Streamlit main function
def main():
    # Set up the Streamlit page layout
    st.title("Real-Time Document Search Engine with Query Expansion")
    # Upload multiple documents or queries in real-time
    uploaded_files = st.file_uploader("Upload documents", type=["txt"], accept_multiple_files=True)
    query_input = st.text_input("Enter your query:")

    if uploaded_files:
        for uploaded_file in uploaded_files:
            text = uploaded_file.getvalue().decode("utf-8")
            # Process the document
            doc_tokens = tokenise(text, stopwords, ps)
            doc_vector = calculate_tf_idf(doc_tokens, idf)
            doc_id = uploaded_file.name  # Use the filename as the document ID
            vectors[doc_id] = doc_vector  # Add new document to vectors

            # Update IDF
            for token in set(doc_tokens):
                idf[token] += 1

        # Display a success message
        st.write(f"{len(uploaded_files)} document(s) uploaded successfully.")

    if query_input:
        # Tokenize and calculate TF-IDF for the query
        query_tokens = tokenise(query_input, stopwords, ps)
        
        # Query expansion
        expanded_query = expand_query(query_tokens)
        st.write(f"Expanded Query: {' '.join(expanded_query)}")
        
        query_vector = calculate_tf_idf(expanded_query, idf)

        # Compute cosine similarity between query and document vectors
        similarity = {}
        for doc_id, doc_vector in vectors.items():
            similarity[doc_id] = cosine_similarity(doc_vector, query_vector)

        # Sort documents by similarity and display top 5
        similarity = sorted(similarity.items(), key=lambda x: (x[1], x[0]), reverse=True)[:5]
        
        st.write("**Top 5 documents matching your query:**")
        for rank, (doc_id, score) in enumerate(similarity, start=1):
            st.write(f"{rank}. Document: {doc_id} - Similarity Score: {score:.4f}")

      

    # Continuously fetch new queries and documents, if needed
    # You can set a refresh timer or wait for new input in the form of a button
    # Example: st.button("Fetch New Data") to manually trigger a data update

if __name__ == "__main__":
    main()
