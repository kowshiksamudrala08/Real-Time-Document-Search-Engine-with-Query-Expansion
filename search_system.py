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

# # If they are not downloaded, download the stopwords and wordnet
nltk.download('stopwords')
nltk.download('wordnet')

# The initialization of Stemmer and Stopwords and IDF and other types of variables
ps = PorterStemmer()
# Load stopwords from nltk
stopwords = set(stopwords.words('english'))
idf = defaultdict(int)  # Hash table to define IDF of respective words
vectors = {}  # We will have a dictionary to store the vectors of TF-IDF for every document that we will be working on.
relevant_num = defaultdict(int)  # # To keep the count of relevance for each document
N = 0  # Total number of documents
doc_query_relevance = {}

# Get section of content between two tags in a document
def extract_section(text, start_tag, end_tag):
    #Uses regex to preview the texts located between the start and the end tags
    return re.search(f'{start_tag}(.*?){end_tag}', text, re.DOTALL).group(1).strip()

# Pre-processing of text done at tokenization stage
def tokenise(text, stopwords, ps):
    # Source This paper uses the NLTK package to tokenise the text, conduct lemmatisation and stopwords removal.
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return [ps.stem(token) for token in tokens if token not in stopwords]

#Using WordNet synonyms and dot product for query expansion
def expand_query(query_tokens):
    expanded_tokens = set(query_tokens)
    for token in query_tokens:
        #By searching through the WordNet database, identify with synonym for each token
        synonyms = wn.synsets(token)
        for syn in synonyms:
            for lemma in syn.lemmas():
                expanded_tokens.add(lemma.name())  # Add synonyms of expanded tokens
    return list(expanded_tokens)

 # Compute the TF–IDF a document
def calculate_tf_idf(document, idf):
    vector = {}
    # Term frequency calculation of each token
    for token in document:
        vector[token] = vector.get(token, 0) + 1
    # Calculate IDF of each token present in the document
    for key in vector:
        if idf.get(key) is not None:
            vector[key] = (1 + math.log10(vector[key])) * idf[key]
    return vector

# Find out the cosine of two vectors.
def cosine_similarity(vector1, vector2):
    dot_product = sum(vector1.get(key, 0) * vector2.get(key, 0) for key in set(vector1) & set(vector2))
    query_norm = math.sqrt(sum(value ** 2 for value in vector1.values()))
    document_norm = math.sqrt(sum(value ** 2 for value in vector2.values()))
    # If any of the norms are zero then it returns cosine similarity or 0
    if query_norm == 0 or document_norm == 0:
        return 0
    return dot_product / (query_norm * document_norm)

# Normalize document names by stripping spaces and converting to lowercase for a fair comparison
def normalize(doc):
    return doc.strip().lower()

# Calculate metrics with normalized document names
# Define relevant documents based on similarity score > 0
def get_relevant_docs(similarity):
    return [doc_id for doc_id, score in similarity.items() if score > 0]

# Define retrieved documents as all uploaded documents
def get_retrieved_docs(uploaded_files):
    return [uploaded_file.name for uploaded_file in uploaded_files]

# Updated evaluation function
def calculate_metrics(retrieved_docs, relevant_docs):
    # Normalize both sets of document names
    normalized_relevant_docs = {normalize(doc) for doc in relevant_docs}
    normalized_retrieved_docs = [normalize(doc) for doc in retrieved_docs]
    
    true_positive = len(set(normalized_retrieved_docs) & normalized_relevant_docs)
    precision = true_positive / len(normalized_retrieved_docs) if normalized_retrieved_docs else 0
    recall = true_positive / len(normalized_relevant_docs) if normalized_relevant_docs else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score

# Streamlit app main logic
def main():
    # Streamlit setup and file upload handling
    st.set_page_config(page_title="Search Engine System", page_icon=None)
    st.title("Real-Time Document Search Engine with Query Expansion")
    
    uploaded_files = st.file_uploader("Upload documents", type=["txt"], accept_multiple_files=True)
    query_input = st.text_input("Enter your query:")
    
    # Process uploaded documents
    if uploaded_files:
        similarity = {}
        for uploaded_file in uploaded_files:
            text = uploaded_file.getvalue().decode("utf-8")
            doc_tokens = tokenise(text, stopwords, ps)
            doc_vector = calculate_tf_idf(doc_tokens, idf)
            doc_id = uploaded_file.name  # Use the filename as the document ID
            vectors[doc_id] = doc_vector  # Store vector for each document

            # Update the IDF values for each token in the document
            for token in set(doc_tokens):
                idf[token] += 1
        
        st.success(f"{len(uploaded_files)} document(s) uploaded successfully.")
    
    # Process the input query
    if query_input:
        query_tokens = tokenise(query_input, stopwords, ps)
        expanded_query = expand_query(query_tokens)
        st.write(f"Expanded Query: {' '.join(expanded_query)}")
        
        # Compute the query vector
        query_vector = calculate_tf_idf(expanded_query, idf)

        # Compute cosine similarity between query and documents
        similarity = {}
        for doc_id, doc_vector in vectors.items():
            similarity[doc_id] = cosine_similarity(doc_vector, query_vector)

        # Rank documents based on similarity and show the top 5
        sorted_similarity = sorted(similarity.items(), key=lambda x: (x[1], x[0]), reverse=True)
        top_docs = sorted_similarity[:5]
        
        st.write("### Top 5 documents matching your query:")
        for rank, (doc_id, score) in enumerate(top_docs, start=1):
            st.write(f"{rank}. Document: {doc_id} - Similarity Score: {score:.4f}")

        # Define relevant docs and retrieved docs
        relevant_docs = get_relevant_docs(similarity)  # All documents with score > 0
        retrieved_docs = get_retrieved_docs(uploaded_files)  # All uploaded documents

        # Calculate evaluation metrics
        precision, recall, f1_score = calculate_metrics(retrieved_docs, relevant_docs)
        
        st.write("### Evaluation Metrics:")
        st.write(f"- Precision: {precision:.4f}")
        st.write(f"- Recall: {recall:.4f}")
        st.write(f"- F1 Score: {f1_score:.4f}")

# Run the main function
if __name__ == "__main__":
    main()
