# Install required libraries
#pip install faiss-cpu sentence-transformers streamlit requests
pip install -r requirements.txt

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import requests

# Load your dataset
data = pd.read_csv('Hydra-Movie-Scrape.csv')

# Prepare the text data for indexing
text_data = data['Summary'].fillna('') + " " + data['Short Summary'].fillna('')

# Initialize the embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(text_data.tolist(), convert_to_tensor=True).cpu().detach().numpy()

# Initialize FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Groq API Key and Endpoint
GROQ_API_KEY = "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"  # Replace with your actual Groq API key
GROQ_API_ENDPOINT = "https://api.groq.com/v1/meta-llama-3-8b-instruct/completions"  # Update endpoint as needed

# Define the RAG function
def rag_query(query, top_k=5):
    # Step 1: Get query embedding
    query_embedding = embedder.encode([query], convert_to_tensor=True).cpu().detach().numpy()

    # Step 2: Search for the most relevant entries in the FAISS index
    distances, indices = index.search(query_embedding, top_k)
    results = data.iloc[indices[0]]

    # Step 3: Generate a context string from retrieved documents
    context = " ".join(results['Summary'][:top_k].fillna(''))

    # Step 4: Send the query to Groq API with the context
    payload = {
        "prompt": f"Question: {query}\nContext: {context}",
        "max_tokens": 50
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(GROQ_API_ENDPOINT, json=payload, headers=headers)
    response_data = response.json()

    # Extract and return the response text
    response_text = response_data["choices"][0]["text"]
    return response_text

# Streamlit interface
st.title("Movie Information RAG Chatbot")
st.write("Ask a question about movies, directors, or actors!")

# User input
query = st.text_input("Enter your query here:")
if st.button("Get Answer"):
    if query:
        # Get the RAG response
        response = rag_query(query)
        st.write("**Response:**")
        st.write(response)
    else:
        st.write("Please enter a query.")
