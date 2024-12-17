import os
import fitz  # PyMuPDF for PDF parsing
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import openai  # Assuming OpenAI for LLM
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Initialize OpenAI API and other services
openai.api_key = 'YOUR_OPENAI_API_KEY'
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained model for embedding
faiss_index = None  # To store the FAISS index

# 1. Data Ingestion
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500):
    """Segment text into chunks."""
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def create_embeddings(chunks):
    """Convert chunks of text into embeddings."""
    embeddings = embedding_model.encode(chunks)
    return embeddings

def store_embeddings_in_faiss(embeddings):
    """Store the embeddings in FAISS for similarity search."""
    global faiss_index
    if faiss_index is None:
        # Initialize the FAISS index with the dimensionality of the embeddings
        faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(np.array(embeddings).astype(np.float32))

def process_pdf(pdf_path):
    """Process PDF file: extract text, chunk it, create embeddings, and store in FAISS."""
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    embeddings = create_embeddings(chunks)
    store_embeddings_in_faiss(embeddings)
    return chunks  # Return the original chunks for query processing later

# 2. Query Handling
def query_to_embedding(query):
    """Convert a query to its vector embedding."""
    return embedding_model.encode([query])[0]

def retrieve_relevant_chunks(query_embedding, top_k=5):
    """Retrieve the most relevant chunks using cosine similarity or FAISS."""
    global faiss_index
    # Perform similarity search using FAISS
    _, indices = faiss_index.search(np.array([query_embedding]).astype(np.float32), top_k)
    return indices

def generate_response(query, relevant_chunks):
    """Generate a response using the LLM based on the retrieved chunks."""
    context = "\n".join(relevant_chunks)
    prompt = f"Answer the following question based on the given context:\n\nContext: {context}\n\nQuestion: {query}\nAnswer:"
    
    response = openai.Completion.create(
        engine="gpt-4",  # or gpt-3.5-turbo, based on your needs
        prompt=prompt,
        max_tokens=500
    )
    return response.choices[0].text.strip()

# 3. Comparison Queries
def extract_comparison_terms(query):
    """Identify the terms to compare in the user's query."""
    # Simple example: extract terms based on specific keywords or fields (improve with NLP methods)
    terms = query.split(' vs ')  # Simplified logic for comparison-based queries
    return terms

def retrieve_comparison_data(terms):
    """Retrieve chunks related to comparison terms."""
    # Assuming terms match chunks
    comparison_chunks = []
    for term in terms:
        query_embedding = query_to_embedding(term)
        relevant_chunk_indices = retrieve_relevant_chunks(query_embedding)
        for idx in relevant_chunk_indices[0]:
            comparison_chunks.append(chunks[idx])
    return comparison_chunks

def generate_comparison_response(query, comparison_chunks):
    """Generate a comparison-based response."""
    context = "\n".join(comparison_chunks)
    prompt = f"Compare the following data based on the context:\n\nContext: {context}\n\nQuery: {query}\nAnswer:"
    
    response = openai.Completion.create(
        engine="gpt-4",  # Or use gpt-3.5-turbo
        prompt=prompt,
        max_tokens=500
    )
    return response.choices[0].text.strip()

# Example usage
pdf_path = 'sample.pdf'  # Your PDF file path
chunks = process_pdf(pdf_path)

# Handling a regular query
query = "What are the key metrics in this document?"
query_embedding = query_to_embedding(query)
relevant_indices = retrieve_relevant_chunks(query_embedding)
relevant_chunks = [chunks[i] for i in relevant_indices[0]]
response = generate_response(query, relevant_chunks)
print(response)

# Handling a comparison query
comparison_query = "Compare the sales figures between 2022 and 2023"
terms = extract_comparison_terms(comparison_query)
comparison_data = retrieve_comparison_data(terms)
comparison_response = generate_comparison_response(comparison_query, comparison_data)
print(comparison_response)
