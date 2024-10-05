import os
from mistralai import Mistral
import requests
import numpy as np
import faiss

def get_mistral_api_key():
    """
    Fetch the Mistral API key from environment variables.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found in environment variables.")
    return api_key

# Load data from a text file for a specific user
def load_user_data(file_name):
    with open(file_name, 'r') as file:
        return file.read()
    
# Split the user data into chunks for RAG
def split_text_into_chunks(text, chunk_size=2048):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Create embeddings for each text chunk
def get_text_embedding(client, input):
    embeddings_batch_response = client.embeddings.create(
        model="mistral-embed",
        inputs=input
    )
    return embeddings_batch_response.data[0].embedding

# Create embeddings
def create_embeddings(client, chunks):
    return np.array([get_text_embedding(client, chunk) for chunk in chunks])

# Load embeddings into FAISS for vector search
def load_embeddings_into_faiss(embeddings):
    d = embeddings.shape[1]  # Embedding dimension
    index = faiss.IndexFlatL2(d)  # L2 distance
    index.add(embeddings)
    return index

# Retrieve relevant chunks from FAISS
def retrieve_similar_chunks(index, query_embedding, chunks, k=2):
    D, I = index.search(query_embedding, k)
    return [chunks[i] for i in I[0]]

# Ask question and retrieve chunks using Mistral API
def ask_mistral_question():
    api_key = get_mistral_api_key()
    client = Mistral(api_key=api_key)

    # Load user data from text file
    user_text = load_user_data('user_1.txt')
    
    # Split text into chunks
    chunks = split_text_into_chunks(user_text)

    # Create embeddings for chunks
    embeddings = create_embeddings(client, chunks)

    # Load embeddings into FAISS
    index = load_embeddings_into_faiss(embeddings)

    # Prompt for user input
    question = input("Ask Mistral AI a question: ")

    # Create embedding for the user's question
    question_embedding = np.array([get_text_embedding(client, question)])

    # Retrieve relevant chunks from FAISS
    retrieved_chunks = retrieve_similar_chunks(index, question_embedding, chunks)

    # Combine retrieved chunks with the user's question for final prompt
    context = "\n".join(retrieved_chunks)
    final_prompt = f"""
    Context information is below:
    ---------------------
    {context}
    ---------------------
    Given the context information and no prior knowledge, answer the query.
    Query: {question}
    Answer:
    """

    # Use Mistral to generate the final response
    chat_response = client.chat.complete(
        model="pixtral-12b-2409", # Use the Pixtral model 
        messages=[
            {
                "role": "user",
                "content": final_prompt,
            },
        ]
    )

    # Print the response
    response_content = chat_response.choices[0].message.content
    print("\nMistral AI Response:\n" + response_content)

if __name__ == "__main__":
    try:
        ask_mistral_question()
    except Exception as e:
        print(f"Error: {e}")
