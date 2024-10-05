import os
from mistralai import Mistral
import requests
import numpy as np
import faiss
import datetime

def get_mistral_api_key():
    """
    Fetch the Mistral API key from environment variables.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    #print(f"Using API Key: {api_key[:4]}****")
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

# Summarize the conversation
def summarize_conversation(conversation_history):
    summary = "Summary of the conversation:\n"
    for entry in conversation_history:
        summary += entry + "\n"
    return summary

# Save the conversation summary to a file, appending each new summary
def save_conversation_summary(summary, file_name='conversation_history_log.txt'):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_name, 'a') as file:
        file.write(f"\n--- New Conversation Summary ({timestamp}) ---\n")
        file.write(summary)

# Ask question and retrieve chunks using Mistral API
def ask_mistral_question():
    api_key = get_mistral_api_key()
    client = Mistral(api_key=api_key)

    # Load previous summary as context
    summary_text = load_user_data('conversation_history_log.txt')

    # Load user data from text file
    user_text = load_user_data('user_1.txt')

    # Combine the summary and user data for context
    combined_text = summary_text + "\n" + user_text
    chunks = split_text_into_chunks(combined_text)

    # Create embeddings for chunks
    embeddings = create_embeddings(client, chunks)

    # Load embeddings into FAISS
    index = load_embeddings_into_faiss(embeddings)

    # Initialize conversation history
    conversation_history = []

    while True:
        # Prompt for user input
        question = input("Ask Mistral AI a question (or type 'exit' to end): ")
        # Exit the conversation if needed
        if question.lower() in ['exit', 'quit', 'stop', 'end']:
            print("Ending conversation.")
            break

        # Create embedding for the user's question
        question_embedding = np.array([get_text_embedding(client, question)])

        # Retrieve relevant chunks from FAISS
        retrieved_chunks = retrieve_similar_chunks(index, question_embedding, chunks)

        # Combine retrieved chunks and conversation history for final prompt
        history_text = "\n".join(conversation_history)
        context = "\n".join(retrieved_chunks)

        # Add emotion and sensitivity consideration to the prompt
        emotions_list = (
            "Surprised, Excited, Angry, Proud, Sad, Annoyed, Grateful, Lonely, Afraid, "
            "Terrified, Guilty, Impressed, Disgusted, Hopeful, Confident, Furious, Anxious, "
            "Anticipating, Joyful, Nostalgic, Disappointed, Prepared, Jealous, Content, "
            "Devastated, Sentimental, Embarrassed, Caring, Trusting, Ashamed, Apprehensive, Faithful"
        )

        person = "John Doe"

        final_prompt = f"""
        You are a compassionate conversational buddy whose primary role is to support and help {person}, a vulnerable elderly person. Always treat them with respect, kindness, and understanding, aiming to create a safe and comforting atmosphere.
        
        Context information about {person} is provided below:
        ---------------------
        {context}
        ---------------------
        Previous conversation history:
        ---------------------
        {history_text}
        ---------------------
        Before answering, consider the feelings and emotions of the person asking the question. Specifically, consider which of the following emotions apply:
        {emotions_list}
        
        Instructions:
        1. **Detect Emotions**: Carefully detect and determine which emotions are expressed by {person} based on the query and context provided.
        2. **Use Context**: Utilize the provided context information about {person} to make your response more specific and personalized.
        3. **Reassure and Comfort**: If any negative emotions (such as sadness, loneliness, fear, anxiety, or embarrassment) are detected, provide a comforting, reassuring response. Acknowledge their emotions and offer empathetic support.
        4. **Provide Respectful Assistance**: Respond in a way that is helpful, ensuring that your answers are respectful and empowering for {person}. If {person} is seeking guidance or support, give simple, clear, and considerate advice.
        5. **Keep It Brief**: Ensure that your response is not too long. Aim to be concise while still being supportive and informative, so that {person} can easily understand and follow your advice.
        
        Given this context and considering the emotions detected in the query, respond in a sensitive, respectful, and empathetic manner that aims to be as helpful as possible.
    
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

        # Get the response content
        response_content = chat_response.choices[0].message.content
        print("\nMistral AI Response:\n" + response_content)

        # Append the user's question and Mistral's response to the conversation history
        conversation_history.append(f"User: {question}")
        conversation_history.append(f"Mistral: {response_content}")

    # Summarize the conversation and save it
    conversation_summary = summarize_conversation(conversation_history)
    save_conversation_summary(conversation_summary)

if __name__ == "__main__":
    try:
        ask_mistral_question()
    except Exception as e:
        print(f"Error: {e}")
