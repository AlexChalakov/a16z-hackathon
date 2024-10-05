import os
from mistralai import Mistral
import requests
import numpy as np
import faiss
import datetime
from gtts import gTTS
from pyt2s.services import stream_elements
import pygame
import speech_recognition as sr

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

# Convert text to speech using pyt2s and play it using pygame
def speak_text(text):
    # Request TTS data using pyt2s
    data = stream_elements.requestTTS(text, stream_elements.Voice.Joanna.value)

    # Save the TTS audio as an mp3 file
    with open("response.mp3", "wb") as file:
        file.write(data)

    # Initialize pygame mixer
    pygame.mixer.init()

    # Load and play the audio file
    pygame.mixer.music.load("response.mp3")
    pygame.mixer.music.play()

    # Wait for playback to finish
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    # Remove the audio file after playing
    os.remove("response.mp3")


# Get voice input and convert to text
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        recognizer.energy_threshold = max(recognizer.energy_threshold, 400)
        print("Listening... Please speak now.")

        try:
            # Attempt to capture speech for a short time
            audio = recognizer.listen(source, timeout=2, phrase_time_limit=5)
            # If no error occurs during listen, try to recognize the speech
            text = recognizer.recognize_sphinx(audio)
            if not text:
                raise sr.UnknownValueError()  # Raise an error if empty text is captured

            print(f"You said: {text}")
            return text
        except sr.WaitTimeoutError:
            # If no speech is detected within the timeout, fall back to typing
            print("No voice detected, please try again")
            return None
        except sr.UnknownValueError:
            print("Sorry, I could not understand what you said. If you said nothing, please type your question")
            return None
        except sr.RequestError:
            print("Could not request results; check your internet connection.")
            return None

# Get text input from the user
def get_text_input():
    text_input = input("Please type your question: ")
    return text_input

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

# Main function for interacting with Mistral API
def ask_mistral_question():
    api_key = get_mistral_api_key()
    client = Mistral(api_key=api_key)

    # Load previous conversation as context
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

    input_mode = input(
        "Would you like to start with 'voice' or 'text' input? (type 'voice' or 'text'): ").strip().lower()
    while input_mode not in ['voice', 'text']:
        input_mode = input("Invalid input. Please type 'voice' or 'text': ").strip().lower()

    while True:
        # Get user input based on the current mode
        question = get_voice_input() if input_mode == 'voice' else get_text_input()

        if not question:
            continue

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

        system_prompt = f"""
        You are a compassionate conversational buddy whose primary role is to support and help {person}, a vulnerable elderly person. Always treat them with respect, kindness, and understanding, aiming to create a safe and comforting atmosphere.
        """
        user_prompt = f"""
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
        2. **Use Context and previous conversation history**: Use the provided context information about {person} and conversation history to make your response more specific and personalized.
        3. **Reassure and Comfort**: If any negative emotions (such as sadness, loneliness, fear, anxiety, or embarrassment) are detected, provide a comforting, reassuring response. Acknowledge their emotions and offer empathetic support.
        4. **Provide Respectful Assistance**: Respond in a way that is helpful, ensuring that your answers are respectful and empowering for {person}. If {person} is seeking guidance or support, give simple, clear, and considerate advice.
        5. **Keep It Brief**: Ensure that your response is not too long. Aim to be concise while still being supportive and informative, so that {person} can easily understand and follow your advice.
        
        Given this context and considering the emotions detected in the query, respond in a sensitive, respectful, and empathetic manner that aims to be helpful.
        
        enQuery: {question}
        Answer:
        """

        # Use Mistral to generate the final response
        chat_response = client.chat.complete(
            model="pixtral-12b-2409", # Use the Pixtral model
            messages=[
                {
                    "role": "system",
                    "content": system_prompt

                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ]
        )

        # Get the response content
        response_content = chat_response.choices[0].message.content

        # Print the response
        print("\nMistral AI Response:\n" + response_content)
        # Speak the response out loud
        speak_text(response_content)

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
