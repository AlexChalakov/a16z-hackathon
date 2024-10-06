import base64
import os
import requests
import numpy as np
import faiss
import datetime
from gtts import gTTS
from pyt2s.services import stream_elements
import pygame
import speech_recognition as sr
from mistralai import Mistral
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from bot import help_command, process_image, start, set_name, save_background, handle_message, user_data  # Import the bot functions

conversation_history = []  # Store the conversation history globally

# --- Mistral AI Functions ---

def get_mistral_api_key():
    """
    Fetch the Mistral API key from environment variables.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found in environment variables.")
    return api_key

def load_user_data(file_name):
    """
    Load data from a text file for a specific user.
    If the file doesn't exist, return an empty string.
    """
    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            return file.read()
    else:
        return ""

def split_text_into_chunks(text, chunk_size=2048):
    """
    Split the user data into chunks for RAG.
    """
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def get_text_embedding(client, input):
    """
    Create embeddings for each text chunk.
    """
    embeddings_batch_response = client.embeddings.create(
        model="mistral-embed",
        inputs=input
    )
    return embeddings_batch_response.data[0].embedding

def create_embeddings(client, chunks):
    """
    Create embeddings for all chunks of text.
    """
    return np.array([get_text_embedding(client, chunk) for chunk in chunks])

def load_embeddings_into_faiss(embeddings):
    """
    Load embeddings into FAISS for vector search.
    """
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

def retrieve_similar_chunks(index, query_embedding, chunks, k=2):
    """
    Retrieve relevant chunks from FAISS by finding the closest match to the query embedding.
    """
    D, I = index.search(query_embedding, k)
    return [chunks[i] for i in I[0]]

def summarize_conversation(conversation_history):
    """
    Summarize the conversation into a readable format.
    """
    summary = "Summary of the conversation:\n"
    for entry in conversation_history:
        summary += entry + "\n"
    return summary

def save_conversation_summary(summary, file_name='conversation_history_log.txt'):
    """
    Save the conversation summary to a file, appending each new summary.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_name, 'a') as file:
        file.write(f"\n--- New Conversation Summary ({timestamp}) ---\n")
        file.write(summary)

def process_user_query(input_data, person=None, is_image=False):
    """
    Process user query (text or image) and generate a response using dynamic background file.
    """
    api_key = get_mistral_api_key()
    client = Mistral(api_key=api_key)

    # Ensure the person's name is set dynamically, fallback to "John Doe" if no name is given
    person = person if person else user_data["name"] if user_data["name"] else "John Doe"

    # Dynamically set the background filename based on the person's name
    background_file = f"{person}_background.txt"

    # Load the user's background information (if it exists)
    background_text = load_user_data(background_file)

    # Load previous summary as context (conversation history log)
    summary_text = load_user_data('conversation_history_log.txt')

    # If input is an image, process the image and add to context
    if is_image:
        image_description = process_image_with_pixtral(input_data)  # Process image
        user_data["background"] += f"\nImage content: {image_description}"
        with open(background_file, 'a') as file:
            file.write(f"\nImage content: {image_description}")

        # Incorporate the image description into the conversation history
        conversation_history.append(f"User shared an image. Description: {image_description}")
        input_data = f"Image content: {image_description}"

    # Combine the summary (conversation history) and user background for context
    combined_text = summary_text + "\n" + background_text
    chunks = split_text_into_chunks(combined_text)

    # Create embeddings for chunks
    embeddings = create_embeddings(client, chunks)

    # Load embeddings into FAISS
    index = load_embeddings_into_faiss(embeddings)

    # Create embedding for the user's query or image description
    query_embedding = np.array([get_text_embedding(client, input_data)])

    # Retrieve relevant chunks from FAISS
    retrieved_chunks = retrieve_similar_chunks(index, query_embedding, chunks)

    # Combine retrieved chunks and conversation history for final prompt
    context = "\n".join(retrieved_chunks)
    history_text = summary_text  # Including the previous conversation history as context
    emotions_list = (
        "Surprised, Excited, Angry, Proud, Sad, Annoyed, Grateful, Lonely, Afraid, "
        "Terrified, Guilty, Impressed, Disgusted, Hopeful, Confident, Furious, Anxious, "
        "Anticipating, Joyful, Nostalgic, Disappointed, Prepared, Jealous, Content, "
        "Devastated, Sentimental, Embarrassed, Caring, Trusting, Ashamed, Apprehensive, Faithful"
    )

    # Create the final prompt with all required context and emotion considerations
    final_prompt = f"""
    You are a compassionate conversational buddy whose primary role is to support and help {person}, a vulnerable elderly person. 
    Always treat them with respect, kindness, and understanding, aiming to create a safe and comforting atmosphere.
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
    
    Query: {input_data}
    Answer:
    """

    # Use Mistral to generate the final response
    chat_response = client.chat.complete(
        model="pixtral-12b-2409",
        messages=[
            {
                "role": "user",
                "content": final_prompt,
            },
        ]
    )

    response_content = chat_response.choices[0].message.content

    # Log the conversation history
    conversation_history.append(f"User: {input_data}")
    conversation_history.append(f"Mistral: {response_content}")
    conversation_summary = summarize_conversation(conversation_history)
    save_conversation_summary(conversation_summary)

    return response_content

# --- Voice and Speech Functions ---

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

# --- Image Recognition Functions ---
def encode_image(image_path):
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None
    
def process_image_with_pixtral(image_path):
    """Process the image using Pixtral's image functionality."""
    base64_image = encode_image(image_path)
    if not base64_image:
        return "Failed to process the image."

    api_key = get_mistral_api_key()
    client = Mistral(api_key=api_key)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }
            ]
        }
    ]

    # Get the chat response from Mistral AI
    chat_response = client.chat.complete(
        model="pixtral-12b-2409",
        messages=messages
    )

    # Return the content of the AI's response
    return chat_response.choices[0].message.content

# --- Bot Functions ---

def run_bot():
    """
    Initialize and run the Telegram bot using Application (for v21+).
    """
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    
    # Create the Application instead of Updater
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Register commands and handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("name", set_name))
    application.add_handler(CommandHandler("background", save_background))
    application.add_handler(CommandHandler("help", help_command))

    # Handle text messages from the user
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, lambda update, context: handle_message(update, context, process_user_query)))

    # Handle image messages from the user
    application.add_handler(MessageHandler(filters.PHOTO, lambda update, context: process_image(update, context, process_user_query)))

    # Start the bot (polling)
    application.run_polling()

# --- CLI or Bot Selection ---

def interactive_loop():
    """
    Interactive CLI method for text-based conversation.
    """
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

        response = process_user_query(question)
        print("\nMistral AI Response:\n" + response)

        # Speak the response out loud
        speak_text(response)

        # Append the user's question and Mistral's response to the conversation history
        conversation_history.append(f"User: {question}")
        conversation_history.append(f"Mistral: {response}")

    conversation_summary = summarize_conversation(conversation_history)
    save_conversation_summary(conversation_summary)

if __name__ == "__main__":
    """
    Main function to run either the CLI or the Telegram bot based on user choice.
    """
    print("Welcome to the AI Emotional Companion")
    print("Would you like to:")
    print("1. Have a conversation (Text-based interaction)")
    print("2. Use Telegram bot")
    
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        interactive_loop()
    elif choice == "2":
        run_bot()
    else:
        print("Invalid choice. Please restart and enter 1 or 2.")
