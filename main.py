import os
import requests
import numpy as np
import faiss
import datetime
from mistralai import Mistral
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from bot import help_command, start, set_name, save_background, handle_message, user_data  # Import the bot functions

conversation_history = []

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

def process_user_query(question, person=None):
    """
    Process user query and generate a response using dynamic background file.
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

    # Combine the summary (conversation history) and user background for context
    combined_text = summary_text + "\n" + background_text
    chunks = split_text_into_chunks(combined_text)

    # Create embeddings for chunks
    embeddings = create_embeddings(client, chunks)

    # Load embeddings into FAISS
    index = load_embeddings_into_faiss(embeddings)

    # Create embedding for the user's question
    question_embedding = np.array([get_text_embedding(client, question)])

    # Retrieve relevant chunks from FAISS
    retrieved_chunks = retrieve_similar_chunks(index, question_embedding, chunks)

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
    
    Query: {question}
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
    conversation_history.append(f"User: {question}")
    conversation_history.append(f"Mistral: {response_content}")
    conversation_summary = summarize_conversation(conversation_history)
    save_conversation_summary(conversation_summary)

    return response_content

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
    application.add_handler(CommandHandler("help", help_command))  # Add help command

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, lambda update, context: handle_message(update, context, process_user_query)))

    # Start the bot (polling)
    application.run_polling()

# --- CLI or Bot Selection ---

def interactive_loop():
    """
    Interactive CLI method for text-based conversation.
    """
    conversation_history = []

    while True:
        question = input("Ask Mistral AI a question (or type 'exit' to end): ")
        if question.lower() in ['exit', 'quit', 'stop', 'end']:
            print("Ending conversation.")
            break

        response = process_user_query(question)
        print("\nMistral AI Response:\n" + response)

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
