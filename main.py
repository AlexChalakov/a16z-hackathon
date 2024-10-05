import os
from mistralai import Mistral

def get_mistral_api_key():
    """
    Fetch the Mistral API key from environment variables.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found in environment variables.")
    return api_key

def ask_mistral_question():
    """
    Ask a question to Mistral's large language model and print the response.
    """
    # Load the API key and initialize the client
    api_key = get_mistral_api_key()
    model = "pixtral-12b-2409"
    client = Mistral(api_key=api_key)

    # Prompt for user input
    question = input("Ask Mistral AI a question: ")

    # Make a chat request to Mistral
    chat_response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "user",
                "content": question,
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
