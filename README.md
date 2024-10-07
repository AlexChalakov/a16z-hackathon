# **a16z & Mistral Hackathon - AI Mind Companion**

This project leverages **Mistral AI** models to create an interactive emotional AI companion, aimed at supporting the elderly and those with memory challenges. The AI Companion uses the **Pixtral-12B** model to process chat-based queries and image-based descriptions. Additionally, we have integrated a **Telegram Bot** for easy and personalized interactions.

We have also fine-tuned **Mistral Large** using the [Empathetic Dialogues](https://github.com/facebookresearch/EmpatheticDialogues) dataset to create a more empathetic conversational agent (API fine-tuning for **Pixtral-12B** is not yet available).

## **Features**

- **Text-Based Chat**: Interact with the AI using natural language. The bot is designed to provide empathetic and context-aware responses to support users.
- **Image Analysis**: Get AI-powered descriptions of images. The bot can analyze photos, such as old family pictures, to provide descriptive and emotional insight.
- **Memory Integration**: The bot maintains an ongoing conversation history, allowing for more personalized interactions.
- **Telegram Bot**: The AI Mind Companion is also available as a Telegram bot for easy access and direct conversations on any device. The bot can handle both text and images seamlessly.
- **Dynamic Name & Background Setting**: Users can dynamically set their name and provide additional background information to make responses more personalized.
- **Voice Interaction**: Voice-to-text functionality is available for text-based interactions through the terminal, providing enhanced accessibility.

## **Setup Instructions**

### **1. Clone the Repository**

```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### **2. Create a Virtual Environment**

Create a Python virtual environment and activate it.

```bash
python3 -m venv pixtral-env
source pixtral-env/bin/activate  # For Mac/Linux
```

### **3. Install Dependencies**

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

### **4. Set Up the Mistral API Key**

You’ll need an API key from [Mistral AI](https://console.mistral.ai). Once you have it, set it as an environment variable:

```bash
export MISTRAL_API_KEY="your_api_key"  # For Mac/Linux
```

### **5. Set Up the Telegram Bot Token**

To use the Telegram bot functionality, you'll need a bot token from [Telegram](https://core.telegram.org/bots). Set it as an environment variable:

```bash
export TELEGRAM_TOKEN="your_telegram_bot_token"  # For Mac/Linux
```

### **6. Run the Application**

To start the application, simply run:

```bash
python main.py
```

---

## **Usage**

- **Interactive Chat**: Run the application, and you will be prompted to choose between a text-based interaction or using the Telegram bot.
- **Telegram Bot**: You can use the Telegram bot for more convenient access. Just search for the bot using the handle provided (e.g., `@AI_Mind_Companion_Bot`) and start chatting or sending images for analysis.
- **Image Analysis**: Upload an image, and the AI will generate a detailed description and emotional context.
- **Voice Interaction**: Voice input is supported for text-based prompts in the terminal, providing a hands-free experience for users.

---

## **Contributing**

Pull requests are welcome. For significant changes, please open an issue first to discuss what you would like to change.

---

## **Future Plans**

- **Enhanced Emotional Analysis**: Further improve the AI's emotional understanding to provide more nuanced responses.
- **Telegram Voice Integration**: Connect the voice interaction to the Telegram bot for enhanced accessibility.
- **Extended Multi-Language Support**: Add support for additional languages to reach a broader audience.

---

## **Contact**

For any questions or suggestions, please feel free to reach out via the repository's issue tracker or by contacting the bot's developer directly. You can see more here: https://devpost.com/software/companionai#updates

