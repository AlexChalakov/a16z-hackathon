# **a16z & Mistral hackathon - AI Companion**

This project leverages the **Mistral AI** models to create an interactive emotional AI companion. It uses the **Pixtral-12B** model to process chat-based queries and image-based descriptions.

## **Features**
- **Text-based chat**: Interact with the AI using natural language.
- **Image analysis**: Get AI-powered descriptions of images.

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

### **5. Run the Application**
To start the application, simply run:
```bash
python main.py
```

---

## **Usage**
When you run the script, you will be prompted to ask a question to Mistral AI or upload an image for analysis. The AI will respond based on the input.

---

## **Contributing**
Pull requests are welcome. For significant changes, please open an issue first to discuss what you would like to change.
