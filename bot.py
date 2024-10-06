import os
from telegram import Update
from telegram.ext import ContextTypes

# Store dynamic data like person name and background info
user_data = {
    "name": "John Doe",  # Default name
    "background": ""
}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sends a welcome message when the bot is started."""
    await update.message.reply_text('Hello! I am your AI companion. Ask me anything!')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE, process_user_query):
    """Handles messages from the user and generates an AI response."""
    user_message = update.message.text
    ai_response = process_user_query(user_message, user_data["name"])
    await update.message.reply_text(ai_response)

async def set_name(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sets the person's name dynamically."""
    if context.args:
        user_data["name"] = " ".join(context.args)  # Set name dynamically
        await update.message.reply_text(f"Name set to: {user_data['name']}")
    else:
        await update.message.reply_text("Please provide a name.")

async def save_background(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Save the background text provided by the user as a file."""
    if context.args:
        background_text = " ".join(context.args)
        user_data["background"] = background_text
        
        # Save the background text to a file named after the user's name
        background_file_name = f'{user_data["name"]}_background.txt'
        with open(background_file_name, 'w') as file:
            file.write(background_text)
        
        await update.message.reply_text(f"Background information saved for {user_data['name']}.")
    else:
        await update.message.reply_text("Please provide the background text.")

import os

async def process_image(update: Update, context: ContextTypes.DEFAULT_TYPE, process_user_query):
    """Handles image messages from the user and processes it with Pixtral."""

    # Create 'images' directory if it doesn't exist
    if not os.path.exists('images'):
        os.makedirs('images')
    
    # Get the file associated with the photo and download it
    photo_file = await update.message.photo[-1].get_file()
    local_image_path = os.path.join('images', 'user_image.jpg')
    await photo_file.download_to_drive(local_image_path)  # Download the image locally

    # Pass the image path to process_user_query for further processing and response
    ai_response = process_user_query(local_image_path, user_data["name"], is_image=True)

    # Respond to the user with the AI's output about the image
    await update.message.reply_text(ai_response)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sends a list of available commands."""
    help_text = (
        "/start - Start the conversation with the AI companion\n"
        "/name - Set the user's name for personalized responses\n"
        "/background - Provide background information for context\n"
        "/help - Get information about available commands"
    )
    await update.message.reply_text(help_text)
