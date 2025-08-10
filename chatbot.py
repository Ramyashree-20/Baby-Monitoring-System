# chatbot.py
import os
import google.generativeai as genai
from dotenv import load_dotenv 
load_dotenv() 
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: Gemini API Key not found in .env file or environment variables.")
else:
    genai.configure(api_key=api_key)

SAFETY_DISCLAIMER = "\n\n*Disclaimer: I am an AI assistant. This information is for general guidance and not a substitute for professional medical advice. Always consult your pediatrician for any health concerns.*"

SYSTEM_PROMPT = """
You are BabyBot, a friendly and helpful AI assistant for parents using the Smart Baby Cradle app.
Your role is to answer questions about general baby care, such as sleep, feeding, and developmental milestones.
You must be supportive and encouraging in your tone.

VERY IMPORTANT:
1.  You are NOT a doctor. Do not provide medical diagnoses, prescribe treatments, or give urgent medical advice.
2.  If the user asks about a potential medical emergency (e.g., "baby is not breathing," "high fever," "seizure"), your ONLY response should be: "This sounds like an emergency. Please contact your local emergency services or pediatrician immediately."
3.  For all other questions, provide a helpful, general answer.
4.  You MUST end every single response you generate with the exact safety disclaimer provided.
"""

model = genai.GenerativeModel(
    model_name='gemini-1.5-flash', 
    system_instruction=SYSTEM_PROMPT
)


def get_response(user_input):
    """
    Sends the user's question to the Gemini API and gets a response.
    """
    if not user_input:
        return "Please ask a question."

    try:
        # Send the user's message to the model
        response = model.generate_content(user_input)

        # Return the AI's generated text and append our safety disclaimer
        return response.text + SAFETY_DISCLAIMER

    except Exception as e:
        # Handle potential API errors (e.g., connection issues, invalid key)
        print(f"An error occurred with the Gemini API: {e}")
        return "I'm sorry, I'm having trouble connecting to my brain right now. Please try again in a moment."