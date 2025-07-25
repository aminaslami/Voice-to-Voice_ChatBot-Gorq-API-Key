import gradio as gr
import whisper  # Corrected import for Whisper
from gtts import gTTS
import os
from groq import Groq
from tempfile import NamedTemporaryFile
import warnings

# Suppress the FP16 warning from Whisper
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Initialize Groq API Client
client = Groq(
    api_key="Your Groq API Key Here - replace with your actual API key",
)

# Load the Whisper model
model = whisper.load_model("base")

def speech_to_text(audio_path):
    try:
        # Check if audio file exists
        if not audio_path or not os.path.exists(audio_path):
            return "Error: Audio file not found"
        
        # Transcribe the audio
        transcription = model.transcribe(audio_path)["text"]
        print(f"Transcribed text: {transcription}")  # Debug log
        return transcription
    except Exception as e:
        print(f"Error in speech_to_text: {str(e)}")  # Debug log
        return f"Error transcribing audio: {str(e)}"

def generate_response(text):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": text,
                }
            ],
            model="llama3-8b-8192",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error in generate_response: {str(e)}")
        return f"Error generating response: {str(e)}"

def text_to_speech(text):
    try:
        tts = gTTS(text)
        output_audio = NamedTemporaryFile(suffix=".mp3", delete=False)
        tts.save(output_audio.name)
        return output_audio.name
    except Exception as e:
        print(f"Error in text_to_speech: {str(e)}")
        return None

def chatbot_pipeline(audio_path):
    try:
        # Step 1: Convert speech to text
        text_input = speech_to_text(audio_path)

        # Step 2: Get response from LLaMA model
        response_text = generate_response(text_input)

        # Step 3: Convert response text to speech
        response_audio_path = text_to_speech(response_text)

        return response_text, response_audio_path

    except Exception as e:
        return str(e), None

# Create Gradio Interface
iface = gr.Interface(
    fn=chatbot_pipeline,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speak or Upload Audio"),
    outputs=[
        gr.Textbox(label="Response Text"),
        gr.Audio(label="Response Audio")
    ],
    title="Real-Time Voice-to-Voice Chatbot",
    description="Record your voice or upload an audio file to chat with the AI assistant."
)

iface.launch()