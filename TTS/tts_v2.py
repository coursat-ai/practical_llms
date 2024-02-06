import streamlit as st
from pathlib import Path
from io import BytesIO
import base64
from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path

# Load API key from .env file
env_path = os.path.join("..", '.env')  # Adjust the path as necessary
load_dotenv(env_path)

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)



def text_to_speech(text, voice):
    
    speech_file_path = Path(__file__).parent / "speech.mp3"
    response = client.audio.speech.create(
    model="tts-1",
    voice=voice,
    input=text
    )

    response.stream_to_file(speech_file_path)
    return speech_file_path

st.title('Text to Speech with OpenAI')

# Dropdown for voice selection
voice = st.selectbox(
    "Choose the voice type:",
    ("alloy", "echo", "fable", "onyx", "nova", "shimmer")
)

user_input = st.text_area("Enter text for speech synthesis:")

if st.button('Generate Speech'):
    if user_input:
        speech_file_path = text_to_speech(user_input, voice)
        # Use the 'audio' method to display an audio player which can play the generated speech
        audio_file = open(speech_file_path, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3', start_time=0)
        audio_file.close()
    else:
        st.write("Please enter some text to convert to speech.")