import streamlit as st
from pytube import YouTube
import os
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
# Function to download audio from YouTube
def download_youtube_audio(url):
    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    output_path = audio_stream.download(output_path="./")  # Save audio to current directory
    base, ext = os.path.splitext(output_path)
    new_file = base + ".mp3"
    os.rename(output_path, new_file)
    return new_file

# Function to transcribe audio using OpenAI Whisper
def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    return transcript

# Function to translate text using OpenAI
def translate_text(text, target_lang):
    translation = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Translate the following text to {target_lang}: {text}",
            }
        ],
        model="gpt-3.5-turbo",
    )
    return translation.choices[0].message.content

def text_to_speech(text, voice):
    
    speech_file_path = Path(__file__).parent / "speech.mp3"
    response = client.audio.speech.create(
    model="tts-1",
    voice=voice,
    input=text
    )

    response.stream_to_file(speech_file_path)
    return speech_file_path
# Streamlit UI
st.title("YouTube Video Transcriber and Translator")

youtube_url = st.text_input("Enter YouTube Video URL:", "")

# Language selection for translation
lang_options = ["Arabic", "Spanish", "French", "German", "Italian", "Portuguese"]
target_lang = st.selectbox("Select language for translation:", options=lang_options)
# Dropdown for voice selection
voice = st.selectbox(
    "Choose the voice type:",
    ("alloy", "echo", "fable", "onyx", "nova", "shimmer")
)
if st.button("Transcribe and Translate"):
    if youtube_url != "":
        # Download the audio
        audio_path = download_youtube_audio(youtube_url)
        # Transcribe the audio
        transcript = transcribe_audio(audio_path)
        # Display the transcript
        
    
        translated_text = translate_text(transcript, target_lang)
        print(translated_text)
        st.text_area("Translated Transcript:", value=translated_text, height=300)

        # Clean up the downloaded file
        os.remove(audio_path)  

        speech_file_path = text_to_speech(translated_text, voice)
        # Use the 'audio' method to display an audio player which can play the generated speech
        audio_file = open(speech_file_path, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3', start_time=0)
        audio_file.close() 
    else:
        st.write("Please enter a YouTube URL.")