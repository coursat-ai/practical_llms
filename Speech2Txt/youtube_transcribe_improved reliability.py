import streamlit as st
from pytube import YouTube
import os
from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path
import time

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

def postprocess(transcript):
    # Use GPT-4 to postprocess the transcript
    # Transcribe in the same language as the video
    
    system_prompt = """The following is a transcript of a YouTube video in Arabic. 
                    Please improve it and make it more readable. 
                    Do not summarize the content.
                    If a term is mentioned in English, keep it as is.
                    Provide your output in the same language of the video."""
    postprocessed_transcript = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": transcript,
        }
    ],
    temperature=0,
    #model="gpt-4-1106-preview",
    model="gpt-4o",
)
    return postprocessed_transcript.choices[0].message.content

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
st.title("YouTube Video Transcriber")

youtube_url = st.text_input("Enter YouTube Video URL:", "")

if st.button("Transcribe"):
    if youtube_url != "":
        # Download the audio
        print("Downloading file...")
        start_time = time.time()
        audio_path = download_youtube_audio(youtube_url)
        end_time = time.time()
        execution_time = end_time - start_time
        print("File downloaded in", execution_time, "seconds")
        
        #audio_path = "C:\\Users\\aelsallab\\My Drive\\Colab Notebooks\\Chatbots\\LLM Course\\LLM Course\\practical_llms\\Speech2Txt\\video.mp3"
        
        # Transcribe the audio
        print("Transcribing audio...")
        start_time = time.time()
        transcript = transcribe_audio(audio_path)
        end_time = time.time()
        execution_time = end_time - start_time
        print("Transcription completed in", execution_time, "seconds")
        st.write("Transcription completed in", execution_time, "seconds")
        # Display the transcript
        st.text_area("Original Transcript:", value=transcript, height=300)
        print('Postprocessing transcript...')
        start_time = time.time()
        transcript = postprocess(transcript)
        end_time = time.time()
        execution_time = end_time - start_time
        print("Postprocessing completed in ", execution_time, "seconds")
        st.write("Postprocessing completed in ", execution_time, "seconds")
        st.text_area("Post Processed Transcript:", value=transcript, height=300)
        # Clean up the downloaded file
        #os.remove(audio_path)
        
        print("Generating speech...")
        start_time = time.time()
        speech_file_path = text_to_speech(transcript, "alloy")
        end_time = time.time()
        execution_time = end_time - start_time
        print("Speech generated in ", execution_time, "seconds")
        st.write("Speech generated in ", execution_time, "seconds")
        # Use the 'audio' method to display an audio player which can play the generated speech
        audio_file = open(speech_file_path, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3', start_time=0)
        audio_file.close() 
    else:
        st.write("Please enter a YouTube URL.")
