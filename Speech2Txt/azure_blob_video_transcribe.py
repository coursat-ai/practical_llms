import streamlit as st
from azure.storage.blob import BlobServiceClient
from moviepy.editor import VideoFileClip
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

# Function to download video from Azure Blob Storage and extract audio
def download_blob_video_extract_audio(container_name, blob_name):
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    
    download_video_path = os.path.join("./", 'video.mp4')
    with open(download_video_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    
    # Use moviepy to extract audio from the video
    try:
        clip = VideoFileClip(download_video_path)
        audio_path = os.path.splitext(download_video_path)[0] + ".mp3"
        clip.audio.write_audiofile(audio_path)
    finally:
        clip.close()    
        #download_file.close()

    # Clean up the downloaded video file
    # Attempt to delete the video file
    try:
        os.remove(download_video_path)
    except PermissionError as e:
        print(f"Warning: Could not delete the video file due to a permission error: {e}")

    
    return audio_path
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
                    Provide your output in the same language of the video."""
    
    # To avoid exceeding the maximum token limit, we split the transcript into chunks and process each chunk separately
    MAX_WORDS = 2096
    transcript_words = transcript.split()
    num_words = len(transcript_words)
    num_iterations = num_words // MAX_WORDS + 1

    postprocessed_transcript = ""
    
    for i in range(num_iterations):
        print(f"Processing chunk {i+1}/{num_iterations}")
        start_index = i * MAX_WORDS
        end_index = (i + 1) * MAX_WORDS
        partial_transcript = " ".join(transcript_words[start_index:end_index])

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": partial_transcript,
                }
            ],
            model="gpt-4-1106-preview",
        )

        postprocessed_transcript += response.choices[0].message.content

    return postprocessed_transcript

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

azure_container_name = st.text_input("Enter Azure Container Name:", "")
blob_name = st.text_input("Enter Blob Name:", "")

if st.button("Transcribe"):
    if azure_container_name != "" and blob_name != "":
        # Download the audio
        print("Downloading the audio...")
        audio_path = download_blob_video_extract_audio(azure_container_name, blob_name)
        # Transcribe the audio
        print("Transcribing the audio...")
        transcript = transcribe_audio(audio_path)
        # Display the transcript
        print("Postprocessing the transcript...")
        transcript = postprocess(transcript)
        st.text_area("Transcript:", value=transcript, height=300)
        # Write the transcript to txt file for future use
        with open("transcript.txt", "wb", encoding="utf-8") as file:
            file.write(transcript)
        # Clean up the downloaded file
        print("Cleaning up the downloaded file...")
        os.remove(audio_path)
        
        print("Generating speech from the transcript...")
        speech_file_path = text_to_speech(transcript, "alloy")
        # Use the 'audio' method to display an audio player which can play the generated speech
        audio_file = open(speech_file_path, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3', start_time=0)
        audio_file.close() 
    else:
        st.write("Please enter file details")
