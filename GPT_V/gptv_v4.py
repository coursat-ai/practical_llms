import streamlit as st
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
def get_image_comment(image_url, prompt):
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content
def text_to_speech(text, voice):
    
    speech_file_path = Path(__file__).parent / "speech.mp3"
    response = client.audio.speech.create(
    model="tts-1",
    voice=voice,
    input=text
    )

    response.stream_to_file(speech_file_path)
    return speech_file_path

# Streamlit app
st.title('Ask me anything about an image!')

# Input for image URL
image_url = st.text_input("Enter the URL of an image:", "")

if image_url:
    # Display the image from the URL: https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg
    st.image(image_url, caption='Image from URL', use_column_width=True)
    prompt = st.text_input('Enter a prompt for the image:', '')
    # Dropdown for voice selection
    voice = st.selectbox(
        "Choose the voice type:",
        ("alloy", "echo", "fable", "onyx", "nova", "shimmer")
        )
    if st.button('Generate Comment') and prompt:
        # Generate a comment for the image using the URL
        comment = get_image_comment(image_url, prompt)
        st.write(comment)
        
        speech_file_path = text_to_speech(comment, voice)
        # Use the 'audio' method to display an audio player which can play the generated speech
        audio_file = open(speech_file_path, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3', start_time=0)
        audio_file.close()

else:
    st.write("Please enter an image URL to get started.")
