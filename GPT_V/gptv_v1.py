import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load API key from .env file
env_path = os.path.join("..", '.env')  # Adjust the path as necessary
load_dotenv(env_path)

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)
def get_image_comment(image_url):
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s in this image?"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content

# Streamlit app
st.title('Image Comment Generator')

# Input for image URL
image_url = st.text_input("Enter the URL of an image:", "")

if image_url:
    # Display the image from the URL
    st.image(image_url, caption='Image from URL', use_column_width=True)
    
    if st.button('Generate Comment'):
        # Generate a comment for the image using the URL
        comment = get_image_comment(image_url)
        st.write(comment)
else:
    st.write("Please enter an image URL to get started.")
