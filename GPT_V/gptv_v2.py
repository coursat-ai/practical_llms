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

# Streamlit app
st.title('Ask me anything about an image!')

# Input for image URL
image_url = st.text_input("Enter the URL of an image:", "")

if image_url:
    # Display the image from the URL
    st.image(image_url, caption='Image from URL', use_column_width=True)
    prompt = st.text_input('Enter a prompt for the image:', '')
    if st.button('Generate Comment') and prompt:
        # Generate a comment for the image using the URL
        comment = get_image_comment(image_url, prompt)
        st.write(comment)
    else:
        st.error('Please enter a prompt.')
else:
    st.write("Please enter an image URL to get started.")
