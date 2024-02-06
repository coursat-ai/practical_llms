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


def generate_image(prompt):
    response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
    image_url = response.data[0].url
    return image_url

def get_image_comment(image_url):
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content

# Streamlit app
st.title('DALL-E Image Generator and Commenter')

# Input for the DALL-E prompt
prompt = st.text_input("Enter a prompt to generate an image:", "")

if prompt:
    if st.button('Generate Image'):
        # Generate an image with DALL-E
        image_data = generate_image(prompt)
        
        # Display the generated image
        # This will depend on how the image data is returned. You might need to convert it to a displayable format
        st.image(image_data, caption='Generated Image', use_column_width=True)
        
        # Generate a comment for the image
        comment = get_image_comment(image_data)
        st.write(comment)
else:
    st.write("Please enter a prompt to generate an image.")
