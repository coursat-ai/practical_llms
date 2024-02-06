import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env file
env_path = os.path.join("..", '.env')  # Adjust the path as necessary
load_dotenv(env_path)

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Streamlit app layout
st.title('DALL-E Image Generator')

# Feature selection
feature = st.selectbox('Select Feature', ['Image Generation', 'Image Variations', 'Image Edits'])

def display_image(url):
    st.image(url, caption='Generated Image')

if feature == 'Image Generation':
    prompt = st.text_input('Enter a prompt for the image:', '')
    if st.button('Generate Image'):
        if prompt:
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            image_url = response.data[0].url
            display_image(image_url)
        else:
            st.error('Please enter a prompt.')

elif feature == 'Image Variations':
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
    if st.button('Create Variations') and uploaded_file is not None:
        response = client.images.create_variation(
            image=uploaded_file,
            n=2,
            size="1024x1024"
        )
        for img in response.data:
            display_image(img.url)

elif feature == 'Image Edits':
    uploaded_image = st.file_uploader("Upload the image you want to edit", type=['png', 'jpg', 'jpeg'])
    uploaded_mask = st.file_uploader("Upload the mask image", type=['png', 'jpg', 'jpeg'])
    edit_prompt = st.text_input('Enter edit instructions:', '')
    if st.button('Edit Image') and uploaded_image is not None and uploaded_mask is not None:
        response = client.images.edit(
            model="dall-e-2",
            image=uploaded_image,
            mask=uploaded_mask,
            prompt=edit_prompt,
            n=1,
            size="1024x1024"
        )
        image_url = response.data[0].url
        display_image(image_url)
