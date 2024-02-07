
import streamlit as st
import os
from gpt4all import GPT4All

# App title
st.set_page_config(page_title="💬 Chatbots evaluation")

# Replicate Credentials
with st.sidebar:
    st.title('💬 Chat with Different Chatbots')    

    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a model', ['Mixtral-8x7b-Instruct', 'Orca-Mini-3B'], key='selected_model')
    if selected_model == 'Mixtral-8x7b-Instruct': 
        llm = 'mistral-7b-instruct-v0.1.Q4_0.gguf'
    elif selected_model == 'Orca-Mini-3B':
        llm = 'orca-mini-3b-gguf2-q4_0.gguf'

    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=120, step=8)    

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


def generate_response(prompt_input):
    model = GPT4All(llm)# This will download the model locally, in windows: C:\Users\<usernmame>\AppData\Local\nomic.ai\GPT4All
    
    output = model.generate("User: " + prompt_input + "\nAssistant: ", 
                            max_tokens=max_length, 
                            top_p=top_p, 
                            temp=temperature)
    return output

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)