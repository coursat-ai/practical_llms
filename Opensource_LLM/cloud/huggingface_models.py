
import streamlit as st
import os
from transformers import AutoTokenizer
import transformers
import torch

# App title
st.set_page_config(page_title="💬 Chatbots evaluation")

# Replicate Credentials
with st.sidebar:
    st.title('💬 Chat with Different Chatbots')
    if 'HF_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        hf_api = st.secrets['HF_TOKEN']
        os.environ['HF_TOKEN'] = hf_api
    else:
        hf_api = st.text_input('Enter HF API token: \n To login, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens', type='password')
        if hf_api:            
            st.success('Proceed to entering your prompt message!', icon='👉')
            os.environ['HF_TOKEN'] = hf_api
        else:
            st.warning('Please enter your HF API token to proceed!', icon='👈')

    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a model', ['Llama2-7B', 'Llama2-13B', 'Mixtral-8x7b-Instruct', 'Faclon40b-Instruct'], key='selected_model')
    if selected_model == 'Llama2-7B':
        llm = "meta-llama/Llama-2-7b-chat-hf"
    elif selected_model == 'Llama2-13B':#https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
        llm = 'meta-llama/Llama-2-13b-chat-hf'
    elif selected_model == 'Mixtral-8x7b-Instruct': #https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1
        llm = 'Mixtral-8x7b-Instruct'
    elif selected_model == 'Faclon40b-Instruct':
        llm = 'tiiuae/falcon-40b-instruct'
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

# Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
def generate_response(prompt_input):
    tokenizer = AutoTokenizer.from_pretrained(llm)
    pipeline = transformers.pipeline(
        "text-generation",
        model=llm,
        torch_dtype=torch.float32,#float16 on GPU, float32 on CPU
        #device_map="auto",
    )

    sequences = pipeline(
        "User: " + prompt_input + "\nAssistant: ",
        do_sample=True,
        top_k=1,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=max_length,
    )
    output = ""
    for seq in sequences:
        #print(f"Result: {seq['generated_text']}")
        output += seq['generated_text']
    return output

# User-provided prompt
if prompt := st.chat_input(disabled=not hf_api):
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