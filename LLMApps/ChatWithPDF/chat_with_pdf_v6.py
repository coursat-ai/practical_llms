import os
import openai
import streamlit as st
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize the OpenAI client
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    raw_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    return raw_text


def generate_embeddings():
    embeddings_model = OpenAIEmbeddings(chunk_size=1000)
    return embeddings_model

def create_vector_database(raw_text):
    # Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )
    texts = text_splitter.split_text(raw_text)

    vec_db = FAISS.from_texts(texts, generate_embeddings())
    return vec_db


def retrieve_relevant_context(query, vec_db, k=4):
    if vec_db != None:
        # This function runs Approximate Nearest Neighbors (ANN) search on the vector database
        docs = vec_db.similarity_search(query, k=k)
        return docs
    else:
        return None
    
def stream_chat_response(message, chat_history, system_msg_content, model_name, temperature, max_history_length):
    system_msg = [{"role": "system", "content": system_msg_content}]
    chat_history.append({"role": "user", "content": message})
    if len(chat_history) > max_history_length:
        chat_history = chat_history[-max_history_length:]
    messages = system_msg + chat_history

    # Here, integrate logic to query the vector database with the user message for additional context
    # ...

    stream = client.chat.completions.create(
        messages=messages,
        model=model_name,
        temperature=temperature,
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

def clear_chat():
    st.session_state.chat_history = []

def format_context(context):
    formatted_context = ""
    for i, doc in enumerate(context):
        formatted_context += f"**Context {i+1}:** {doc.page_content}\n\n"
    return formatted_context

def main():
    st.title("💬 Chat with AI - RAG Model and Vector DB")

    
    # Sidebar controls
    model_name = st.sidebar.selectbox("Choose the Model", ["text-davinci-003", "gpt-3.5-turbo", "gpt-4"], index=1)
    temperature = st.sidebar.slider("Set Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    max_history_length = int(st.sidebar.number_input("Max History Length", min_value=1, max_value=10, value=3))

    system_msg = st.sidebar.text_area("System Message (Persona)", value="", height=100)

    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")
    if 'vec_db' not in st.session_state:
        st.session_state.vec_db = None

    if st.sidebar.button("Create Vector Database") and uploaded_file:
        with st.spinner("Reading file..."):
            text = extract_text_from_pdf(uploaded_file)
            # Write the extracted text in temp file temp.txt
            with open("temp.txt", "w", encoding='utf-8') as f:
                f.write(text)
            f.close()
            st.session_state.vec_db = create_vector_database(text)
            st.sidebar.text("PDF processed and vector database created.")
    if st.sidebar.button("Delete Vector Database") and st.session_state.vec_db:
        st.session_state.vec_db = None
        st.sidebar.text("Vector database deleted.")

    if st.sidebar.button("Clear Chat"):
        clear_chat()



    # Session state to store chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    user_input = st.chat_input("Enter your message:", key="user_input")
    
    if user_input:
        st.chat_message("user").write(user_input)
        with st.spinner("Thinking..."):
            accumulated_response = ""
            placeholder = st.chat_message("AI").empty()
            
            
            system_msg += "\nUse the following extra context :\n{context}" 
            context = retrieve_relevant_context(query=user_input, vec_db=st.session_state.vec_db)
            if context != None:                           
                system_msg = system_msg.format(context=context)
                st.session_state.last_context = context
            else: 
                system_msg = system_msg.format(context="No context found")
                st.session_state.last_context = None

            for response_chunk in stream_chat_response(user_input, 
                                                       st.session_state.chat_history, 
                                                       system_msg, 
                                                       model_name, 
                                                       temperature, 
                                                       max_history_length):
                accumulated_response += response_chunk
                placeholder.markdown(accumulated_response)
            st.session_state.chat_history.append({"role": "assistant", "content": accumulated_response})

            # Dispaly the last query relevant context in side bar
            if 'last_context' in st.session_state:
                if st.session_state.last_context != None:
                    formatted_context = format_context(st.session_state.last_context)
                    st.sidebar.text_area("Last query relevant context:", value=formatted_context, height=300)
                    st.session_state.last_context = None

if __name__ == "__main__":
    main()
