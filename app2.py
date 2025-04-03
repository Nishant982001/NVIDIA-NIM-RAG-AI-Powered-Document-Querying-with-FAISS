import streamlit as st  # Streamlit for UI
import os  # OS module for environment variable management
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA  # NVIDIA AI endpoints for LLM and embeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader  # Loads PDF files from directory and individual PDFs
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splits text into chunks for processing
from langchain.chains.combine_documents import create_stuff_documents_chain  # Chain to combine document information
from langchain_core.prompts import ChatPromptTemplate  # Creates structured prompts for LLM
from langchain_core.output_parsers import StrOutputParser  # Parses output from LLM
from langchain.chains import create_retrieval_chain  # Creates retrieval chain for querying documents
from langchain_community.vectorstores import FAISS  # FAISS for vector storage and similarity search

from dotenv import load_dotenv  # Loads environment variables from .env file
load_dotenv()

st.title("NVIDIA NIM RAG: AI-Powered Document Querying with FAISS")

# Load NVIDIA API Key from environment variables
nvidia_api_key = st.sidebar.text_input(label="Nvidia API key",type="password")

available_models = [
    "meta/llama3-70b-instruct",
    "mistralai/mistral-7b-instruct",
    "ai21/j2-ultra",
    "cohere/command-r-plus",
    "deepseek-ai/deepseek-r1"
]
model=st.sidebar.selectbox("Select an AI Model", available_models)
#["meta/llama3-70b-instruct","deepseek-ai/deepseek-r1"]

if not nvidia_api_key:
    st.info("Please add your nvidia api key to continue")
    st.stop()

# Initialize NVIDIA LLM with LLaMA3-70B-Instruct model
llm = ChatNVIDIA(model_name=model, nvidia_api_key=nvidia_api_key)

def vector_embedding(pdf_path):
    """
    Function to create vector embeddings from uploaded PDF document.
    Stores embeddings in session state to avoid reprocessing.
    """
    if "vectors" not in st.session_state:
        # Initialize NVIDIA embeddings
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.write("Vector Embedding is in process please wait......")
        
        # Load uploaded PDF file
        st.session_state.loader = PyPDFLoader(pdf_path)
        st.session_state.docs = st.session_state.loader.load()
        
        # Split documents into chunks of 700 characters with 50-character overlap
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        
        # Store document embeddings in FAISS vector database
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        

# Streamlit UI title


# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    pdf_path = os.path.join("./", uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Pdf uploaded succesfully!")
    vector_embedding(pdf_path)
    st.write("FAISS Vector Store DB is ready using NVIDIA Embeddings")

# Define structured prompt template for querying documents
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# User input field for entering query
prompt1 = st.text_input("Enter Your Question From Documents")

import time  # Importing time to measure response time

# If user has entered a question
if prompt1 and "vectors" in st.session_state:
    # Create document chain with LLM and prompt template
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Retrieve stored vectors for similarity search
    retriever = st.session_state.vectors.as_retriever()
    
    # Create a retrieval chain for querying documents
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Measure response time
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    print("Response time:", time.process_time() - start)
    
    def stream_data():
        for word in response['answer'].split(" "):
            yield word + " "
            time.sleep(0.02)
    
    # Display response in Streamlit UI
    # st.write(f"Assistant as [{model}] -", response['answer'])
    st.write(stream_data)
    st.write("Response time:", time.process_time() - start)

    # Expandable section to show relevant document chunks
    with st.expander("Document Similarity Search"):
        # Display retrieved document chunks
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("--------------------------------------")
