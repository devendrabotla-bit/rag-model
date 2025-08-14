import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
From langchain.chains.combine documents import cre create stuff_documents_chain
From langchain_core.prompts Import Chat PromptTemplate
from langchain.chains import create_retrieval_chain
From langchain_community vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
import time
#Load environment variables
Load_dotenv()
Set up Grog API key
#gres api key a os.getenv("GROO APIKEY"
groq api key=st.secrets["GROQ API KEY"]
Raw
Load environment variables
Load_dotenv()
#Set up Groq API key
#groq_api_key= os.getenv("GROQ_API KEY)
groq_api key = st.secrets ("GROO_API_KEY"
st.set_page_config(page_title="Dynamic RAG with Groq", layout="wide")
st.inge("PragyanAI_Transperent.png")
st.title("Dynamic RAG with Groq, FAISS, and Llama3")
Initialize session state for vector store and chat history
if "vector" not in st.session_state:
st.session_state.vector = None
30
if "chat_history" not in st.session state:
st.session_state.chat_history = [1]
Sidebar for document upload
with st.sidebar:
st.header("Upload Documents")
uploaded_files = st.file_uploader("Upload your PDF documents", type="pdf", accept_multiple_files=True)
if st.button("Process Documents"):
if uploaded_files:
with st.spinner("Processing documents..."):
spop
for file in uploaded_files:
To read the file, we first write it to a temporary file
wath opion(filename, "wb") as fo
fawrste(file.getbuffer())
Loader PyPOFLoader(file.name)
docs.extend(Loader, load())
[10:54 AM, 8/14/2025] J.Balaji Btech Classmate: text_splitter = RecursiveCharacter TextSplitter(chunk_size=1888, chunk_overlap-288)
final_documents = text_splitter.split_documents(docs)
# Use pre-trained model from Hugging Face for embeddings
embeddings HuggingFaceEmbeddings (model_name="all-MiniLM-L6-v2")
st.session state.vector = FAISS.from documents(final_o from documents(final_documents, embeddings)
st.success("Documents processed successfully!")
else:
st.warning("Please upload at least one document.")
Main chat interface
st.header("Chat with your Documents")
11m ChatGroqfgroq api key-groq_api_key, model name="Llama3-86-81921
[10:54 AM, 8/14/2025] J.Balaji Btech Classmate: text_splitter = RecursiveCharacter TextSplitter(chunk_size=1888, chunk_overlap-288)
final_documents = text_splitter.split_documents(docs)
# Use pre-trained model from Hugging Face for embeddings
embeddings HuggingFaceEmbeddings (model_name="all-MiniLM-L6-v2")
st.session state.vector = FAISS.from documents(final_o from documents(final_documents, embeddings)
st.success("Documents processed successfully!")
else:
st.warning("Please upload at least one document.")
Main chat interface
st.header("Chat with your Documents")
11m ChatGroqfgroq api key-groq_api_key, model name="Llama3-86-81921
[10:54 AM, 8/14/2025] J.Balaji Btech Classmate: #Create the prompt template
prompt = ChatPromptTemplate.from_templatet
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
(context)
<context>
Questions: (input)
for message in st.session_state.chat_history:
with st.chat_message(message["role"]):
st.markdown (message ["content"}
#Get user Input
if prompt_input st.chat_input("Ask a question about your documents..."):
If st.session_state.vector is not None:
with st.chat_message("user"):
st.markdown(prompt_input)
st.session_state.chat_history.append({"rote": "user", "content": prompt_input}}
with st.spinner("Thinking..."):
document_chain create stuff_documents_chain(tlm, prompt)
retriever = st.session_state.vector.as_retrieverli
retrieval chain = create_retrieval_chainfretriever, document chain)
start = time.process time()
with st.spinner("Thinking..."):2+0
document_chain = create_stuff_documents_chain(Lim, prompt)
retriever = st.session_state.vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
start time.process_time()
response = retrieval_chain.invoke({"input": prompt_input>>
response_time = time.process_time()
start
with st.chat_message("assistant"):
st.markdownt response['answer'])
st. info(f"Response time: {response_time:.2f) seconds")
st.session_state.chat history.append(("role": "assistant", "content": response['answer']}}
st.warning!"Please process your documents before asking questions.")
