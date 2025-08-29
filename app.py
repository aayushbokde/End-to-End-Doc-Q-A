# import os
# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS #vector store db
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings #vector Embedding

# from dotenv import load_dotenv
# load_dotenv()

# groq_api_key = os.getenv("GROQ_API_KEY")
# os.environ['GOOGLE_API_KEY']=os.getenv("GOOGLE_API_KEY")

# st.title("Gemma Model DOC Q&A")

# llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

# # print(llm)

# prompt = ChatPromptTemplate.from_template(

# """
# Answer the questions based on the provided context only.
# Please provide the most accurate response based on the question
# <context>
# {context}
# <context>
# Questions:{input}
# """
# )

# def vector_embedding():
#     if "vectors" not in st.session_state:
#         st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#         st.session_state.loader=PyPDFDirectoryLoader("./US_Census") # data ingestion
#         st.session_state.docs=st.session_state.loader.load()#doc load
#         st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
#         st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


# prompt1 = st.text_input("Ask from the document")

# if st.button("Document Embedding"):
#     vector_embedding()
#     st.write("vector store db is ready")

# import time

# if prompt1:
#     document_chain=create_stuff_documents_chain(llm, prompt)
#     retriever = st.session_state.vectors.as_retriever()
#     retrieval_chain=create_retrieval_chain(retriever, document_chain)

#     start=time.process_time()
#     response=retrieval_chain.invoke({'input':prompt1})
#     st.write(response['answer'])

#     #With a streamlit expander
#     with st.expander("Document Similarity Search"): 
#         #Find the relevant chunks
#         for i, doc in enumerate(response["context"]):   
#             st.write(doc.page_content)
#             st.write("-------------------------------------------------")


import os
import time
import asyncio
import streamlit as st

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS  # vector store db
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # vector Embedding
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

# Streamlit app title
st.title("Gemma Model DOC Q&A")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Vector embedding function with asyncio fix
def vector_embedding(uploaded_file):
    if "vectors" not in st.session_state:
        # Ensure event loop exists (fix for grpc + streamlit threads)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
        
        #save uploaded file in temp dir
        file_path = os.path.join("temp_dir",uploaded_file.name)
        os.makedirs("temp_dir",exist_ok=True)
        with open(file_path,"wb") as f:
            f.write(uploaded_file.read())
        
        #load pdf
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        #process docs

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)
        vectors = FAISS.from_documents(final_documents, embeddings)

        st.session_state.embeddings = embeddings
        st.session_state.vectors = vectors
        st.write("Vector store db is ready")

#file upload
uploaded_file = st.file_uploader("Upload a pdf", type=["pdf"])

# User input
prompt1 = st.text_input("Ask from the document")

# Button to build vector store
if uploaded_file and st.button("Document Embedding"):
    vector_embedding(uploaded_file)
    st.write("Ready")
    

# Run query if user entered a question
if prompt1 and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.write(response['answer'])

    # Show relevant chunks
    with st.expander("📄 Document Similarity Search"): 
        for i, doc in enumerate(response["context"]):   
            st.write(doc.page_content)
            st.write("-------------------------------------------------")
