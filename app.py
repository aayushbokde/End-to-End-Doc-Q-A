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
from PyPDF2 import PdfReader   # for extracting links

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📄 Gemma Model DOC Q&A")

# Initialize session state variables safely
if "links" not in st.session_state:
    st.session_state["links"] = {}
if "vectors" not in st.session_state:
    st.session_state["vectors"] = None

# -------------------------------
# Initialize LLM
# -------------------------------
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based only on the provided context.
    Provide the most accurate response.

    <context>
    {context}
    <context>
    Question: {input}
    """
)

# -------------------------------
# Function to extract links
# -------------------------------
def extract_links(file_path):
    page_links = {}
    try:
        reader = PdfReader(file_path)
        for page_num, page in enumerate(reader.pages, start=1):
            links_set = page_links.setdefault(page_num, set())
            if "/Annots" in page:
                for annot in page["/Annots"]:
                    annot_obj = annot.get_object()
                    if "/A" in annot_obj and "/URI" in annot_obj["/A"]:
                        uri = annot_obj["/A"]["/URI"].strip()
                        links_set.add(uri)
    except Exception as e:
        st.warning(f"⚠️ Could not extract links: {e}")
    return page_links

# -------------------------------
# Vector embedding function
# -------------------------------
def vector_embedding(uploaded_file):
    # Ensure event loop exists (fix for grpc + streamlit threads)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    
    # Save uploaded file
    file_path = os.path.join("temp_dir", uploaded_file.name)
    os.makedirs("temp_dir", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    
    # Load PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Extract links
    st.session_state["links"] = extract_links(file_path)

    # Process docs
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)
    vectors = FAISS.from_documents(final_documents, embeddings)

    # Save in session state
    st.session_state["vectors"] = vectors
    st.session_state["embeddings"] = embeddings
    st.success("✅ Vector store DB is ready")

# -------------------------------
# File upload
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload a PDF", type=["pdf"])

# User input
prompt1 = st.text_input("💬 Ask a question from the document")

# Button to build vector store
if uploaded_file and st.button("⚡ Document Embedding"):
    vector_embedding(uploaded_file)
    st.info("✅ Ready to query")

# -------------------------------
# Run query if user entered a question
# -------------------------------
if prompt1 and st.session_state["vectors"]:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state["vectors"].as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({'input': prompt1})
    st.subheader("🤖 Answer")
    st.write(response['answer'])

    # Show relevant chunks
    with st.expander("📄 Document Similarity Search"): 
        for i, doc in enumerate(response["context"]):   
            st.write(doc.page_content)
            st.write("-------------------------------------------------")

    # Show extracted links only if asked
    if any(word in prompt1.lower() for word in ["link", "url", "reference", "hyperlink"]):
        st.subheader("🔗 Extracted Links from PDF")
        if st.session_state["links"]:
            for page, uris in st.session_state["links"].items():
                if uris:
                    st.markdown(f"**Page {page}:**")
                    for uri in sorted(uris):
                        st.markdown(f"- [{uri}]({uri})")
        else:
            st.write("No hyperlinks found in this PDF.")




