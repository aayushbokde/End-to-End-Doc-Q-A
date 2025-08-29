# 📄 End-to-End Document Q&A App  

🚀 [**Live Demo on Streamlit Cloud**](https://end-to-end-doc-q-a-2mxejzhuuwriawrevu4plc.streamlit.app/)  

![Python](https://img.shields.io/badge/Python-3.12-blue)  
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-orange)  
![LangChain](https://img.shields.io/badge/LangChain-Integration-green)  
![License](https://img.shields.io/badge/License-MIT-lightgrey)  

An interactive **Streamlit application** that lets you **upload a PDF and ask questions directly from it**.  
The app uses **LangChain**, **Groq LLM**, and **Google Generative AI embeddings** to generate vector embeddings, store them in a **FAISS vector database**, and retrieve the most relevant chunks for accurate answers.  

---

## ✨ Features
- 📂 Upload **any PDF** dynamically  
- 🧩 Automatic text chunking with LangChain  
- 🔍 Embedding generation using **Google Generative AI**  
- 🗂️ Vector storage & similarity search with **FAISS**  
- 🤖 Q&A powered by **Groq LLM (Gemma 9B IT)**  
- 📑 View retrieved document chunks for full transparency  

---

## 🛠️ Tech Stack
- [Python](https://www.python.org/)  
- [Streamlit](https://streamlit.io/)  
- [LangChain](https://www.langchain.com/)  
- [Groq LLM](https://groq.com/) (`gemma2-9b-it`)  
- [Google Generative AI](https://ai.google.dev/) (Embeddings)  
- [FAISS](https://faiss.ai/) – Vector Database  

---

## 📦 Installation (Local Setup)

1. **Clone the repository**
   ```bash
   git clone https://github.com/aayushbokde/End-to-End-Doc-Q-A.git
   cd End-to-End-Doc-Q-A
2. **Install dependencies**
pip install -r requirements.txt

3. **Set environment variables**

Create a .env file in the project root and add:

GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key

▶️ **Usage**

Run the Streamlit app:

streamlit run app.py

**Steps:**

Upload your PDF 📂

Click “Document Embedding” to build the vector store 🧩

Enter a question in the text box 💬

Get accurate answers + check retrieved context 🎯

**📌 Future Improvements**

✅ Support for multiple PDFs

🔄 Add support for more LLMs

📊 Visual analytics of retrieved chunks

💾 Persistent vector storage


**📜 License**

This project is licensed under the MIT License – feel free to use and modify.

✨ Built with Streamlit, LangChain, Groq, and Google Generative AI
