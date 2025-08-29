# 📄 End-to-End Document Q&A App

An interactive **Streamlit application** powered by **LangChain**, **Groq LLM**, and **Google Generative AI embeddings** that allows you to **upload a PDF and ask questions from it**.  
The app generates embeddings, stores them in a **FAISS vector database**, and retrieves the most relevant chunks to answer your queries.

---

## 🚀 Features
- 📂 Upload **any PDF** dynamically
- 🧩 Automatic document chunking with LangChain
- 🔍 Embedding generation using **Google Generative AI**
- 🗂️ Vector storage and similarity search using **FAISS**
- 🤖 Q&A with **Groq LLM (Gemma 9B IT)**
- 📑 View document chunks retrieved during query

---

## 🛠️ Tech Stack
- [Python](https://www.python.org/)  
- [Streamlit](https://streamlit.io/)  
- [LangChain](https://www.langchain.com/)  
- [Groq](https://groq.com/) LLM (`gemma2-9b-it`)  
- [Google Generative AI Embeddings](https://ai.google.dev/)  
- [FAISS](https://faiss.ai/) for vector storage  

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/aayushbokde/End-to-End-Doc-Q-A.git
   cd End-to-End-Doc-Q-A

Create a virtual environment

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows


Install dependencies

pip install -r requirements.txt


Set environment variables
Create a .env file in the root directory:

GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key

▶️ Usage

Run the Streamlit app:

streamlit run app.py


Steps:

Upload your PDF 📂

Click "Document Embedding" to build the vector store 🧩

Enter a question in the text box 💬

Get answers with context + similarity search 🎯

📌 To-Do / Future Improvements

✅ Support for multiple PDF uploads

✅ Deploy on Streamlit Cloud for live demo

🔄 Add support for more LLMs

📊 Add visual analytics for retrieved chunks

🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss your ideas.

📜 License

This project is licensed under the MIT License – feel free to use and modify.

✨ Built using Streamlit, LangChain, and Groq.


