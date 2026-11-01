# 🏥 AI-Powered Medical Chatbot

An intelligent AI-powered medical chatbot designed to provide users with general health information, symptom guidance, and answers to medical-related questions. The application uses Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to generate contextual responses using relevant medical knowledge.

## 🚀 Project Overview

The Medical Chatbot combines Artificial Intelligence, Natural Language Processing (NLP), and vector search technology to make medical information more accessible.

It uses Groq API for AI-powered conversational responses, Sentence Transformers for generating text embeddings, and Pinecone as a vector database for storing and retrieving relevant medical information.

## ✨ Features

- 💬 **AI Medical Chat:** Ask health-related questions and receive AI-generated responses.
- 🩺 **Symptom Guidance:** Get general information about common symptoms.
- 📚 **Medical Knowledge Retrieval:** Retrieve relevant information from medical documents using RAG.
- 🔍 **Semantic Search:** Find relevant medical information based on the meaning of questions.
- ⚡ **Fast AI Responses:** Use Groq API for fast LLM inference.
- 🧠 **Vector Embeddings:** Generate embeddings using Sentence Transformers.
- 🗄️ **Vector Database:** Store and retrieve medical document embeddings using Pinecone.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| FastAPI | Backend API |
| Groq API | LLM inference |
| Sentence Transformers | Text embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| Pinecone | Vector database |
| LangChain | RAG pipeline |
| React.js | Frontend (if implemented) |

## 🏗️ System Architecture

```text
                 User
                   |
                   v
             React Frontend
                   |
                   v
              FastAPI Backend
                   |
                   v
             User Question
                   |
                   v
          Sentence Transformers
             (Embeddings)
                   |
                   v
              Pinecone
           (Vector Search)
                   |
                   v
          Relevant Medical Data
                   |
                   v
                Groq API
             (LLM Model)
                   |
                   v
          Generated Response
                   |
                   v
             User Interface
```

## 📂 Project Structure

```text
Medical-chat-bot-application/
│
├── app/
│   ├── config.py
│   ├── clients.py
│   ├── knowledge.py
│   ├── chatbot.py
│   └── main.py
│
├── data/
│   └── medical_documents/
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Medical-chat-bot-application.git
```

### 2. Navigate to the Project Directory

```bash
cd Medical-chat-bot-application
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure Environment Variables

Create a `.env` file in the root directory.

```env
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=medical-chatbot
```

Get your API keys:

- Groq: https://console.groq.com/
- Pinecone: https://app.pinecone.io/

### 7. Run the Application

```bash
uvicorn app.main:app --reload
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

## 🧠 How RAG Works

The application follows a Retrieval-Augmented Generation workflow:

1. **Document Loading:** Load medical documents such as PDFs.
2. **Text Splitting:** Divide documents into smaller text chunks.
3. **Embedding Generation:** Convert text chunks into numerical vectors using Sentence Transformers.
4. **Vector Storage:** Store generated vectors in Pinecone.
5. **User Query:** Receive a medical question from the user.
6. **Similarity Search:** Retrieve relevant medical information from Pinecone.
7. **Response Generation:** Send the retrieved context and user question to the Groq LLM.
8. **Final Response:** Return the generated answer to the user.

## 🔐 Environment Variables

Never upload your `.env` file or API keys to GitHub.

Add the following to `.gitignore`:

```gitignore
venv/
.env
__pycache__/
*.pyc
.pytest_cache/
```

## ⚠️ Medical Disclaimer

This application is developed for educational and informational purposes only. It does not provide professional medical diagnosis, treatment, or prescriptions. AI-generated responses may contain inaccuracies. Always consult a qualified healthcare professional for medical advice.

For emergencies, contact your local emergency services.

## 🔮 Future Enhancements

- Medical report analysis.
- Medical image understanding.
- Voice-based medical conversations.
- User authentication and chat history.
- Multilingual medical assistance.
- Improved medical document retrieval.
- Integration with verified medical knowledge sources.

## 👨‍💻 Author

**Jins Joseph**

Self-taught MERN Stack Developer | AI & Machine Learning Enthusiast

- GitHub: https://github.com/JinzJoseph
- LinkedIn: https://www.linkedin.com/in/jins-joseph-79291b22b/

---

⭐ If you find this project interesting, consider giving it a star!

**Built with Python, AI, RAG, Groq, and Pinecone.**