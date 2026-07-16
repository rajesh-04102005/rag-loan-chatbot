# RAG Loan Advisory Chatbot

A full-stack Retrieval-Augmented Generation (RAG) chatbot that provides
loan-related guidance using FAISS vector search and Gemini LLM.

## Features
- AI-powered loan advisory chatbot
- FAISS-based document retrieval
- Gemini LLM integration
- Flask backend
- Clean frontend UI

## Tech Stack
- Python
- Flask
- FAISS
- Sentence Transformers
- Gemini API
- HTML, CSS, JavaScript

## How to Run
1. Clone the repository
2. Create a virtual environment
3. Install dependencies
4. Add a `.env` file with `GEMINI_API_KEY`
5. Run `python app.py`

## Deploy on Render
1. Push the repository to GitHub.
2. Create a new Render Web Service from the repository.
3. Use the included `render.yaml` or set the start command to `gunicorn app:app`.
4. Add the environment variable `GEMINI_API_KEY` in Render.
5. Keep `requirements.txt`, `index/`, `templates/`, and the source files committed so Render can build the app correctly.

## License
MIT License
