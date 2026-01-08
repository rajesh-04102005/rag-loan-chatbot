import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
    
FAISS_INDEX_PATH = r"E:\Rajesh Resume projects\Rag_Application\index\faiss.index"
DOCS_PATH = r"E:\Rajesh Resume projects\Rag_Application\index\documents.npy"


client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    local_files_only=True   
)

index = faiss.read_index(FAISS_INDEX_PATH)
documents = np.load(DOCS_PATH, allow_pickle=True)

print("✅ RAG core loaded successfully")


def ask_rag(question: str) -> str:
    query_embedding = model.encode([question])

    distances, indices = index.search(query_embedding, k=5)
    context = "\n\n".join([documents[i] for i in indices[0]])

    prompt = f"""
You are a professional loan advisor.

Answer the question clearly and neatly using the context below.

Formatting rules:
- Use short paragraphs
- Use bullet points where applicable
- Highlight key terms using **bold**
- Do NOT add emojis
- Keep the answer concise and professional

Context:
{context}

Question:
{question}

Final Answer:
"""


    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text
