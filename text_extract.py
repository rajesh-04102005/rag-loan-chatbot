import os
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


PDF_PATH = r"E:\Rajesh Resume projects\Rag_Application\data\CN\DATASET.pdf"
INDEX_DIR = r"E:\Rajesh Resume projects\Rag_Application\index"
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
DOCS_PATH = os.path.join(INDEX_DIR, "documents.npy")


os.makedirs(INDEX_DIR, exist_ok=True)


def extract_text_from_pdf(file_path):
    extracted_text = ""

    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_text += page_text + "\n"

    return extracted_text


def chunk_text(text, chunk_size=4):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return [
        " ".join(lines[i:i + chunk_size])
        for i in range(0, len(lines), chunk_size)
    ]



model = SentenceTransformer("all-MiniLM-L6-v2")



text = extract_text_from_pdf(PDF_PATH)

documents = chunk_text(text)

embeddings = model.encode(documents)
embeddings = np.array(embeddings)



dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


faiss.write_index(index, FAISS_INDEX_PATH)
np.save(DOCS_PATH, documents)

print("FAISS index and documents saved successfully!")
