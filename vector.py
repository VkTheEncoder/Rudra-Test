# vector.py

import os
import glob
import pandas as pd

from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# Constants
CSV_FOLDER = "./csv_data"
INDEX_DIR = "./faiss_index"

def get_retriever():
    # 1️⃣ Find all CSV files
    csv_files = glob.glob(os.path.join(CSV_FOLDER, "*.csv"))
    if not csv_files:
        print("❌ No CSV files found in folder:", CSV_FOLDER)
        exit(1)

    # 2️⃣ Initialize your embedding model
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    # 3️⃣ Load or build the FAISS index
    if os.path.exists(INDEX_DIR):
        print("📂 Loading existing FAISS index...")
        vector_store = FAISS.load_local(INDEX_DIR, embeddings)
    else:
        print("📦 Creating new FAISS index and indexing CSV rows…")
        documents, ids = [], []
        counter = 0

        for path in csv_files:
            print(f"🔍 Processing {path}")
            try:
                df = pd.read_csv(path, on_bad_lines="skip")
                df.columns = df.columns.str.strip()
                for i, row in df.iterrows():
                    text = " ".join(str(v) for v in row.values)
                    metadata = {
                        "source_file": os.path.basename(path),
                        "row_index": i,
                        "columns": df.columns.tolist(),
                    }
                    documents.append(Document(page_content=text, metadata=metadata))
                    ids.append(str(counter))
                    counter += 1
            except Exception as e:
                print(f"⚠️ Skipped {path}: {e}")

        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(INDEX_DIR)
        print(f"✅ Indexed {len(documents)} rows from {len(csv_files)} files.")

    # 4️⃣ Return a retriever (top-10 hits)
    return vector_store.as_retriever(search_kwargs={"k": 10})

# Export for easy import
retriever = get_retriever()
