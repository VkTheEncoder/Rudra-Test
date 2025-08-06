# vector.py

import os
import glob
import pandas as pd

from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

# Constants
CSV_FOLDER = "./csv_data"
DB_LOCATION = "./chroma_learning_db"
COLLECTION_NAME = "learning_materials"

def get_retriever():
    # 1️⃣ Gather all CSVs
    csv_files = glob.glob(os.path.join(CSV_FOLDER, "*.csv"))
    if not csv_files:
        print("❌ No CSV files found in folder:", CSV_FOLDER)
        exit(1)

    # 2️⃣ Initialize your embedding model
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    # 3️⃣ Configure Chroma to persist with DuckDB+Parquet
    client_settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=DB_LOCATION,
    )

    # 4️⃣ Instantiate (or load) the Chroma vector store
    is_first_time = not os.path.exists(DB_LOCATION)
    vector_store = Chroma(
        client_settings=client_settings,
        collection_name=COLLECTION_NAME,
        persist_directory=DB_LOCATION,
        embedding_function=embeddings
    )

    # 5️⃣ On first run, read CSVs → Documents → index
    if is_first_time:
        print("📦 Creating new vector store and indexing CSV rows...")
        documents, ids = [], []
        doc_id = 0

        for file_path in csv_files:
            print(f"🔍 Processing file: {file_path}")
            try:
                df = pd.read_csv(file_path, on_bad_lines="skip")
                df.columns = df.columns.str.strip()
                for i, row in df.iterrows():
                    row_text = " ".join(str(val) for val in row.values)
                    metadata = {
                        "source_file": os.path.basename(file_path),
                        "row_index": i,
                        "columns": df.columns.tolist(),
                    }
                    documents.append(Document(page_content=row_text, metadata=metadata))
                    ids.append(str(doc_id))
                    doc_id += 1
            except Exception as e:
                print(f"⚠️ Skipping {file_path}: {e}")

        vector_store.add_documents(documents=documents, ids=ids)
        vector_store.persist()
        print(f"✅ Indexed {len(documents)} rows from {len(csv_files)} files.")
    else:
        print("📂 Loading existing vector store...")

    # 6️⃣ Return a retriever (MMR + top-10)
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10}
    )

# Export for easy import elsewhere
retriever = get_retriever()
