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
    # 1. find your CSVs
    csv_files = glob.glob(os.path.join(CSV_FOLDER, "*.csv"))
    if not csv_files:
        print("❌ No CSV files found in folder:", CSV_FOLDER)
        exit(1)

    # 2. embeddings
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    # 3. Chroma settings: DuckDB+Parquet for persistence
    client_settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=DB_LOCATION,
    )

    # 4. instantiate or load
    is_first_time = not os.path.exists(DB_LOCATION)
    vector_store = Chroma(
        client_settings=client_settings,
        collection_name=COLLECTION_NAME,
        persist_directory=DB_LOCATION,
        embedding_function=embeddings,
    )

    # 5. on first run, index all rows
    if is_first_time:
        print("📦 Creating new vector store and indexing CSV rows…")
        documents, ids = [], []
        counter = 0

        for path in csv_files:
            print(f"🔍 Processing {path}")
            try:
                df = pd.read_csv(path, on_bad_lines="skip")
                df.columns = df.columns.str.strip()
                for i, row in df.iterrows():
                    text = " ".join(str(v) for v in row.values)
                    meta = {
                        "source_file": os.path.basename(path),
                        "row_index": i,
                        "columns": df.columns.tolist(),
                    }
                    documents.append(Document(page_content=text, metadata=meta))
                    ids.append(str(counter))
                    counter += 1
            except Exception as e:
                print(f"⚠️ Skipped {path}: {e}")

        vector_store.add_documents(documents=documents, ids=ids)
        vector_store.persist()
        print(f"✅ Indexed {len(documents)} rows from {len(csv_files)} files.")
    else:
        print("📂 Loading existing vector store…")

    # 6. return retriever
    return vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 10})

# for easy import elsewhere
retriever = get_retriever()
