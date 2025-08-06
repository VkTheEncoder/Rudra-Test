# vector.py

from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
import glob

# Constants
CSV_FOLDER = "./csv_data"
DB_LOCATION = "./chroma_learning_db"
COLLECTION_NAME = "learning_materials"

def get_retriever():
    # Get all CSV files
    csv_files = glob.glob(os.path.join(CSV_FOLDER, "*.csv"))
    if not csv_files:
        print("❌ No CSV files found in folder:", CSV_FOLDER)
        exit()

    # Load embedding model
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    # Determine if we need to create the vector DB
    is_first_time = not os.path.exists(DB_LOCATION)

    if is_first_time:
        print("📦 Creating new vector store and indexing CSV rows...")
        documents = []
        ids = []
        doc_id = 0

        for file_path in csv_files:
            print(f"🔍 Processing file: {file_path}")
            try:
                df = pd.read_csv(file_path, on_bad_lines='skip')
                df.columns = df.columns.str.strip()

                for i, row in df.iterrows():
                    try:
                        row_text = " ".join([str(val) for val in row.values])
                        metadata = {
                            "source_file": os.path.basename(file_path),
                            "row_index": i,
                            "columns": df.columns.tolist()
                        }

                        doc = Document(page_content=row_text, metadata=metadata)
                        documents.append(doc)
                        ids.append(str(doc_id))
                        doc_id += 1

                    except Exception as e:
                        print(f"⚠️ Skipping row {i} in {file_path}: {e}")

            except Exception as e:
                print(f"❌ Failed to read {file_path}: {e}")

        # Create and save vector store
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=DB_LOCATION,
            embedding_function=embeddings
        )
        vector_store.add_documents(documents=documents, ids=ids)
        print(f"✅ Indexed {len(documents)} documents from {len(csv_files)} files.")

    else:
        print("📂 Loading existing vector store...")
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=DB_LOCATION,
            embedding_function=embeddings
        )

    # ✅ Return retriever with MMR + k=10
    return vector_store.as_retriever(
        search_type="mmr",          # Better diversity and precision
        search_kwargs={"k": 10}     # More documents = richer context
    )

# Export retriever directly for import
retriever = get_retriever()
