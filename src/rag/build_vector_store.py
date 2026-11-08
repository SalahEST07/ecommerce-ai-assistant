from pathlib import Path

import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

POLICY_DIR = PROJECT_ROOT / "data" / "policies"
CHROMA_DIR = PROJECT_ROOT / "data" / "processed" / "chroma_db"

COLLECTION_NAME = "ecommerce_policies_small_chunks"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_vector_store():
    if not POLICY_DIR.exists():
        raise FileNotFoundError(f"Policy directory not found: {POLICY_DIR}")

    policy_files = sorted(POLICY_DIR.glob("*.txt"))

    if not policy_files:
        raise FileNotFoundError("No policy documents found.")

    documents = []

    for path in policy_files:
        content = path.read_text(encoding="utf-8")

        documents.append(
            Document(
                page_content=content,
                metadata={"source": path.name}
            )
        )

        print(f"Loaded: {path.name}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=40
    )

    chunks = splitter.split_documents(documents)

    print(f"\nPolicy documents: {len(documents)}")
    print(f"Generated chunks: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # Replace only the small-chunk collection.
    # Other collections are preserved.
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Replaced existing collection: {COLLECTION_NAME}")
    except Exception:
        print(f"Creating collection: {COLLECTION_NAME}")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR)
    )

    count = vector_store._collection.count()

    print(f"\nVector store created successfully.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Stored chunks: {count}")


if __name__ == "__main__":
    build_vector_store()
