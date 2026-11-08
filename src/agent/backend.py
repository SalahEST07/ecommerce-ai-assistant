from pathlib import Path
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.agent.assistant_graph import build_assistant_graph

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DB_PATH = PROJECT_ROOT / "data" / "processed" / "olist.db"
CHROMA_PATH = PROJECT_ROOT / "data" / "processed" / "chroma_db"

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MCP_URL = "http://127.0.0.1:8000/mcp"


def load_assistant():
    """Load all components and build the LangGraph assistant."""

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    if not CHROMA_PATH.exists():
        raise FileNotFoundError(f"Chroma database not found: {CHROMA_PATH}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_store = Chroma(
        persist_directory=str(CHROMA_PATH),
        collection_name="ecommerce_policies_small_chunks",
        embedding_function=embeddings
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    llm_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    )

    llm_model.to("cpu")
    llm_model.eval()

    assistant = build_assistant_graph(
        vector_store=vector_store,
        tokenizer=tokenizer,
        llm_model=llm_model,
        db_path=str(DB_PATH),
        mcp_url=MCP_URL
    )

    return assistant
