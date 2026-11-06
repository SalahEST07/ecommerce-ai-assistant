
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DB_PATH = PROJECT_ROOT / "data" / "processed" / "olist.db"
CHROMA_PATH = PROJECT_ROOT / "data" / "processed" / "chroma_db"

# Make project modules importable
sys.path.insert(0, str(PROJECT_ROOT))

from src.sql.sql_engine import execute_sql

# --------------------------------------------------
# MCP server
# --------------------------------------------------

mcp = FastMCP("ecommerce-assistant")

# --------------------------------------------------
# Load the policy vector database
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

policy_store = Chroma(
    persist_directory=str(CHROMA_PATH),
    collection_name="ecommerce_policies_small_chunks",
    embedding_function=embeddings
)

# --------------------------------------------------
# Tool 1: Search policy documents
# --------------------------------------------------

@mcp.tool()
def search_policies(query: str, k: int = 3) -> dict:
    """
    Search the company's simulated policy documents.
    Returns relevant passages and their source filenames.
    """

    if not query.strip():
        return {"success": False, "error": "Search query cannot be empty."}

    if not 1 <= k <= 5:
        return {"success": False, "error": "k must be between 1 and 5."}

    documents = policy_store.similarity_search(query, k=k)

    results = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "content": doc.page_content
        }
        for doc in documents
    ]

    return {
        "success": True,
        "result_count": len(results),
        "results": results
    }

# --------------------------------------------------
# Tool 2: Execute safe SQL queries
# --------------------------------------------------

@mcp.tool()
def run_sql(query: str) -> dict:
    """
    Execute a read-only SELECT query against the Olist SQLite database.
    Modification queries are rejected by the SQL safety layer.
    """

    return execute_sql(query, DB_PATH)

# --------------------------------------------------
# Tool 3: Explain database schema
# --------------------------------------------------

@mcp.tool()
def explain_schema() -> dict:
    """
    Explain the available database tables, columns, and relationships.
    """

    return {
        "tables": {
            "orders": [
                "order_id",
                "customer_id",
                "order_status",
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date"
            ],
            "customers": [
                "customer_id",
                "customer_unique_id",
                "customer_zip_code_prefix",
                "customer_city",
                "customer_state"
            ],
            "order_items": [
                "order_id",
                "order_item_id",
                "product_id",
                "seller_id",
                "shipping_limit_date",
                "price",
                "freight_value"
            ],
            "products": [
                "product_id",
                "product_category_name",
                "product_name_lenght",
                "product_description_lenght",
                "product_photos_qty",
                "product_weight_g",
                "product_length_cm",
                "product_height_cm",
                "product_width_cm"
            ]
        },
        "relationships": [
            "customers.customer_id -> orders.customer_id",
            "orders.order_id -> order_items.order_id",
            "products.product_id -> order_items.product_id"
        ]
    }

# --------------------------------------------------
# Start MCP server
# --------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
