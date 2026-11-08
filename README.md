# E-commerce AI Assistant

An AI-powered e-commerce assistant that combines Retrieval-Augmented Generation (RAG), SQL analytics, and tool orchestration to answer customer policy questions and analyze e-commerce data.

## 1. Project Overview

This project implements an intelligent assistant capable of answering questions about e-commerce policies and retrieving information from a structured database.

It uses a routing system to determine which workflow should handle each question:

- **POLICY:** Retrieves relevant policy passages using RAG.
- **SQL:** Generates and executes SQL queries against the Olist database.
- **HYBRID:** Combines policy retrieval and SQL analysis to answer questions requiring both.
- **UNKNOWN:** Handles questions outside the assistant's supported domain.

## 2. Architecture

The application follows this workflow:

1. The user submits a question through the Streamlit interface.
2. A keyword-based router classifies the question.
3. LangGraph selects the appropriate processing workflow.
4. The assistant calls MCP tools to access policy documents or database operations.
5. Qwen generates the response using the retrieved information.
6. The application displays the answer, detected route, sources, and generated SQL when applicable.

### Main components

| Component | Responsibility |
|---|---|
| Streamlit | Web interface |
| LangGraph | Workflow orchestration |
| LangChain | LLM and retrieval integration |
| ChromaDB | Policy vector database |
| Sentence Transformers | Text embeddings |
| Qwen2.5-0.5B-Instruct | Answer and SQL generation |
| MCP | Tool communication |
| SQLite | Structured e-commerce database |

### MCP Tools

The MCP server exposes three tools:

- `search_policies`: Searches the policy vector database.
- `run_sql`: Executes validated, read-only SQL queries.
- `explain_schema`: Provides database schema information.

## 3. Dataset

The project uses the **Olist Brazilian E-commerce Dataset**.

The SQLite database contains four main tables:

- `orders`
- `customers`
- `order_items`
- `products`

The database contains:

| Table | Number of rows |
|---|---:|
| Orders | 99,441 |
| Customers | 99,441 |
| Order items | 112,650 |
| Products | 32,951 |

The database relationships connect customers to orders, orders to order items, and products to order items.

## 4. Policy Knowledge Base

Five simulated policy documents were created for this project:

1. Shipping policy
2. Returns and refunds policy
3. Cancellation policy
4. Payment policy
5. Customer support policy

**Important:** These are fictional policies created for academic demonstration. They are not official Olist policies.

## 5. Evaluation Results

The assistant was evaluated using a 50-question benchmark:

- 20 policy questions
- 20 SQL questions
- 10 hybrid questions

### End-to-end evaluation

| Metric | Result |
|---|---:|
| Routing accuracy | 50/50 (100%) |
| Policy source retrieval | 20/20 |
| SQL execution success | 20/20 |
| SQL result correctness | 20/20 |
| Hybrid source retrieval | 10/10 |
| Hybrid SQL result correctness | 10/10 |

The SQL generator also achieved 20/20 correct results on the project-specific SQL benchmark after adding deterministic corrections for several known query patterns.

### RAG retrieval evaluation

Using the smaller chunk configuration:

| Metric | Score |
|---|---:|
| Recall@1 | 1.00 |
| Recall@3 | 1.00 |
| Recall@5 | 1.00 |
| MRR | 1.00 |

These retrieval metrics are based on document-level relevance labels.

**Evaluation limitations:** The benchmark is relatively small and tailored to this dataset and policy collection. The results should not be interpreted as proof of general-purpose accuracy. Initial human review of generated policy answers found 16 fully satisfactory answers and 4 partially complete answers out of 20.

## 6. Technologies

- Python
- Pandas
- SQLite
- PyTorch
- Hugging Face Transformers
- Qwen2.5-0.5B-Instruct
- Sentence Transformers
- LangChain
- LangGraph
- ChromaDB
- Model Context Protocol (MCP)
- Streamlit

## 7. Project Structure

```text
ecommerce-ai-assistant/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── policies/
│
├── notebooks/
│
├── src/
│   ├── agent/
│   │   ├── assistant_graph.py
│   │   ├── backend.py
│   │   └── router.py
│   │
│   ├── data/
│   ├── rag/
│   │   └── rag_pipeline.py
│   │
│   ├── sql/
│   │   ├── sql_engine.py
│   │   ├── sql_generator.py
│   │   └── sql_pipeline.py
│   │
│   └── mcp/
│       └── server.py
│
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

## 8. Installation

Clone the repository:

```bash
git clone https://github.com/SalahEST07/ecommerce-ai-assistant.git
cd ecommerce-ai-assistant
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 9. Data Preparation

The database and ChromaDB vector store are generated project assets and may not be included in the Git repository.

Before launching the application:

1. Obtain and prepare the Olist dataset.
2. Run the data preparation notebook or scripts to generate `data/processed/olist.db`.
3. Prepare the five policy documents in `data/policies/`.
4. Run the RAG indexing workflow to generate the ChromaDB collection `ecommerce_policies_small_chunks`.

The application expects these paths:

```text
data/processed/olist.db
data/processed/chroma_db/
data/policies/
```

## 10. Running the Application

The MCP server and Streamlit interface must run as separate processes.

**Terminal 1 — Start the MCP server:**

```bash
python src/mcp/server.py
```

The MCP server uses:

```text
http://127.0.0.1:8000/mcp
```

**Terminal 2 — Start Streamlit:**

```bash
streamlit run app/streamlit_app.py
```

Open the local URL displayed by Streamlit, usually:

```text
http://localhost:8501
```

The first application launch downloads the required Hugging Face models if they are not already cached.

## 11. Example Questions

**POLICY**

```text
How long does standard delivery take?
```

**SQL**

```text
How many orders are in the database?
```

**HYBRID**

```text
What is the cancellation policy, and how many orders have been canceled?
```

**UNKNOWN**

```text
Tell me a joke.
```

## 12. Security and Limitations

- SQL execution is restricted to read-only operations.
- SQL queries are validated before execution.
- Policy answers are grounded in retrieved project policy documents.
- The assistant uses a keyword-based router, which may not generalize to every possible question.
- The language model runs locally on CPU in the tested environment.
- The policy documents are simulated and must not be treated as real commercial policies.
- The application is an academic prototype, not a production-ready customer support system.

## 13. Author

Developed as an individual academic project for an Advanced AI Engineering course.

GitHub repository:  
https://github.com/SalahEST07/ecommerce-ai-assistant
