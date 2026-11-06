
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END

from src.agent.router import route_question
from src.rag.rag_pipeline import generate_customer_answer
from src.sql.sql_generator import generate_sql
from src.sql.sql_engine import execute_sql


class AssistantState(TypedDict):
    question: str
    route: str
    reason: str
    answer: str
    sources: list
    generated_sql: Optional[str]
    error: Optional[str]


def build_assistant_graph(vector_store, tokenizer, llm_model, db_path):

    # 1. Router node
    def router_node(state: AssistantState):

        result = route_question(state["question"])

        return {
            "route": result["route"],
            "reason": result["reason"]
        }

    # 2. Policy node
    def policy_node(state: AssistantState):

        result = generate_customer_answer(
            state["question"],
            vector_store,
            tokenizer,
            llm_model
        )

        return {
            "answer": result["answer"],
            "sources": result["sources"]
        }

    # 3. SQL node
    def sql_node(state: AssistantState):

        generated = generate_sql(
            state["question"],
            tokenizer,
            llm_model
        )

        if not generated["success"]:
            return {
                "answer": "I could not generate a valid SQL query.",
                "error": generated["error"],
                "sources": []
            }

        execution = execute_sql(
            generated["sql"],
            db_path
        )

        if not execution["success"]:
            return {
                "answer": "I could not retrieve the requested database information.",
                "generated_sql": generated["sql"],
                "error": execution["error"],
                "sources": ["Olist SQLite database"]
            }

        answer = "Here are the results:\n"

        for index, row in enumerate(execution["rows"], start=1):
            answer += f"{index}. " + " | ".join(
                f"{key}: {value:,}" if isinstance(value, (int, float))
                else f"{key}: {value}"
                for key, value in row.items()
            ) + "\n"

        return {
            "answer": answer.strip(),
            "generated_sql": generated["sql"],
            "sources": ["Olist SQLite database"],
            "error": None
        }

        # 4. Hybrid node
    def hybrid_node(state: AssistantState):

        question = state["question"]

        # Separate policy and database questions
        parts = question.split(" and ", 1)

        if len(parts) == 2:
            policy_question = parts[0].strip()
            sql_question = parts[1].strip()
        else:
            policy_question = question
            sql_question = question

        # Retrieve policy information
        policy_result = generate_customer_answer(
            policy_question,
            vector_store,
            tokenizer,
            llm_model
        )

        # Generate SQL from the database-specific question
        sql_generated = generate_sql(
            sql_question,
            tokenizer,
            llm_model
        )

        if not sql_generated["success"]:
            return {
                "answer": (
                    "Policy information:\n"
                    + policy_result["answer"]
                    + "\n\nDatabase query could not be generated."
                ),
                "sources": policy_result["sources"],
                "error": sql_generated["error"]
            }

        # Execute SQL safely
        sql_result = execute_sql(
            sql_generated["sql"],
            db_path
        )

        if not sql_result["success"]:
            return {
                "answer": (
                    "Policy information:\n"
                    + policy_result["answer"]
                    + "\n\nDatabase query failed: "
                    + sql_result["error"]
                ),
                "generated_sql": sql_generated["sql"],
                "sources": policy_result["sources"] + [
                    "Olist SQLite database"
                ],
                "error": sql_result["error"]
            }

        # Format database results
        database_answer = "\n".join(
            str(row) for row in sql_result["rows"]
        )

        combined_answer = (
            "POLICY INFORMATION:\n"
            + policy_result["answer"]
            + "\n\nDATABASE INFORMATION:\n"
            + database_answer
        )

        return {
            "answer": combined_answer,
            "generated_sql": sql_generated["sql"],
            "sources": list(set(
                policy_result["sources"]
                + ["Olist SQLite database"]
            )),
            "error": None
        }

    # 5. Unknown question node
    def unknown_node(state: AssistantState):

        return {
            "answer": (
                "I'm sorry, but I can currently help with "
                "e-commerce policies and information available "
                "in the Olist database."
            ),
            "sources": [],
            "error": None
        }

    # Build the graph
    graph = StateGraph(AssistantState)

    graph.add_node("router", router_node)
    graph.add_node("policy", policy_node)
    graph.add_node("sql", sql_node)
    graph.add_node("hybrid", hybrid_node)
    graph.add_node("unknown", unknown_node)

    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "POLICY": "policy",
            "SQL": "sql",
            "HYBRID": "hybrid",
            "UNKNOWN": "unknown"
        }
    )

    graph.add_edge("policy", END)
    graph.add_edge("sql", END)
    graph.add_edge("hybrid", END)
    graph.add_edge("unknown", END)

    return graph.compile()
