import traceback

import json
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, START, END

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from src.agent.router import route_question
from src.sql.sql_generator import generate_sql


MCP_URL = "http://127.0.0.1:8000/mcp"


class AssistantState(TypedDict):
    question: str
    route: str
    reason: str
    answer: str
    sources: list
    generated_sql: Optional[str]
    error: Optional[str]


# --------------------------------------------------
# MCP CLIENT HELPER
# --------------------------------------------------

async def call_mcp_tool(tool_name, arguments, mcp_url=MCP_URL):

    async with streamable_http_client(mcp_url) as (
        read_stream, write_stream
    ): 

        async with ClientSession(read_stream, write_stream) as session:

            await session.initialize()

            result = await session.call_tool(
                tool_name,
                arguments=arguments
            )

            if result.isError:
                raise RuntimeError(
                    f"MCP tool {tool_name} returned an error."
                )

            for item in result.content:
                if hasattr(item, "text"):
                    return json.loads(item.text)

            raise RuntimeError("MCP tool returned no text result.")


# --------------------------------------------------
# POLICY ANSWER GENERATION
# --------------------------------------------------


def check_cancellation_policy(question, passages):

    question_lower = question.lower()

    if not any(word in question_lower for word in ["cancel", "cancellation"]):
        return None

    context = "\\n".join(
        item["content"].lower()
        for item in passages
    )

    if "cancellation" not in context and "cancel" not in context:
        return None

    answer = []

    if "before the order is dispatched" in context:
        answer.append(
            "Customers may request cancellation before the order is dispatched."
        )

    if "if approved" in context:
        answer.append(
            "Cancellation requires approval. If approved, the customer receives a full refund."
        )

    if "cannot be canceled directly" in context:
        answer.append(
            "Orders already dispatched cannot be canceled directly. "
            "Customers may request a return after delivery, subject to the returns policy."
        )

    if "provide their order id" in context:
        answer.append(
            "Customers must contact support and provide their order ID."
        )

    if "7 business days" in context:
        answer.append(
            "Approved cancellation refunds are processed within 7 business days, "
            "using the original payment method whenever possible."
        )

    return " ".join(answer) if answer else None


def generate_answer_from_mcp(question, passages, tokenizer, llm_model):

    # Use deterministic policy extraction for cancellation questions.
    deterministic_answer = check_cancellation_policy(question, passages)

    if deterministic_answer is not None:
        return deterministic_answer

    context = "\\n\\n".join(
        f"Source: {item['source']}\\n{item['content']}"
        for item in passages
    )

    system_prompt = """
You are an e-commerce customer support assistant.

Answer using ONLY the supplied policy passages.

Rules:
- Do not invent policy conditions.
- Preserve exact numbers and time periods.
- Distinguish calendar days from business days.
- Do not confuse cancellation refunds with return refunds.
- If the answer is not specified, say so clearly.
- Give a concise and direct answer.
"""

    user_prompt = f"""
POLICY PASSAGES:

{context}

CUSTOMER QUESTION:
{question}

Provide the answer based only on the passages above.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    inputs = {
        key: value.to(next(llm_model.parameters()).device)
        for key, value in inputs.items()
    }

    output = llm_model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_tokens = output[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return answer


def format_sql_results(execution):

    if not execution["rows"]:
        return "The query executed successfully, but no results were found."

    answer = "Here are the database results:\n"

    for index, row in enumerate(execution["rows"], start=1):

        formatted_row = " | ".join(
            f"{key}: {value:,}"
            if isinstance(value, (int, float))
            else f"{key}: {value}"
            for key, value in row.items()
        )

        answer += f"{index}. {formatted_row}\n"

    return answer.strip()


# --------------------------------------------------
# LANGGRAPH
# --------------------------------------------------

def build_assistant_graph(
    vector_store,
    tokenizer,
    llm_model,
    db_path,
    mcp_url=MCP_URL
):

    # vector_store and db_path are retained for compatibility
    # with the previous function call.
    # Policy and database access now happen through MCP.

    # 1. Router node
    def router_node(state: AssistantState):

        result = route_question(state["question"])

        return {
            "route": result["route"],
            "reason": result["reason"]
        }

    # 2. Policy node
    async def policy_node(state: AssistantState):

        try:

            result = await call_mcp_tool(
                "search_policies",
                {
                    "query": state["question"],
                    "k": 3
                },
                mcp_url
            )

            if not result["success"]:
                raise RuntimeError(result.get("error", "Policy search failed."))

            passages = result["results"]

            if not passages:
                return {
                    "answer": "I could not find relevant policy information.",
                    "sources": [],
                    "error": None
                }

            answer = generate_answer_from_mcp(
                state["question"],
                passages,
                tokenizer,
                llm_model
            )

            sources = list(set(
                item["source"] for item in passages
            ))

            return {
                "answer": answer,
                "sources": sources,
                "error": None
            }

        except Exception as error:

            return {
                "answer": "Policy retrieval failed.",
                "sources": [],
                "error": traceback.format_exc()
            }

    # 3. SQL node
    async def sql_node(state: AssistantState):

        try:

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

            execution = await call_mcp_tool(
                "run_sql",
                {"query": generated["sql"]},
                mcp_url
            )

            if not execution["success"]:
                return {
                    "answer": "I could not retrieve the requested database information.",
                    "generated_sql": generated["sql"],
                    "error": execution["error"],
                    "sources": ["Olist SQLite database"]
                }

            return {
                "answer": format_sql_results(execution),
                "generated_sql": generated["sql"],
                "sources": ["Olist SQLite database"],
                "error": None
            }

        except Exception as error:

            return {
                "answer": "Database retrieval failed.",
                "sources": [],
                "error": str(error)
            }

    # 4. Hybrid node
    async def hybrid_node(state: AssistantState):

        question = state["question"]

        # Separate policy and database questions
        parts = question.split(" and ", 1)

        if len(parts) == 2:
            policy_question = parts[0].strip()
            sql_question = parts[1].strip()
        else:
            policy_question = question
            sql_question = question

        try:

            # Retrieve policy information through MCP
            policy_result = await call_mcp_tool(
                "search_policies",
                {
                    "query": policy_question,
                    "k": 3
                },
                mcp_url
            )

            if not policy_result["success"]:
                raise RuntimeError("Policy search failed.")

            passages = policy_result["results"]

            if not passages:
                policy_answer = "No relevant policy passages found."
                policy_sources = []
            else:
                policy_answer = generate_answer_from_mcp(
                    policy_question,
                    passages,
                    tokenizer,
                    llm_model
                )

                policy_sources = list(set(
                    item["source"] for item in passages
                ))

            # Generate SQL
            generated = generate_sql(
                sql_question,
                tokenizer,
                llm_model
            )

            if not generated["success"]:
                return {
                    "answer": (
                        "POLICY INFORMATION:\n"
                        + policy_answer
                        + "\n\nDatabase query could not be generated."
                    ),
                    "sources": policy_sources,
                    "error": generated["error"]
                }

            # Execute SQL through MCP
            execution = await call_mcp_tool(
                "run_sql",
                {"query": generated["sql"]},
                mcp_url
            )

            if not execution["success"]:
                return {
                    "answer": (
                        "POLICY INFORMATION:\n"
                        + policy_answer
                        + "\n\nDatabase query failed: "
                        + execution["error"]
                    ),
                    "generated_sql": generated["sql"],
                    "sources": policy_sources + [
                        "Olist SQLite database"
                    ],
                    "error": execution["error"]
                }

            database_answer = format_sql_results(execution)

            combined_answer = (
                "POLICY INFORMATION:\n"
                + policy_answer
                + "\n\nDATABASE INFORMATION:\n"
                + database_answer
            )

            return {
                "answer": combined_answer,
                "generated_sql": generated["sql"],
                "sources": list(set(
                    policy_sources + ["Olist SQLite database"]
                )),
                "error": None
            }

        except Exception as error:

            return {
                "answer": "The hybrid request could not be completed.",
                "sources": [],
                "error": str(error)
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

    # Build graph
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
