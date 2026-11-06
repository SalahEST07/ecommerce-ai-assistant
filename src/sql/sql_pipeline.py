
from pathlib import Path

from src.sql.sql_generator import generate_sql
from src.sql.sql_engine import execute_sql


def format_sql_answer(result):
    """
    Convert SQL query results into a readable text answer.
    """

    if not result["rows"]:
        return "No matching records were found."

    rows = result["rows"]

    # One row and one column: simple numerical or textual answer
    if len(rows) == 1 and len(rows[0]) == 1:
        value = next(iter(rows[0].values()))

        if isinstance(value, (int, float)):
            value = f"{value:,}"

        return f"The result is {value}."

    # Multiple rows: display each result as a readable list
    answer_lines = ["Here are the results:"]

    for index, row in enumerate(rows, start=1):
        formatted_values = []

        for column, value in row.items():
            if isinstance(value, (int, float)):
                value = f"{value:,}"

            formatted_values.append(f"{column}: {value}")

        answer_lines.append(
            f"{index}. " + " | ".join(formatted_values)
        )

    if result.get("truncated"):
        answer_lines.append(
            "Note: The results were limited to the maximum number of rows."
        )

    return "\n".join(answer_lines)


def answer_sql_question(
    question,
    tokenizer,
    llm_model,
    db_path,
    max_rows=100
):
    """
    Complete SQL assistant workflow:
    1. Generate SQL.
    2. Execute SQL using the read-only engine.
    3. Format the results.
    """

    # Step 1: Generate SQL
    generation = generate_sql(
        question,
        tokenizer,
        llm_model
    )

    if not generation["success"]:
        return {
            "success": False,
            "question": question,
            "sql": None,
            "answer": None,
            "error": generation["error"],
            "source": None
        }

    sql_query = generation["sql"]

    # Step 2: Execute SQL safely
    execution = execute_sql(
        sql_query,
        db_path,
        max_rows=max_rows
    )

    if not execution["success"]:
        return {
            "success": False,
            "question": question,
            "sql": sql_query,
            "answer": None,
            "error": execution["error"],
            "source": "olist.db"
        }

    # Step 3: Format results
    answer = format_sql_answer(execution)

    return {
        "success": True,
        "question": question,
        "sql": sql_query,
        "answer": answer,
        "error": None,
        "source": "Olist SQLite database",
        "row_count": execution["row_count"],
        "truncated": execution["truncated"]
    }
