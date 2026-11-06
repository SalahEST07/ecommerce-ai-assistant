
import sqlite3
from pathlib import Path


def execute_sql(query, db_path, max_rows=100):
    """
    Execute a read-only SELECT query against the SQLite database.
    Returns query results as a dictionary.
    """

    query = query.strip()

    # Basic query validation
    if not query:
        return {"success": False, "error": "Empty SQL query."}

    # Only allow SELECT statements
    if not query.upper().startswith("SELECT"):
        return {
            "success": False,
            "error": "Only SELECT queries are allowed."
        }

    # Reject multiple statements and SQL comments
    if ";" in query.rstrip(";"):
        return {
            "success": False,
            "error": "Multiple SQL statements are not allowed."
        }

    if "--" in query or "/*" in query or "*/" in query:
        return {
            "success": False,
            "error": "SQL comments are not allowed."
        }

    db_path = Path(db_path).resolve()

    if not db_path.exists():
        return {
            "success": False,
            "error": "Database file not found."
        }

    try:
        # Open database in read-only mode
        connection = sqlite3.connect(
            f"{db_path.as_uri()}?mode=ro",
            uri=True
        )

        # Additional protection through SQLite authorizer
        allowed_actions = {
            sqlite3.SQLITE_SELECT,
            sqlite3.SQLITE_READ,
            sqlite3.SQLITE_FUNCTION,
            sqlite3.SQLITE_RECURSIVE
        }

        def read_only_authorizer(action, arg1, arg2, database, trigger):
            if action in allowed_actions:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        connection.set_authorizer(read_only_authorizer)

        cursor = connection.cursor()
        cursor.execute(query)

        columns = [description[0] for description in cursor.description]

        rows = cursor.fetchmany(max_rows + 1)

        truncated = len(rows) > max_rows
        rows = rows[:max_rows]

        results = [
            dict(zip(columns, row))
            for row in rows
        ]

        connection.close()

        return {
            "success": True,
            "columns": columns,
            "rows": results,
            "row_count": len(results),
            "truncated": truncated,
            "error": None
        }

    except sqlite3.Error as error:
        return {
            "success": False,
            "error": str(error)
        }
