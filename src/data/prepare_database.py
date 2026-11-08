import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "olist"
DB_DIR = PROJECT_ROOT / "data" / "processed"
DB_PATH = DB_DIR / "olist.db"

TABLES = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
}


def prepare_database():
    DB_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {RAW_DIR}")

    with sqlite3.connect(DB_PATH) as connection:
        for table_name, filename in TABLES.items():
            csv_path = RAW_DIR / filename

            if not csv_path.exists():
                raise FileNotFoundError(f"Missing dataset file: {csv_path}")

            print(f"Loading {filename}...")

            dataframe = pd.read_csv(csv_path)

            dataframe.to_sql(
                table_name,
                connection,
                if_exists="replace",
                index=False
            )

            print(f"  {table_name}: {len(dataframe):,} rows")

        # Create indexes to improve common SQL queries.
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_orders_customer "
            "ON orders(customer_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_items_order "
            "ON order_items(order_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_items_product "
            "ON order_items(product_id)"
        )

        print("\nDatabase validation:")

        expected_counts = {
            "orders": 99441,
            "customers": 99441,
            "order_items": 112650,
            "products": 32951,
        }

        for table, expected in expected_counts.items():
            actual = connection.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]

            status = "OK" if actual == expected else "CHECK"

            print(
                f"{status}: {table}: {actual:,} rows "
                f"(expected {expected:,})"
            )

        integrity = connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        print(f"SQLite integrity: {integrity}")

    print(f"\nDatabase created: {DB_PATH}")


if __name__ == "__main__":
    prepare_database()
