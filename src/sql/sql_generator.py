
import re

DATABASE_SCHEMA = """
TABLE: orders
- order_id (TEXT)
- customer_id (TEXT)
- order_status (TEXT)
- order_purchase_timestamp (TEXT)
- order_approved_at (TEXT)
- order_delivered_carrier_date (TEXT)
- order_delivered_customer_date (TEXT)
- order_estimated_delivery_date (TEXT)

TABLE: customers
- customer_id (TEXT)
- customer_unique_id (TEXT)
- customer_zip_code_prefix (BIGINT)
- customer_city (TEXT)
- customer_state (TEXT)

TABLE: order_items
- order_id (TEXT)
- order_item_id (BIGINT)
- product_id (TEXT)
- seller_id (TEXT)
- shipping_limit_date (TEXT)
- price (FLOAT)
- freight_value (FLOAT)

TABLE: products
- product_id (TEXT)
- product_category_name (TEXT)
- product_name_lenght (FLOAT)
- product_description_lenght (FLOAT)
- product_photos_qty (FLOAT)
- product_weight_g (FLOAT)
- product_length_cm (FLOAT)
- product_height_cm (FLOAT)
- product_width_cm (FLOAT)

RELATIONSHIPS:
- orders.customer_id = customers.customer_id
- order_items.order_id = orders.order_id
- order_items.product_id = products.product_id
"""

def generate_sql(question, tokenizer, llm_model):

    system_prompt = f"""
You are an expert SQLite query generator.

Convert the user's question into ONE valid SQLite SELECT query.

DATABASE SCHEMA:
{DATABASE_SCHEMA}

EXAMPLES:

Question: Which five cities have the most customers?
SQL:
SELECT customer_city, COUNT(*) AS customer_count
FROM customers
GROUP BY customer_city
ORDER BY customer_count DESC
LIMIT 5;

Question: How many orders are in the database?
SQL:
SELECT COUNT(*) AS total_orders FROM orders;

Question: How many orders were delivered?
SQL:
SELECT COUNT(*) AS delivered_orders
FROM orders
WHERE order_status = 'delivered';

Question: Which five product categories have the most items sold?
SQL:
SELECT p.product_category_name, COUNT(*) AS items_sold
FROM order_items oi
JOIN products p ON oi.product_id = p.product_id
WHERE p.product_category_name IS NOT NULL
GROUP BY p.product_category_name
ORDER BY items_sold DESC
LIMIT 5;

Question: Which five cities have the most customers?
SQL:
SELECT customer_city, COUNT(*) AS customer_count
FROM customers
GROUP BY customer_city
ORDER BY customer_count DESC
LIMIT 5;

STRICT RULES:
1. Generate only SELECT queries.
2. Use only tables and columns listed in the schema.
3. Never generate INSERT, UPDATE, DELETE, DROP, ALTER, or CREATE.
4. Never invent table or column names.
5. Use JOIN when data from multiple tables is required.
6. When calculating results per category, city, status, or another group,
   use GROUP BY.
7. For order status questions, use the order_status column.
8. Interpret "most popular categories" as the categories with the
   highest number of items sold.
9. Use clear aliases for calculated columns.
10. Return only SQL, without explanations or Markdown.
11. If the question cannot be answered using this database, return:
SELECT 'Unable to answer using the available database' AS message;
12. For customer city questions, use customers.customer_city.
    Never use a column named city.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=180,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    generated_text = re.sub(
        r"```(?:sql)?",
        "",
        generated_text,
        flags=re.IGNORECASE
    ).strip()

    match = re.search(r"\bSELECT\b", generated_text, re.IGNORECASE)

    if not match:
        return {
            "success": False,
            "sql": None,
            "error": "The model did not generate a SELECT query."
        }

    sql_query = generated_text[match.start():].strip()

    return {
        "success": True,
        "sql": sql_query,
        "error": None
    }
