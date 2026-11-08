POLICY_KEYWORDS = [
    "policy", "policies",
    "return", "refund", "cancellation", "cancel",
    "before dispatch", "after dispatch", "dispatched",
    "shipping policy", "delivery timeframe",
    "standard delivery", "business days",
    "prepare an order", "preparation",
    "track orders", "track their orders", "track their order",
    "tracking",
    "package delayed", "package is delayed",
    "package was delayed", "package delay",
    "delayed delivery",
    "payment method", "payment methods",
    "payment failed", "payment fails",
    "charged twice", "duplicate charge",
    "order confirmed", "considered confirmed",
    "customer support", "support team",
    "provide to support", "contact support",
    "customer issue", "unresolved",
    "escalated", "escalation",
    "return shipping", "defective product",
    "original packaging", "order id"
]

SQL_KEYWORDS = [
    "how many", "count", "total", "number of",
    "top ", "which five", "five most",
    "most popular", "most customers",
    "highest average", "average",
    "revenue", "sales", "distinct",
    "items sold", "each year", "per year",
    "each order status", "each product category",
    "in 2017", "database", "excluding shipping"
]


def route_question(question):
    q = question.lower().strip()

    policy_match = any(
        keyword in q for keyword in POLICY_KEYWORDS
    )
    sql_match = any(
        keyword in q for keyword in SQL_KEYWORDS
    )

    if policy_match and sql_match and " and " in q:
        return {
            "route": "HYBRID",
            "reason": "The question requires both policy and database information."
        }

    database_status_terms = [
        "canceled", "cancelled", "delivered",
        "shipped", "processing", "invoiced",
        "unavailable", "approved"
    ]

    if (
        "how many orders" in q
        and any(term in q for term in database_status_terms)
        and "policy" not in q
    ):
        return {
            "route": "SQL",
            "reason": "The question requires counting database orders by status."
        }

    if policy_match:
        return {
            "route": "POLICY",
            "reason": "The question concerns e-commerce policies."
        }

    if sql_match:
        return {
            "route": "SQL",
            "reason": "The question requires structured database information."
        }

    return {
        "route": "UNKNOWN",
        "reason": "The question does not match a supported information source."
    }
