
POLICY_KEYWORDS = [
    "policy", "policies", "return", "refund",
    "cancel", "cancellation", "shipping",
    "delivery", "dispatch", "payment method",
    "payment failed", "charged twice",
    "support", "business days", "tracking",
    "eligibility"
]

SQL_KEYWORDS = [
    "how many", "count", "total", "number of",
    "top ", "most popular", "most customers",
    "average", "revenue", "sales", "database",
    "customers", "orders", "product categories",
    "items sold", "which city", "which cities",
    "order status", "canceled orders",
    "cancelled orders"
]


def route_question(question):

    question_lower = question.lower()

    policy_match = any(
        keyword in question_lower
        for keyword in POLICY_KEYWORDS
    )

    sql_match = any(
        keyword in question_lower
        for keyword in SQL_KEYWORDS
    )

    if policy_match and sql_match:
        route = "HYBRID"
        reason = "The question requires both policy and database information."

    elif policy_match:
        route = "POLICY"
        reason = "The question concerns company policies."

    elif sql_match:
        route = "SQL"
        reason = "The question requires structured database information."

    else:
        route = "UNKNOWN"
        reason = "The question does not clearly match a supported information source."

    return {
        "route": route,
        "reason": reason
    }
