
import re


def check_return_eligibility(question):

    question_lower = question.lower()

    if "return" not in question_lower:
        return None

    match = re.search(
        r"(\d+)\s+days?\s+after\s+"
        r"(delivery|placing my order|placing the order|ordering|the order|order)",
        question_lower
    )

    if not match:
        return None

    days = int(match.group(1))
    reference = match.group(2)

    if reference == "delivery":

        if days <= 30:
            answer = (
                f"Yes. {days} days after delivery is within the "
                "30-calendar-day return period. The product must "
                "be unused and include its original packaging and "
                "accessories, except in cases involving defective "
                "or damaged products."
            )
        else:
            answer = (
                f"No. {days} days after delivery is outside the "
                "30-calendar-day return period."
            )

    else:
        answer = (
            "The return period is 30 calendar days after delivery, "
            "not after placing the order. I need to know the "
            "delivery date to determine whether the product "
            "is still eligible for return."
        )

    return {
        "answer": answer,
        "sources": ["returns_refunds_policy.txt"],
        "method": "deterministic_policy_check"
    }


def generate_rag_answer(question, vector_store, tokenizer,
                        llm_model, k=3):

    retrieved_docs = vector_store.similarity_search(
        question,
        k=k
    )

    context_parts = []

    for i, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata["source"]

        context_parts.append(
            f"Passage {i}\n"
            f"Source: {source}\n"
            f"Content: {doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": """
You are an e-commerce customer support assistant.

Answer using ONLY the supplied policy passages.

Rules:
- Never invent policies or conditions.
- Preserve numbers and time periods exactly.
- Pay attention to reference points such as delivery,
  dispatch, and payment confirmation.
- Include relevant conditions.
- If information is missing, say that the available
  policy documents do not specify it.
- Be concise and professional.
"""
        },
        {
            "role": "user",
            "content": f"""
POLICY PASSAGES:

{context}

CUSTOMER QUESTION:
{question}

Answer using only the supplied passages.
"""
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    import torch

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    sources = list(dict.fromkeys(
        doc.metadata["source"] for doc in retrieved_docs
    ))

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "method": "rag_generation",
        "retrieved_documents": retrieved_docs
    }


def generate_customer_answer(question, vector_store,
                             tokenizer, llm_model):

    rule_result = check_return_eligibility(question)

    if rule_result is not None:
        return {
            "question": question,
            **rule_result
        }

    return generate_rag_answer(
        question,
        vector_store,
        tokenizer,
        llm_model
    )



def check_cancellation_eligibility(question):
    question_lower = question.lower()

    cancellation_words = ["cancel", "cancellation"]
    dispatch_words = ["dispatch", "dispatched", "shipment", "shipped"]

    if not any(word in question_lower for word in cancellation_words):
        return None

    if not any(word in question_lower for word in dispatch_words):
        return None

    after_dispatch_phrases = [
        "after dispatch",
        "after being dispatched",
        "after it has been dispatched",
        "already been dispatched",
        "already dispatched",
        "after shipment",
        "after being shipped",
        "already shipped",
        "already been shipped",
        "has been shipped",
        "has already been shipped"
    ]

    if any(phrase in question_lower for phrase in after_dispatch_phrases):
        return {
            "answer": (
                "No. Orders that have already been dispatched cannot be "
                "canceled directly. Customers may request a return after "
                "delivery, subject to the returns and refunds policy."
            ),
            "sources": ["cancellation_policy.txt"],
            "method": "deterministic_cancellation_check"
        }

    before_dispatch_phrases = [
        "before dispatch",
        "before being dispatched",
        "before it is dispatched",
        "before shipment",
        "before being shipped"
    ]

    if any(phrase in question_lower for phrase in before_dispatch_phrases):
        return {
            "answer": (
                "Yes. Customers may request cancellation before dispatch. "
                "The cancellation is successful only after support confirms it. "
                "If approved, a full refund is provided."
            ),
            "sources": ["cancellation_policy.txt"],
            "method": "deterministic_cancellation_check"
        }

    return None


def generate_customer_answer(question, vector_store,
                             tokenizer, llm_model):

    # Check cancellation rules first
    cancellation_result = check_cancellation_eligibility(question)

    if cancellation_result is not None:
        return {
            "question": question,
            **cancellation_result
        }

    # Check return rules
    return_result = check_return_eligibility(question)

    if return_result is not None:
        return {
            "question": question,
            **return_result
        }

    # Otherwise, use RAG
    return generate_rag_answer(
        question,
        vector_store,
        tokenizer,
        llm_model
    )
