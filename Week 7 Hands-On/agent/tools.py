# agent/tools.py
import json

def query_snowflake_transactions(user_id: str) -> str:
    """Tool 1: Simulates fetching Snowflake data."""
    # In production, this imports from features/feature_pipeline.py
    if user_id == "USER_9981":
        data = [{"merchant": "Kroger", "amount": 54.30, "location": "Columbus, OH"}]
        return json.dumps(data)
    return json.dumps([{"error": "User not found or no recent transactions."}])

def check_nist_compliance(query: str) -> str:
    """Tool 2: Simulates RAG retrieval from NIST guidelines."""
    if "AAL2" in query.upper():
        return "NIST SP 800-63B: AAL2 requires two distinct authentication factors."
    elif "AAL3" in query.upper():
        return "NIST SP 800-63B: AAL3 requires a hardware-based cryptographic authenticator."
    return "NIST guidelines retrieved: Refer to Section 5 for general authenticator requirements."

def generate_dynamic_challenge(transaction_context: str) -> str:
    """Tool 3: Generates the actual auth challenge."""
    if "Kroger" in transaction_context:
        return "Challenge: You recently made a grocery purchase in Columbus, OH. What was the exact total amount?"
    return "Challenge: Please verify the location of your last transaction."
