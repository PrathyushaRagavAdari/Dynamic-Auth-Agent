# agent/tool_schemas.py

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_snowflake_transactions",
            "description": "Fetch recent transaction history for a specific user to build a context graph.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "The ID of the user (e.g., USER_9981)"}
                },
                "required": ["user_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_nist_compliance",
            "description": "Query the vector database for NIST SP 800-63B guidelines regarding AAL levels.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The compliance question (e.g., 'What is required for AAL2?')"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_dynamic_challenge",
            "description": "Generate a security question based on a user's transaction history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_context": {"type": "string", "description": "JSON string of user transactions"}
                },
                "required": ["transaction_context"]
            }
        }
    }
]
