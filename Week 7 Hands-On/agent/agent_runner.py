import json
from agent.tools import query_snowflake_transactions, check_nist_compliance, generate_dynamic_challenge

class GraphGuardAgent:
    def __init__(self):
        self.system_prompt = "You are GraphGuard, an AI agent for Bank Compliance and Identity Verification."

    def run(self, user_prompt: str) -> str:
        prompt_lower = user_prompt.lower()
        
        # Scenario 1: Auditor asks for a compliance check
        if "nist" in prompt_lower or "aal" in prompt_lower:
            evidence = check_nist_compliance(user_prompt)
            return f"🛡️ **Agent Reasoning:** I used the `check_nist_compliance` tool.\n\n**Result:** {evidence}"
            
        # Scenario 2: Auditor asks to look up a user
        elif "user_" in prompt_lower or "transactions" in prompt_lower:
            user_id = "USER_9981" if "9981" in prompt_lower else "UNKNOWN"
            data = query_snowflake_transactions(user_id)
            return f"🔍 **Agent Reasoning:** I used the `query_snowflake_transactions` tool for {user_id}.\n\n**Result:** {data}"
            
        # Scenario 3: Complex Multi-step (Generate challenge for a user)
        elif "generate" in prompt_lower or "challenge" in prompt_lower:
            user_id = "USER_9981" # Defaulting for demo
            txn_data = query_snowflake_transactions(user_id)
            challenge = generate_dynamic_challenge(txn_data)
            return f"🧠 **Agent Reasoning:** I executed a multi-step workflow. First, I used `query_snowflake_transactions`. Then, I passed the data to `generate_dynamic_challenge`.\n\n**Result:** {challenge}"
            
        else:
            return "Hello! I am the GraphGuard Agent. Ask me to check NIST compliance, fetch user transactions, or generate a dynamic challenge."