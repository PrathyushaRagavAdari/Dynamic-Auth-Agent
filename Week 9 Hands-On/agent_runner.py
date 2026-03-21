import json
import logging
import os
from agent.tools import query_snowflake_transactions, check_nist_compliance, generate_dynamic_challenge

# Set up Developer Debug Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename='logs/debug.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - AGENT_RUNNER - %(message)s'
)

class GraphGuardAgent:
    def __init__(self):
        self.system_prompt = "You are GraphGuard, an AI agent for Bank Compliance and Identity Verification."
        logging.info("GraphGuardAgent initialized.")

    def run(self, user_prompt: str) -> dict:
        """
        Returns a dictionary containing the final response and the reasoning trace.
        """
        prompt_lower = user_prompt.lower()
        logging.info(f"Received user prompt: {user_prompt}")
        
        try:
            # Scenario 1: Auditor asks for a compliance check
            if "nist" in prompt_lower or "aal" in prompt_lower:
                logging.info("Routing to check_nist_compliance tool.")
                evidence = check_nist_compliance(user_prompt)
                return {
                    "reasoning": "Intent classified as 'Compliance Check'. Queried FAISS Vector DB for NIST SP 800-63B guidelines.",
                    "response": f"**Result:** {evidence}"
                }
                
            # Scenario 2: Auditor asks to look up a user
            elif "user_" in prompt_lower or "transactions" in prompt_lower:
                user_id = "USER_9981" if "9981" in prompt_lower else "UNKNOWN"
                logging.info(f"Routing to query_snowflake_transactions for {user_id}.")
                data = query_snowflake_transactions(user_id)
                return {
                    "reasoning": f"Intent classified as 'Data Retrieval'. Executed Snowflake SQL view extraction for {user_id}.",
                    "response": f"**Result:** {data}"
                }
                
            # Scenario 3: Complex Multi-step (Generate challenge for a user)
            elif "generate" in prompt_lower or "challenge" in prompt_lower:
                user_id = "USER_9981" 
                logging.info(f"Executing multi-step workflow for {user_id}.")
                txn_data = query_snowflake_transactions(user_id)
                challenge = generate_dynamic_challenge(txn_data)
                return {
                    "reasoning": "Intent classified as 'Challenge Generation'. \n1. Fetched Snowflake graph context. \n2. Passed context to LLM for dynamic question formulation.",
                    "response": f"**Result:** {challenge}"
                }
                
            else:
                logging.warning("Prompt did not match any specific tool intent.")
                return {
                    "reasoning": "No specific tool intent detected. Defaulting to system greeting.",
                    "response": "Hello! I am the GraphGuard Agent. Ask me to check NIST compliance, fetch user transactions, or generate a dynamic challenge."
                }
        except Exception as e:
            logging.error(f"Error during agent execution: {str(e)}")
            return {
                "reasoning": "System encountered an unexpected exception.",
                "response": "An error occurred while processing your request. Please check the debug logs."
            }