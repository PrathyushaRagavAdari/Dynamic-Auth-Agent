# Task 4: AI Agent Evaluation Report

To ensure the GraphGuard agent correctly reasons and routes requests, we designed three evaluation scenarios ranging from simple tool execution to complex multi-step reasoning.

## Scenario 1: Simple Database Query (Single Tool)
* **User Prompt:** "Fetch transaction history for USER_9981."
* **Expected Agent Behavior:** The agent should parse the ID, select `query_snowflake_transactions`, and return the JSON payload.
* **Actual Result:** Success. The agent correctly reasoned and routed the prompt, returning the Kroger transaction context.

## Scenario 2: Analytics/RAG Query (Single Tool)
* **User Prompt:** "What are the NIST requirements for AAL3?"
* **Expected Agent Behavior:** The agent should ignore Snowflake and route the query to `check_nist_compliance` to search the vector database.
* **Actual Result:** Success. The agent retrieved the specific hardware-authenticator requirement from NIST SP 800-63B without hallucinating transactional data.

## Scenario 3: Complex Multi-Step Workflow (Multi-Tool)
* **User Prompt:** "Generate a dynamic security challenge for USER_9981."
* **Expected Agent Behavior:** The agent must recognize a dependency. It needs to first call `query_snowflake_transactions` to get the context, and then pass that context into `generate_dynamic_challenge`.
* **Actual Result:** Success. The agent executed the multi-step workflow sequentially, resulting in a highly specific, context-aware question ("What was the exact total amount of your Kroger purchase?").

## Limitations & Future Work
While the agent successfully routes intents, the current prototype relies heavily on deterministic keyword parsing in the `agent_runner.py` for stability during the demo. In a fully scaled production environment, we will hand over the tool selection entirely to the LLM's native function-calling API, though this will require stricter output validation to prevent the agent from chaining tools infinitely (agent looping).
