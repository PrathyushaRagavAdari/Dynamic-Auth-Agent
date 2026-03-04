# Week 6 Individual Contribution Report
**Name:** Prathyusha Ragav Adari (Prathyu)  
**Project:** GraphGuard - Dynamic Identity Verification Agent  
**Role:** Product Intelligence, Strategy & Insight Lead  

## Individual Contributions & Responsibilities
For the Week 6 Agent Integration sprint, I led the architectural transition from a static application to an intelligent, multi-tool agent system. My specific contributions included:
1. **Tool Schema Engineering (`agent/tool_schemas.py` & `agent/tools.py`):** I designed the functional boundaries of our system. I extracted our Week 5 Snowflake logic and RAG retrieval logic and wrapped them into discrete, callable Python tools. I also authored the strict JSON schemas required for the agent to understand what inputs each tool requires.
2. **Agent Runner Implementation (`agent/agent_runner.py`):** I engineered the central routing logic that interprets user prompts, selects the appropriate tool, and orchestrates multi-step workflows (e.g., fetching Snowflake data *before* generating a challenge).
3. **Application Interface (`app/main.py`):** I integrated the agent logic into our Streamlit dashboard, replacing standard inputs with a dynamic `st.chat_message` interface, allowing auditors to interact with the system conversationally.
4. **Evaluation Scenarios:** I designed the three progressively complex testing scenarios documented in `task4_evaluation_report.md` to validate the agent's reasoning capabilities.

##Implemented Components
* [Commit: Created agent tool schemas and Python tool functions]
* [Commit: Engineered agent_runner.py multi-step workflow logic]
* [Commit: Updated Streamlit UI with interactive chat interface]

## Reflection on Work and Learning
Integrating an agent layer using Antigravity IDE fundamentally shifted my perspective on software architecture. Previously, I was thinking linearly (Data -> RAG -> Output). This week, I learned how to design a system for *autonomy*. 

The hardest challenge was managing multi-step reasoning—teaching the system that it cannot generate an authentication challenge until it successfully queries the database tool first. Wrapping our existing functions into strict schema definitions taught me the importance of clear API design; if a tool's description is vague, the agent will hallucinate or select the wrong tool. This sprint successfully pushed our capstone past the 60% mark, evolving GraphGuard from a basic pipeline into a true intelligent decision-support application.
