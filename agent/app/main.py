# app/main.py
import streamlit as st
import os, sys

# Ensure agent can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.agent_runner import GraphGuardAgent

st.set_page_config(page_title="GraphGuard Agent Dashboard", layout="wide")

st.title("🛡️ GraphGuard: Intelligent Agent Interface")

tab1, tab2 = st.tabs(["🤖 GraphGuard AI Agent", "📊 Pipeline Analytics"])

with tab1:
    st.subheader("Chat with the GraphGuard Decision Agent")
    st.info("I can query Snowflake, check NIST compliance, and generate auth challenges. Ask me something!")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("E.g., Generate a challenge for USER_9981, or What is AAL3?"):
        # Add user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Run Agent
        agent = GraphGuardAgent()
        response = agent.run(prompt)
        
        # Add assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

with tab2:
    st.header("System Extensions (Week 5 Legacy)")
    st.write("Analytics and legacy pipeline features run here.")
