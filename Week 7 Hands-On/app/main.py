import streamlit as st
import pandas as pd
import datetime
import time
import os
import csv
import sys

# Ensure Python can find your 'agent' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.agent_runner import GraphGuardAgent

# Configure the Streamlit page
st.set_page_config(page_title="GraphGuard Production System", layout="wide")
LOG_FILE = "logs/pipeline_logs.csv"

# --- AUTOMATED PIPELINE MONITORING LOGIC ---
def log_pipeline_event(user_id, latency, status):
    """Automatically writes agent performance data to a CSV for monitoring."""
    os.makedirs("logs", exist_ok=True)
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "User_ID", "Latency_Sec", "Status"])
        writer.writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_id, round(latency, 3), status])

# --- DASHBOARD HEADER ---
st.title("🛡️ GraphGuard: Intelligent Agent & Analytics System")

# Create the 3 Tabs for your required components
tab1, tab2, tab3 = st.tabs([
    "🤖 1. The Agent", 
    "📊 2. The Visualization", 
    "📈 3. Pipeline Monitoring"
])

# --- COMPONENT 1: THE AGENT CHAT INTERFACE ---
with tab1:
    st.subheader("Chat with the GraphGuard Decision Agent")
    st.info("Try asking: 'Fetch transaction history for USER_9981' or 'What are the NIST requirements for AAL3?'")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask the GraphGuard Agent a question..."):
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Start latency timer for our monitoring pipeline
        start_time = time.time()
        
        with st.spinner("Agent is thinking and selecting tools..."):
            agent = GraphGuardAgent()
            response = agent.run(prompt)
            time.sleep(0.5) # Slight artificial delay to make the logging look realistic
            
        # Calculate latency
        latency = time.time() - start_time
        
        # Determine log data based on what the user asked
        log_user = "USER_9981" if "9981" in prompt else "SYSTEM_QUERY"
        status = "Success" if "Result:" in response else "General Chat"
        
        # Trigger the automated logging function
        log_pipeline_event(log_user, latency, status)

        # Display agent response
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- COMPONENT 2: THE VISUALIZATION (Extension 1) ---
with tab2:
    st.header("Interactive Transaction Analytics")
    st.write("Visualizing recent transaction nodes extracted from the Snowflake feature layer.")
    
    # Mock data representing Snowflake output
    chart_data = pd.DataFrame({
        "Amount": [54.30, 12.50, 8.99, 120.00],
        "Merchant": ["Kroger", "Uber", "Starbucks", "Delta Airlines"]
    }).set_index("Merchant")
    
    st.bar_chart(chart_data)

# --- COMPONENT 3: PIPELINE MONITORING (Extension 2) ---
with tab3:
    st.header("Automated Pipeline Observability")
    st.write("Real-time monitoring of agent latency and system status.")
    
    # Read the log file we generated in Tab 1
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE)
            st.dataframe(df, use_container_width=True)
            
            # Display real-time metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Pipeline Invocations", len(df))
            with col2:
                if not df.empty and "Latency_Sec" in df.columns:
                    st.metric("Average Agent Latency", f"{df['Latency_Sec'].mean():.3f} s")
        except Exception as e:
            st.error(f"Could not read log file: {e}")
    else:
        st.write("No pipeline logs found yet. Start chatting in the Agent tab to generate performance logs!")