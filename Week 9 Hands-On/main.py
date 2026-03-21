import streamlit as st
import pandas as pd
import datetime
import time
import os
import csv
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.agent_runner import GraphGuardAgent

st.set_page_config(page_title="GraphGuard Production System", layout="wide")
LOG_FILE = "logs/pipeline_logs.csv"

def log_pipeline_event(user_id, latency, status):
    os.makedirs("logs", exist_ok=True)
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "User_ID", "Latency_Sec", "Status"])
        writer.writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_id, round(latency, 3), status])

st.title("🛡️ GraphGuard (Veriscan): Production Dashboard")

tab1, tab2, tab3 = st.tabs(["🤖 1. The Agent", "📊 2. Visualization", "📈 3. System Evaluation"])

# --- CATEGORY 1: IMPROVED APP WORKFLOW & UX ---
with tab1:
    st.subheader("Interactive Agent Console")
    st.info("Try: 'Fetch transaction history for USER_9981' or 'Generate a challenge for USER_9981'")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "reasoning" in message:
                with st.expander("🔍 View Agent Reasoning Trace"):
                    st.caption(message["reasoning"])

    if prompt := st.chat_input("Ask the agent..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        start_time = time.time()
        with st.spinner("Agent orchestrating tools..."):
            agent = GraphGuardAgent()
            result_dict = agent.run(prompt)
            time.sleep(0.4) 
            
        latency = time.time() - start_time
        log_user = "USER_9981" if "9981" in prompt else "SYSTEM_QUERY"
        status = "Success" if "Result:" in result_dict["response"] else "General Chat"
        log_pipeline_event(log_user, latency, status)

        with st.chat_message("assistant"):
            st.markdown(result_dict["response"])
            with st.expander("🔍 View Agent Reasoning Trace"):
                st.caption(result_dict["reasoning"])
                
        st.session_state.messages.append({
            "role": "assistant", 
            "content": result_dict["response"],
            "reasoning": result_dict["reasoning"]
        })

# --- TAB 2: VISUALIZATION ---
with tab2:
    st.header("Behavioral Fingerprinting Nodes")
    chart_data = pd.DataFrame({
        "Amount": [54.30, 12.50, 8.99, 120.00],
        "Merchant": ["Kroger", "Uber", "Starbucks", "Delta Airlines"]
    }).set_index("Merchant")
    st.bar_chart(chart_data)

# --- CATEGORY 2: SYSTEM EVALUATION & MONITORING ---
with tab3:
    st.header("Real-Time Latency & Evaluation Monitoring")
    
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        
        # Display Key Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total API Invocations", len(df))
        with col2:
            avg_lat = df['Latency_Sec'].mean()
            st.metric("Avg Latency (Target < 2.0s)", f"{avg_lat:.3f} s", delta=f"{round(2.0 - avg_lat, 2)}s margin", delta_color="normal")
        with col3:
            success_rate = (len(df[df['Status'] == 'Success']) / len(df)) * 100 if len(df) > 0 else 0
            st.metric("Tool Success Rate", f"{success_rate:.1f}%")

        # Visual Latency Tracking
        st.subheader("Latency Trend Over Time")
        st.line_chart(df['Latency_Sec'])
        
        st.subheader("Raw Telemetry Logs")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No system logs found. Interact with the Agent to generate evaluation data.")