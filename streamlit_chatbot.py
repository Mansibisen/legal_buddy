"""
Streamlit-based Chatbot UI for Legal AI System
Modern Python-native web interface
"""

import streamlit as st
import requests
import json
from datetime import datetime
from typing import Optional, Dict, Any
import time

# Page configuration
st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #2563eb;
        --secondary-color: #7c3aed;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
    }
    
    /* Remove default padding */
    .main {
        padding-top: 1rem;
    }
    
    /* Custom message styling */
    .message-container {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .user-message {
        background-color: #dbeafe;
        border-left: 4px solid #2563eb;
    }
    
    .assistant-message {
        background-color: #f3f4f6;
        border-left: 4px solid #7c3aed;
    }
    
    .citation {
        color: #2563eb;
        text-decoration: underline;
        cursor: pointer;
        font-weight: 500;
    }
    
    .metrics {
        background-color: #f0fdf4;
        border: 1px solid #86efac;
        border-radius: 0.5rem;
        padding: 0.75rem;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{datetime.now().timestamp()}"

if "settings" not in st.session_state:
    st.session_state.settings = {
        "show_citations": True,
        "show_metrics": True,
        "check_hallucinations": True,
        "auto_retry": True,
        "max_retries": 3,
    }

if "api_status" not in st.session_state:
    st.session_state.api_status = "unknown"


# Helper functions
def call_api(endpoint: str, method: str = "GET", data: Dict = None) -> Optional[Dict]:
    """Call FastAPI backend"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "POST":
            response = requests.post(url, json=data, timeout=60)
        else:
            response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API server. Make sure it's running at http://localhost:8000")
        return None
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return None


def check_api_status() -> bool:
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def format_answer(answer: str) -> str:
    """Format answer text"""
    return answer.replace("\n", "\n\n")


def display_citations(citations: list):
    """Display citations in expandable section"""
    if not citations or not st.session_state.settings["show_citations"]:
        return
    
    with st.expander("📚 Sources & Citations"):
        for i, citation in enumerate(citations, 1):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"**{i}. {citation.get('source_id', 'Unknown Source')}**")
            with col2:
                if citation.get('page'):
                    st.caption(f"Page {citation['page']}")
            with col3:
                confidence = citation.get('confidence', 0.8)
                st.caption(f"Confidence: {confidence:.0%}")


def display_metrics(metrics: dict):
    """Display performance metrics"""
    if not metrics or not st.session_state.settings["show_metrics"]:
        return
    
    with st.expander("⚡ Performance Metrics"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Attempts",
                metrics.get("total_attempts", 0),
                help="Number of retrieval attempts"
            )
        
        with col2:
            retrieval_ms = metrics.get("retrieval_time", 0) * 1000
            st.metric(
                "Retrieval Time",
                f"{retrieval_ms:.0f}ms",
                help="Time to retrieve documents"
            )
        
        with col3:
            grading_ms = metrics.get("grading_time", 0) * 1000
            st.metric(
                "Grading Time",
                f"{grading_ms:.0f}ms",
                help="Time to grade documents"
            )
        
        with col4:
            total_ms = metrics.get("total_time", 0) * 1000
            st.metric(
                "Total Time",
                f"{total_ms:.0f}ms",
                help="Total processing time"
            )


def display_sources(sources: list):
    """Display source documents"""
    if not sources:
        return
    
    with st.expander("📄 Source Documents"):
        for i, source in enumerate(sources[:3], 1):
            with st.container():
                st.markdown(f"**Document {i}:**")
                st.text_area(
                    f"Source {i}",
                    value=source[:500] + "..." if len(source) > 500 else source,
                    height=150,
                    disabled=True,
                    label_visibility="collapsed"
                )
                st.divider()


# Main header
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.image("https://via.placeholder.com/100x50?text=Legal+AI", use_column_width=True)
with col2:
    st.title("⚖️ Legal AI Assistant")
    st.markdown("Your intelligent legal document Q&A system")
with col3:
    # API Status indicator
    if st.button("🔄 Check Status"):
        st.session_state.api_status = "online" if check_api_status() else "offline"
    
    status_color = "🟢" if check_api_status() else "🔴"
    st.markdown(f"{status_color} **API Status**: {'Online' if check_api_status() else 'Offline'}")


st.divider()

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Settings & Control")
    
    # Session info
    st.markdown("### Session Info")
    st.caption(f"Session ID: `{st.session_state.session_id}`")
    st.caption(f"Messages: {len(st.session_state.messages)}")
    
    st.divider()
    
    # Display settings
    st.markdown("### Display Settings")
    st.session_state.settings["show_citations"] = st.checkbox(
        "Show Citations",
        value=st.session_state.settings["show_citations"],
        help="Display source citations and references"
    )
    st.session_state.settings["show_metrics"] = st.checkbox(
        "Show Performance Metrics",
        value=st.session_state.settings["show_metrics"],
        help="Display response time and attempt metrics"
    )
    
    st.divider()
    
    # Processing settings
    st.markdown("### Processing Settings")
    st.session_state.settings["check_hallucinations"] = st.checkbox(
        "Check for Hallucinations",
        value=st.session_state.settings["check_hallucinations"],
        help="Verify answers against source documents"
    )
    st.session_state.settings["auto_retry"] = st.checkbox(
        "Auto-Retry on Failure",
        value=st.session_state.settings["auto_retry"],
        help="Automatically retry with improved query if first attempt fails"
    )
    st.session_state.settings["max_retries"] = st.slider(
        "Max Retry Attempts",
        min_value=1,
        max_value=5,
        value=st.session_state.settings["max_retries"],
        help="Maximum number of retrieval attempts"
    )
    
    st.divider()
    
    # System info
    st.markdown("### System Information")
    if st.button("📋 Fetch System Info"):
        system_info = call_api("/status")
        if system_info:
            st.json(system_info)
    
    # Quick actions
    st.divider()
    st.markdown("### Quick Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🆕 New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = f"session_{datetime.now().timestamp()}"
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.messages = []
            st.success("Chat cleared!")
            st.rerun()
    
    # API info
    st.divider()
    st.markdown("### API Configuration")
    st.caption(f"**Base URL:** {API_BASE_URL}")
    if st.checkbox("Show API Docs"):
        st.markdown(f"🔗 [Swagger UI]({API_BASE_URL}/docs)")
        st.markdown(f"🔗 [ReDoc]({API_BASE_URL}/redoc)")


# Main chat area
st.markdown("### Chat History")

# Display chat messages
for message in st.session_state.messages:
    with st.container():
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Assistant:** {message['content']}")
            
            # Display citations if available
            if "citations" in message:
                display_citations(message["citations"])
            
            # Display metrics if available
            if "metrics" in message:
                display_metrics(message["metrics"])
            
            # Display sources if available
            if "sources" in message:
                display_sources(message["sources"])


st.divider()

# Input area
st.markdown("### Ask a Question")

# Quick prompt examples
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("📋 Employment Contract", use_container_width=True):
        st.session_state.quick_prompt = "What are the key clauses in an employment contract?"
        st.rerun()

with col2:
    if st.button("🛡️ Liability", use_container_width=True):
        st.session_state.quick_prompt = "Explain the concept of limitation of liability."
        st.rerun()

with col3:
    if st.button("🔐 Confidentiality", use_container_width=True):
        st.session_state.quick_prompt = "What should be included in a confidentiality agreement?"
        st.rerun()

with col4:
    if st.button("💡 IP Rights", use_container_width=True):
        st.session_state.quick_prompt = "Explain intellectual property rights in contracts."
        st.rerun()

# Message input
col1, col2 = st.columns([20, 1])

with col1:
    user_input = st.text_area(
        "Your question:",
        value=st.session_state.get("quick_prompt", ""),
        placeholder="Ask about legal documents, contracts, clauses, etc.",
        height=80,
        label_visibility="collapsed"
    )
    if "quick_prompt" in st.session_state:
        del st.session_state.quick_prompt

with col2:
    send_button = st.button("📤 Send", use_container_width=True, key="send_btn")

# Process message
if send_button and user_input:
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Show thinking indicator
    with st.spinner("🔍 Searching documents... 💭 Analyzing... ✍️ Generating response..."):
        # Call API
        response = call_api(
            "/chat/message",
            method="POST",
            data={
                "question": user_input,
                "session_id": st.session_state.session_id,
                "max_attempts": st.session_state.settings["max_retries"],
                "check_hallucinations": st.session_state.settings["check_hallucinations"]
            }
        )
    
    if response and response.get("success"):
        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.get("answer", "No answer generated"),
            "citations": response.get("citations", []),
            "metrics": response.get("metrics", {}),
            "sources": response.get("sources", [])
        })
        
        st.success("✅ Response generated successfully!")
        st.rerun()
    
    elif response:
        st.error(f"❌ Error: {response.get('error', 'Unknown error')}")
    else:
        st.error("❌ Failed to get response from API")


# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🔐 Legal AI Assistant v1.0")
with col2:
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with col3:
    st.caption("💡 Powered by LangChain & OpenAI")
