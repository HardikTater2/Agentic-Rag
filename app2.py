import streamlit as st
from datetime import datetime

# Import your workflow function from your core logic
from core import workflow

st.set_page_config(page_title="College Multi-Agent Assistant", layout="wide")
st.title("🎓 College Multi-Agent Assistant")
st.markdown("Ask your question or describe your issue. The assistant will intelligently route your query!")

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of dicts: {"role": "user"/"agent", "content": str, "timestamp": str}
if "current_agent" not in st.session_state:
    st.session_state.current_agent = ""
if "metadata" not in st.session_state:
    st.session_state.metadata = {}

# --- Chat Display ---
def render_chat():
    chat_css = """
    <style>
    .chat-container { width: 100%; }
    .chat-bubble {
        max-width: 70%;
        padding: 0.7em 1.1em;
        margin-bottom: 0.5em;
        border-radius: 1.3em;
        font-size: 1.1em;
        display: inline-block;
        word-break: break-word;
    }
    .chat-user {
        background: #d1e7dd;
        color: #1b4332;
        margin-left: 30%;
        margin-right: 0;
        text-align: right;
        float: right;
        clear: both;
    }
    .chat-agent {
        background: #f8f9fa;
        color: #212529;
        margin-right: 30%;
        margin-left: 0;
        text-align: left;
        float: left;
        clear: both;
    }
    .avatar {
        font-size: 1.5em;
        vertical-align: middle;
        margin-right: 0.5em;
    }
    .right-avatar {
        margin-left: 0.5em;
        margin-right: 0;
    }
    </style>
    """
    st.markdown(chat_css, unsafe_allow_html=True)
    for idx, msg in enumerate(st.session_state.chat_history):
        is_user = msg["role"] == "user"
        avatar = "🧑" if is_user else "🤖"
        bubble_class = "chat-user" if is_user else "chat-agent"
        # Avatar on right for user, left for agent
        if is_user:
            html = f"""
            <div class="chat-container">
                <div class="chat-bubble {bubble_class}">
                    <span>{msg["content"]}</span>
                    <span class="avatar right-avatar">{avatar}</span>
                </div>
            </div>
            """
        else:
            html = f"""
            <div class="chat-container">
                <div class="chat-bubble {bubble_class}">
                    <span class="avatar">{avatar}</span>
                    <span>{msg["content"]}</span>
                </div>
            </div>
            """
        st.markdown(html, unsafe_allow_html=True)

render_chat()

# --- User Input ---
with st.form(key="user_input_form", clear_on_submit=True):
    user_input = st.text_input(
        "Type your message:",
        key=f"user_input_{len(st.session_state.chat_history)}"
    )
    submitted = st.form_submit_button("Send")

# --- Helper: Extract text from agent output ---
def extract_agent_output_text(agent_output):
    # If agent_output is a LangChain AIMessage or similar
    if hasattr(agent_output, "content"):
        return agent_output.content
    elif callable(getattr(agent_output, "text", None)):
        return agent_output.text()
    elif isinstance(agent_output, dict) and "content" in agent_output:
        return agent_output["content"]
    else:
        return str(agent_output)

# --- Process New Input ---
if submitted and user_input.strip():
    # Avoid duplicate processing
    last_msg = st.session_state.chat_history[-1]["content"] if st.session_state.chat_history else None
    if user_input.strip() != last_msg:
        # Append user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input.strip(),
            "timestamp": datetime.now().isoformat()
        })

        # Prepare agent state for workflow
        agent_state = {
            "input": user_input.strip(),
            "sender": "user",
            "chat_history": st.session_state.chat_history.copy(),
            "current_agent": st.session_state.current_agent,
            "agent_output": "",
            "metadata": st.session_state.metadata.copy()
        }

        # Call core workflow
        result = workflow(agent_state)

        # Extract agent output text robustly
        agent_output = result.get("agent_output", "")
        agent_output_text = extract_agent_output_text(agent_output)
        st.session_state.metadata = result.get("metadata", {})

        st.session_state.chat_history.append({
            "role": "agent",
            "content": agent_output_text,
            "timestamp": datetime.now().isoformat()
        })

        # Optionally update current_agent if needed
        st.session_state.current_agent = agent_state.get("current_agent", "")

        # Rerun to display the new message
        st.rerun()

# --- Optional: Reset/Restart Button ---
if st.button("🔄 Start New Conversation"):
    st.session_state.chat_history = []
    st.session_state.current_agent = ""
    st.session_state.metadata = {}
    st.rerun()
