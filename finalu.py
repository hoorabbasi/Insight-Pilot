import streamlit as st
from datetime import datetime

from test import (
    read_file,
    create_sql_database,
    AnalysisAgent,
    save_chat_to_file,
    load_chat_from_file,
    get_saved_chats,
    delete_saved_chat,
    generate_pdf_report
)

# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------
st.set_page_config(
    page_title="Business Data Chat",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Insight Pilot")

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
.stChatMessage {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
[data-testid="stChatMessageContent"] {
    background-color: transparent;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SESSION STATE INIT
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_name" not in st.session_state:
    st.session_state.chat_name = f"Chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Setup")

    secret_api_key = st.secrets.get("GOOGLE_API_KEY", "")
    user_api_key = st.text_input("Enter Gemini API Key (optional)", type="password")

    api_key = user_api_key if user_api_key else secret_api_key

    uploaded_file = st.file_uploader(
        "Upload your data file",
        type=["csv", "xlsx", "xls"]
    )

    st.divider()

    # Sidebar tools only after agent exists
    if st.session_state.get("agent") is not None:

        col1, col2 = st.columns(2)

        # ---------------- SAVE CHAT ----------------
        with col1:
            if st.button("💾 Save Chat", use_container_width=True):
                if st.session_state.messages:
                    chat_data = {
                        "chat_name": st.session_state.chat_name,
                        "messages": st.session_state.messages,
                        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    if save_chat_to_file(chat_data, f"{st.session_state.chat_name}.json"):
                        st.success("Chat saved!")
                else:
                    st.warning("No messages to save")

        # ---------------- CLEAR CHAT ----------------
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.agent.clear_history()
                st.rerun()

        st.divider()

        # ---------------- SAVED CHATS ----------------
        try:
            saved_chats = get_saved_chats()
            if saved_chats:
                st.subheader("📁 Saved Chats")

                for chat_file in saved_chats:
                    chat_name = chat_file.replace(".json", "")
                    c1, c2 = st.columns([3, 1])

                    with c1:
                        if st.button(f"📂 {chat_name}", key=f"load_{chat_file}"):
                            data = load_chat_from_file(chat_file)
                            if data:
                                st.session_state.messages = data["messages"]
                                st.session_state.chat_name = data["chat_name"]
                                st.rerun()

                    with c2:
                        if st.button("🗑️", key=f"del_{chat_file}"):
                            delete_saved_chat(chat_file)
                            st.rerun()

        except Exception:
            st.error("Failed to load saved chats")

# --------------------------------------------------
# GLOBAL WARNINGS (NON-BLOCKING)
# --------------------------------------------------
if not api_key:
    st.warning("⚠️ Please add a Gemini API key in Streamlit Secrets or enter one in the sidebar.")

# --------------------------------------------------
# LOAD DATA + CREATE AGENT
# --------------------------------------------------
if api_key and uploaded_file and "agent" not in st.session_state:
    with st.spinner("Loading your data..."):
        data, error = read_file(uploaded_file)

        if error:
            st.error(error)
        else:
            db, schema, error = create_sql_database(data)

            if error:
                st.error(error)
            else:
                st.session_state.agent = AnalysisAgent(db, api_key)
                st.session_state.data = data
                st.success("✅ Data loaded! Ask me anything about your data.")

                with st.expander("📊 Preview Your Data"):
                    st.dataframe(data.head(10))

# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            st.markdown(msg.get("analysis", ""))
            if msg.get("chart"):
                st.plotly_chart(msg["chart"], use_container_width=True)

# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------
if prompt := st.chat_input("Ask a question about your data..."):

    if "agent" not in st.session_state:
        st.warning("⚠️ Please upload a file and enter your API key first!")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.agent.analyze(prompt)

        st.session_state.messages.append({
            "role": "assistant",
            "question": prompt,
            "sql_results": result.get("sql_results", ""),
            "analysis": result.get("analysis", ""),
            "chart": result.get("chart")
        })

        st.markdown(result.get("analysis", ""))

        if result.get("chart"):
            st.plotly_chart(result["chart"], use_container_width=True)
