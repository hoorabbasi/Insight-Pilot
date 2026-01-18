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

    api_key = st.text_input("Enter Gemini API Key", type="password")

    uploaded_file = st.file_uploader(
        "Upload your data file",
        type=["csv", "xlsx", "xls"]
    )

    st.divider()

    # ✅ FIX 1: Safe agent check
    if st.session_state.get("agent") is not None:

        col1, col2 = st.columns(2)

        # ---------------- SAVE CHAT ----------------
        with col1:
            
            if st.button("💾 Save Chat", use_container_width=True):
                if len(st.session_state.messages) > 0:
                    chat_data = {
                        "chat_name": st.session_state.chat_name,
                        "messages": st.session_state.messages,
                        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    filename = f"{st.session_state.chat_name}.json"
                    if save_chat_to_file(chat_data, filename):
                        st.success("Chat saved!")
                    else:
                        st.error("Save failed")
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
        # ✅ FIX 2: Prevent sidebar crash
        try:
            saved_chats = get_saved_chats()
            if saved_chats:
                st.subheader("📁 Saved Chats")

                for chat_file in saved_chats:
                    chat_name = chat_file.replace(".json", "")

                    c1, c2 = st.columns([3, 1])

                    with c1:
                        if st.button(
                            f"📂 {chat_name}",
                            key=f"load_{chat_file}",
                            use_container_width=True
                        ):
                            loaded_data = load_chat_from_file(chat_file)
                            if loaded_data:
                                st.session_state.messages = loaded_data["messages"]
                                st.session_state.chat_name = loaded_data["chat_name"]
                                st.success(f"Loaded: {chat_name}")
                                st.rerun()

                    with c2:
                        if st.button("🗑️", key=f"del_{chat_file}"):
                            if delete_saved_chat(chat_file):
                                st.success("Deleted!")
                                st.rerun()

        except Exception:
            st.error("Failed to load saved chats")

        st.divider()

# --------------------------------------------------
# LOAD DATA + CREATE AGENT
# --------------------------------------------------
if api_key and uploaded_file:
    if "agent" not in st.session_state:
        with st.spinner("Loading your data..."):
            data, error = read_file(uploaded_file)

            if error:
                st.error(f"Error: {error}")
            else:
                db, schema, error = create_sql_database(data)

                if error:
                    st.error(f"Database error: {error}")
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

            if msg.get("sql_results"):
                pdf = generate_pdf_report(
                    msg.get("question", ""),
                    msg.get("sql_results", ""),
                    msg.get("analysis", "")
                )
                st.download_button(
                    label="📥 Download Report",
                    data=pdf,
                    file_name=f"report_{i}.pdf",
                    mime="application/pdf",
                    key=f"pdf_{i}"
                )

# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------
if prompt := st.chat_input("Ask a question about your data..."):

    if "agent" not in st.session_state:
        st.warning("⚠️ Please upload a file and enter your API key first!")
    else:
        # user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.agent.analyze(prompt)

            # ✅ FIX 3: SAVE FIRST
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

            if result.get("sql_results"):
                pdf = generate_pdf_report(
                    prompt,
                    result.get("sql_results", ""),
                    result.get("analysis", "")
                )
                st.download_button(
                    label="📥 Download Report",
                    data=pdf,
                    file_name=f"report_{len(st.session_state.messages)}.pdf",
                    mime="application/pdf",
                    key="pdf_new"
                )
