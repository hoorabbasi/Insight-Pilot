import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv

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

# Load environment variables
load_dotenv()

# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------
st.set_page_config(
    page_title="Insight Pilot - AI Business Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# ENHANCED CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Title Styling */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle */
    .subtitle {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        border-right: 1px solid #e2e8f0;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #1e293b;
        font-weight: 600;
        font-size: 1.3rem;
    }
    
    /* Chat Messages */
    .stChatMessage {
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .stChatMessage:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    [data-testid="stChatMessageContent"] {
        background-color: transparent;
    }
    
    /* User Message */
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Assistant Message */
    [data-testid="stChatMessage"][data-testid*="assistant"] {
        background: white;
        border: 1px solid #e2e8f0;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-1px);
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 8px;
        font-weight: 500;
        border: none;
        padding: 0.5rem 1rem;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: white;
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background: #f8fafc;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #cbd5e1;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Chat Input */
    .stChatInput > div > div {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .stChatInput > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f8fafc;
        border-radius: 8px;
        font-weight: 500;
        color: #334155;
    }
    
    /* Success/Warning/Error Messages */
    .stSuccess, .stWarning, .stError {
        border-radius: 8px;
        padding: 1rem;
        font-weight: 500;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #667eea !important;
    }
    
    /* Divider */
    hr {
        margin: 1.5rem 0;
        border: none;
        border-top: 1px solid #e2e8f0;
    }
    
    /* Custom Badge */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 0.5rem;
    }
    
    /* Feature Cards */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    
    /* Saved Chat Items */
    .chat-item {
        transition: all 0.2s ease;
    }
    
    .chat-item:hover {
        transform: translateX(4px);
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER SECTION
# --------------------------------------------------
st.title("🚀 Insight Pilot")
st.markdown('<p class="subtitle">Transform your business data into actionable insights with AI-powered analytics</p>', unsafe_allow_html=True)

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
    st.markdown("### ⚙️ Configuration")
    
    # API Key - Load silently from secrets (no UI element)
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    
    if not api_key:
        st.error("⚠️ API Key not configured. Please contact administrator.")
    
    # File Upload Section
    st.markdown("**📁 Data Upload**")
    uploaded_file = st.file_uploader(
        "Upload your business data",
        type=["csv", "xlsx", "xls"],
        help="Supported formats: CSV, Excel (XLSX, XLS)"
    )
    
    if uploaded_file:
        st.success(f"✅ {uploaded_file.name}")

    st.divider()

    # Chat Management Section
    if st.session_state.get("agent") is not None:
        st.markdown("### 💬 Chat Management")
        
        col1, col2 = st.columns(2)

        with col1:
            if st.button("💾 Save", use_container_width=True, help="Save current chat"):
                if len(st.session_state.messages) > 0:
                    chat_data = {
                        "chat_name": st.session_state.chat_name,
                        "messages": st.session_state.messages,
                        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    filename = f"{st.session_state.chat_name}.json"
                    if save_chat_to_file(chat_data, filename):
                        st.success("✅ Saved!")
                    else:
                        st.error("❌ Failed")
                else:
                    st.warning("⚠️ No messages")

        with col2:
            if st.button("🗑️ Clear", use_container_width=True, help="Clear chat history"):
                st.session_state.messages = []
                st.rerun()

        st.divider()

        # Saved Chats Section
        try:
            saved_chats = get_saved_chats()
            if saved_chats:
                st.markdown("### 📚 Saved Conversations")
                st.caption(f"{len(saved_chats)} saved chat(s)")
                
                for chat_file in saved_chats:
                    chat_name = chat_file.replace(".json", "")
                    
                    with st.container():
                        c1, c2 = st.columns([4, 1])
                        
                        with c1:
                            if st.button(
                                f"💬 {chat_name}",
                                key=f"load_{chat_file}",
                                use_container_width=True,
                                help=f"Load {chat_name}"
                            ):
                                loaded_data = load_chat_from_file(chat_file)
                                if loaded_data:
                                    st.session_state.messages = loaded_data["messages"]
                                    st.session_state.chat_name = loaded_data["chat_name"]
                                    st.success(f"✅ Loaded!")
                                    st.rerun()
                        
                        with c2:
                            if st.button("🗑️", key=f"del_{chat_file}", help="Delete chat"):
                                if delete_saved_chat(chat_file):
                                    st.success("✅")
                                    st.rerun()
            else:
                st.info("📭 No saved chats yet")
                
        except Exception:
            st.error("❌ Failed to load saved chats")

        st.divider()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748b; font-size: 0.85rem;'>
        <p>🚀 Powered by Gemini AI</p>
        <p style='font-size: 0.75rem;'>© 2024 Insight Pilot</p>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# MAIN CONTENT AREA
# --------------------------------------------------

    # Welcome Section (shown when no data loaded)
if "agent" not in st.session_state and not uploaded_file:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; 
                border-radius: 16px; 
                color: white; 
                margin-bottom: 2rem;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);'>
        <h2 style='color: white; margin: 0; font-size: 1.8rem;'>👋 Welcome to Insight Pilot!</h2>
        <p style='margin-top: 1rem; font-size: 1rem; opacity: 0.95;'>
            Your AI-powered business analytics assistant. Upload your data and start asking questions in natural language.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='feature-card'>
            <h3 style='color: #667eea; font-size: 1.2rem; margin-bottom: 0.5rem;'>📊 Smart Analysis</h3>
            <p style='color: #64748b; font-size: 0.9rem; margin: 0;'>
                Ask questions in plain English and get instant insights from your data
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-card'>
            <h3 style='color: #667eea; font-size: 1.2rem; margin-bottom: 0.5rem;'>📈 Auto Visualization</h3>
            <p style='color: #64748b; font-size: 0.9rem; margin: 0;'>
                Automatic chart generation for trends, comparisons, and distributions
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='feature-card'>
            <h3 style='color: #667eea; font-size: 1.2rem; margin-bottom: 0.5rem;'>📄 PDF Reports</h3>
            <p style='color: #64748b; font-size: 0.9rem; margin: 0;'>
                Download professional reports with insights and recommendations
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Getting Started
    with st.expander("🚀 Getting Started", expanded=True):
        st.markdown("""
        **Follow these simple steps:**
        
        1. **📁 Upload Data** - Upload your CSV or Excel file containing business data
        2. **💬 Start Chatting** - Ask questions like:
           - "What were the top 5 products by sales?"
           - "Show me monthly revenue trends"
           - "Which customers spent the most?"
        3. **📥 Download Reports** - Export your insights as PDF reports
        
        **Supported File Formats:** CSV, XLSX, XLS
        """)

# --------------------------------------------------
# LOAD DATA + CREATE AGENT
# --------------------------------------------------
if api_key and uploaded_file:
    if "agent" not in st.session_state:
        with st.spinner("🔄 Loading your data..."):
            data, error = read_file(uploaded_file)

            if error:
                st.error(f"❌ Error: {error}")
            else:
                db, schema, error = create_sql_database(data)

                if error:
                    st.error(f"❌ Database error: {error}")
                else:
                    st.session_state.agent = AnalysisAgent(db, api_key)
                    st.session_state.data = data
                    st.success("✅ Data loaded successfully! Ask me anything about your data.")

                    with st.expander("📊 Data Preview", expanded=False):
                        st.markdown(f"**Total Rows:** {len(data)} | **Columns:** {len(data.columns)}")
                        st.dataframe(data.head(10), use_container_width=True)

# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------
if st.session_state.messages:
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
                        file_name=f"insight_report_{i}.pdf",
                        mime="application/pdf",
                        key=f"pdf_{i}"
                    )
else:
    if st.session_state.get("agent") is not None:
        st.info("💡 Start by asking a question about your data below!")

# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------
if prompt := st.chat_input("💬 Ask a question about your data... (e.g., 'What are the top 5 products by revenue?')"):

    if "agent" not in st.session_state:
        st.warning("⚠️ Please upload a data file to get started!")
    else:
        # User message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing your data..."):
                result = st.session_state.agent.analyze(prompt)

            # Save message
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
                    file_name=f"insight_report_{len(st.session_state.messages)}.pdf",
                    mime="application/pdf",
                    key="pdf_new"
                )
