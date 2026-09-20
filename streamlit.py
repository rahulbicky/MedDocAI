import streamlit as st
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
import os
from datetime import datetime

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

# Page configuration
st.set_page_config(
    page_title="MedQuery AI - Your Intelligent Health Companion.",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Clean and Professional CSS
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #f5f7fa;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(120deg, #2c5f8d 0%, #1e3a5f 100%);
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 2px 8px rgba(44, 95, 141, 0.2);
    }
    
    .main-header h1 {
        font-size: 2.2em;
        font-weight: 600;
        margin: 0;
    }
    
    .main-header p {
        font-size: 1em;
        margin-top: 8px;
        opacity: 0.95;
    }
    
    /* User message bubble */
    .user-message {
        background-color: #2c5f8d;
        color: white;
        padding: 14px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 12px 0;
        margin-left: 15%;
        box-shadow: 0 2px 6px rgba(44, 95, 141, 0.25);
    }
    
    /* Bot message bubble */
    .bot-message {
        background-color: #ffffff;
        color: #2c3e50;
        padding: 14px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 12px 0;
        margin-right: 15%;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
        border-left: 3px solid #2c5f8d;
    }
    
    /* Timestamp styling */
    .timestamp {
        font-size: 0.7em;
        margin-top: 6px;
        opacity: 0.7;
    }
    
    /* Welcome message */
    .welcome-box {
        background: white;
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        margin: 20px 0;
        border-top: 4px solid #2c5f8d;
    }
    
    /* Chat input styling */
    .stChatInput>div>div>input {
        border-radius: 20px;
        border: 2px solid #2c5f8d;
        padding: 12px 18px;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background-color: #e0e4e8;
        margin: 20px 0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
    }
    
    /* Suggestion buttons */
    .stButton>button {
        background-color: white;
        color: #2c5f8d;
        border: 1px solid #2c5f8d;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 0.9em;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: #2c5f8d;
        color: white;
        border-color: #2c5f8d;
    }
    
    /* Info boxes */
    .info-card {
        background: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2c5f8d;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'suggested_clicked' not in st.session_state:
    st.session_state.suggested_clicked = False

# Initialize chatbot
@st.cache_resource
def initialize_chatbot():
    embeddings = download_hugging_face_embeddings()
    index_name = "medical-queryai"
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    chatModel = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=GROQ_API_KEY
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain

# Header
st.markdown("""
<div class="main-header">
    <h1>🩺 MedQuery AI</h1>
    <p>Your Intelligent Health Companion.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 💡 Suggested Questions")
    st.markdown("Click any question below to get started:")
    
    suggestions = [
        "What is Acromegaly?",
        "Symptoms of diabetes",
        "Treatment for hypertension",
        "What causes migraine?",
        "Prevention of heart disease",
        "Side effects of aspirin"
    ]
    
    for suggestion in suggestions:
        if st.button(suggestion, key=suggestion, use_container_width=True):
            st.session_state.suggested_clicked = suggestion
    
    st.divider()
    
    st.markdown("### ⚠️ Medical Disclaimer")
    st.markdown("""
    <div class="info-card">
        <small>This Medical Query AI provides information for educational purposes only. 
        Always consult qualified healthcare professionals for medical advice.</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### 🔒 Privacy")
    st.markdown("""
    <div class="info-card">
        <small>Your conversations are secure and not stored permanently.</small>
    </div>
    """, unsafe_allow_html=True)

# Initialize chatbot
try:
    rag_chain = initialize_chatbot()
    chatbot_ready = True
except Exception as e:
    st.error(f"⚠️ Error initializing chatbot: {str(e)}")
    chatbot_ready = False

# Display chat history
chat_container = st.container()

with chat_container:
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="welcome-box">
            <h3 style="color: #2c5f8d; margin-bottom: 10px;"> Welcome to MedQuery AI</h3>
            <p style="color: #5a6c7d; margin: 0;">Ask me anything about medical conditions, symptoms, treatments, or medications.</p>
            <p style="color: #7a8c9d; font-size: 0.9em; margin-top: 10px;"> Try the suggested questions in the sidebar or type your own question below.</p>
        </div>
        """, unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <strong>You</strong><br>
                {message["content"]}
                <div class="timestamp"> {message["timestamp"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bot-message">
                <strong>🤖 MedQuery AI</strong><br>
                {message["content"]}
                <div class="timestamp"> {message["timestamp"]}</div>
            </div>
            """, unsafe_allow_html=True)

# Handle suggested question click
if st.session_state.suggested_clicked:
    user_input = st.session_state.suggested_clicked
    st.session_state.suggested_clicked = False
else:
    user_input = st.chat_input("💬 Type your medical question here...")

# Process input
if chatbot_ready and user_input:
    timestamp = datetime.now().strftime("%H:%M")
    
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": timestamp
    })
    
    with st.spinner("🔍 Analyzing your question..."):
        try:
            response = rag_chain.invoke({"input": user_input})
            bot_response = response["answer"]
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": bot_response,
                "timestamp": timestamp
            })
        except Exception as e:
            st.error(f" Error: {str(e)}")
            bot_response = "I apologize, but I encountered an error. Please try rephrasing your question."
            st.session_state.messages.append({
                "role": "assistant",
                "content": bot_response,
                "timestamp": timestamp
            })
    
    st.rerun()
