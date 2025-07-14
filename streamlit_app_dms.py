import os
import streamlit as st
import vertexai
import json
import logging
from typing import List, Dict, Any, Optional

# --- IMPORTS FOR GOOGLE.GENAI (THE CORRECT WAY FOR RAG CORPUS) ---
from google import genai
from google.genai import types

# --- Authentication specific import ---
from google.oauth2 import service_account

# --- Configuration Variables ---
GCP_PROJECT_ID = "prj-auropro-dev"
GCP_REGION = "us-central1"
RAG_CORPUS_RESOURCE_NAME = "projects/694447741103/locations/us-central1/ragCorpora/6917529027641081856"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="DMS Chatbot", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="📄"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        text-align: center;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .source-info {
        background-color: #f5f5f5;
        padding: 0.5rem;
        border-radius: 5px;
        margin-top: 1rem;
        border-left: 3px solid #4caf50;
    }
    .error-message {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 5px;
        border-left: 3px solid #f44336;
    }
    .success-message {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 5px;
        border-left: 3px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>📄 DMS User Manual Chatbot</h1></div>', unsafe_allow_html=True)
st.markdown("**Ask questions about your DMS application, and I'll provide intelligent answers using the documentation.**")

class DMSChatbot:
    def __init__(self):
        self.genai_client = None
        self.tools_for_gemini = []
        self.model_name = "gemini-1.5-flash-001"  # Using stable model for reliability
        self.credentials = None
        self.auth_success = False
        
    def setup_authentication_from_secrets(self) -> bool:
        """Setup authentication automatically using Streamlit secrets"""
        try:
            if "gcp_service_account" in st.secrets:
                service_account_info = st.secrets["gcp_service_account"]
                self.credentials = service_account.Credentials.from_service_account_info(
                    dict(service_account_info),
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                st.sidebar.success("✅ Authenticated via Secrets")
                return True
            else:
                st.sidebar.error("❌ GCP service account not found in Streamlit secrets.")
                return False
        except Exception as e:
            st.sidebar.error(f"❌ Authentication from secrets failed: {str(e)}")
            return False
    
    def initialize_clients(self) -> bool:
        """Initialize Vertex AI and GenAI clients"""
        try:
            vertexai.init(
                project=GCP_PROJECT_ID, 
                location=GCP_REGION, 
                credentials=self.credentials
            )
            
            self.genai_client = genai.Client(
                vertexai=True,
                project=GCP_PROJECT_ID,
                location=GCP_REGION,
            )
            
            self.tools_for_gemini = [
                types.Tool(
                    retrieval=types.Retrieval(
                        vertex_rag_store=types.VertexRagStore(
                            rag_resources=[
                                types.VertexRagStoreRagResource(
                                    rag_corpus=RAG_CORPUS_RESOURCE_NAME,
                                )
                            ],
                            similarity_top_k=10,
                            vector_distance_threshold=0.5,
                        )
                    )
                )
            ]
            
            st.sidebar.success("✅ All clients initialized successfully")
            return True
            
        except Exception as e:
            st.sidebar.error(f"❌ Client initialization failed: {str(e)}")
            logger.error(f"Client initialization error: {str(e)}")
            return False
    
    def classify_query_type(self, user_query: str) -> str:
        query_lower = user_query.lower().strip()
        greeting_patterns = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'how are you', 'what\'s up', 'how\'s it going', 'nice to meet you', 'thanks', 'thank you', 'bye', 'goodbye', 'see you', 'good day']
        general_patterns = ['what is your name', 'who are you', 'what can you do', 'how can you help', 'tell me about yourself', 'what are your capabilities', 'how do you work', 'what\'s the weather', 'how\'s the weather', 'what time is it', 'tell me a joke', 'can you help me', 'what\'s new']
        dms_keywords = ['document', 'file', 'upload', 'download', 'create', 'edit', 'delete', 'user', 'role', 'permission', 'access', 'login', 'dashboard', 'workflow', 'approval', 'review', 'version', 'share', 'folder', 'search', 'filter', 'metadata', 'tag', 'category', 'archive', 'backup', 'restore', 'export', 'import', 'template', 'form', 'report', 'analytics', 'audit', 'security', 'compliance', 'notification', 'alert', 'settings', 'configuration', 'admin', 'dms', 'system', 'application', 'platform', 'software']
        if any(pattern in query_lower for pattern in greeting_patterns): return "greeting"
        if any(pattern in query_lower for pattern in general_patterns): return "general"
        if any(keyword in query_lower for keyword in dms_keywords): return "dms_specific"
        if len(query_lower.split()) < 3: return "general"
        return "dms_specific"
    
    def get_enhanced_persona_prompt(self, user_query: str, query_type: str) -> str:
        if query_type == "greeting": return f"You are a friendly and helpful DMS (Document Management System) assistant. The user is greeting you.\n\nRespond naturally and warmly to their greeting. Keep it brief and welcoming. You can mention that you're here to help with DMS-related questions if appropriate, but don't force it.\n\nUser said: \"{user_query}\"\n\nRespond in a conversational, friendly manner."
        elif query_type == "general": return f"You are a helpful DMS (Document Management System) assistant. The user is asking a general question that doesn't require searching the DMS documentation.\n\nRespond naturally and helpfully. If the question is about your capabilities, mention that you specialize in helping with DMS-related questions and can provide guidance on document management, user roles, workflows, and system features.\n\nUser asked: \"{user_query}\"\n\nProvide a helpful, conversational response using your general knowledge."
        else: return f"You are an expert DMS (Document Management System) assistant with deep knowledge of the system's functionality and user manual. \n\n**Core Instructions:**\n1. **Primary Focus**: Always prioritize information from the provided DMS documentation when answering DMS-related questions\n2. **Accuracy**: Provide precise, step-by-step instructions based on the documentation\n3. **Context Intelligence**: Use your understanding to connect related concepts and provide comprehensive answers\n4. **Completeness**: Provide thorough answers that address the user's needs\n\n**Response Guidelines:**\n- Base your response on the documentation but enhance with logical connections and explanations\n- Include relevant details about features, steps, and best practices\n- If referencing images/figures from documentation, describe their purpose and location\n- For unclear documentation, explain what you understand and what might need clarification\n- Connect related features and workflows when relevant\n\n**Current User Query Context:**\nThe user is asking: \"{user_query}\"\n\n**Response Style:**\n- Be conversational yet professional\n- Use clear, numbered steps for processes\n- Highlight important points with formatting\n- Provide context for why certain steps are necessary\n- Suggest related features or alternatives when appropriate\n\nIf the question cannot be answered from the documentation, clearly state this and explain what information would be needed."

    def process_query(self, user_query: str) -> tuple[str, List[Dict[str, Any]]]:
        try:
            query_type = self.classify_query_type(user_query)
            genai_contents = [types.Content(role="user", parts=[types.Part(text=self.get_enhanced_persona_prompt(user_query, query_type))])]
            
            # --- START OF API CALL FIX ---
            # This is the definitive fix. We use the old-style syntax that passes parameters
            # individually, which is compatible with the library version on Streamlit Cloud.
            if query_type in ["greeting", "general"]:
                response_stream = self.genai_client.models.generate_content_stream(
                    model_name=self.model_name,
                    contents=genai_contents,
                    # Old-style parameters
                    temperature=0.7,
                    top_p=0.9,
                    top_k=40,
                    max_output_tokens=2048,
                    candidate_count=1
                )
            else: # dms_specific
                response_stream = self.genai_client.models.generate_content_stream(
                    model_name=self.model_name,
                    contents=genai_contents,
                    # Old-style parameters
                    temperature=0.1,
                    top_p=0.9,
                    top_k=40,
                    max_output_tokens=8192,
                    candidate_count=1
                    # 'tools' parameter is correctly omitted as RAG is backend-enabled.
                )
            # --- END OF API CALL FIX ---
            
            response_text = ""
            retrieved_sources = []
            sources_set = set()
            
            for chunk in response_stream:
                if chunk.candidates and chunk.candidates[0].content:
                    if query_type == "dms_specific" and hasattr(chunk.candidates[0].content, 'grounding_metadata') and chunk.candidates[0].content.grounding_metadata:
                        metadata = chunk.candidates[0].content.grounding_metadata
                        if hasattr(metadata, 'retrieval_queries'):
                            for query in metadata.retrieval_queries:
                                if hasattr(query, 'retrieved_references'):
                                    for ref in query.retrieved_references:
                                        source_key = (getattr(ref, 'uri', 'N/A'), getattr(ref, 'title', 'N/A'), getattr(ref, 'page_number', 'N/A'))
                                        if source_key not in sources_set:
                                            sources_set.add(source_key)
                                            retrieved_sources.append({"uri": source_key[0], "title": source_key[1], "page_number": source_key[2], "relevance_score": getattr(ref, 'relevance_score', 0.0)})
                    if hasattr(chunk.candidates[0].content, 'parts'):
                        for part in chunk.candidates[0].content.parts:
                            if hasattr(part, 'text') and part.text: response_text += part.text
            
            if not response_text:
                if query_type in ["greeting", "general"]: response_text = "Hello! I'm here to help you with your DMS (Document Management System). What would you like to know?"
                else: response_text = "I apologize, but I couldn't generate a response for your query. Please try rephrasing your question or check if it's related to the DMS system."
            if retrieved_sources: retrieved_sources.sort(key=lambda x: x.get('relevance_score', 0.0), reverse=True)
            return response_text, retrieved_sources
        except Exception as e:
            logger.error(f"Query processing error: {str(e)}", exc_info=True)
            return f"I encountered an error while processing your query: {str(e)}. Please try again or rephrase your question.", []
    
    def display_sources(self, sources: List[Dict[str, Any]], query_type: str):
        if query_type == "dms_specific" and sources:
            st.markdown("---")
            st.markdown("### 🔍 **Sources Referenced:**")
            for i, source in enumerate(sources[:5]):
                title = source.get("title", "Unknown Document")
                uri = source.get("uri", "#")
                page = source.get("page_number", "N/A")
                relevance = source.get("relevance_score", 0.0)
                with st.expander(f"📄 **Source {i+1}:** {title}", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1: st.write(f"**Page:** {page}")
                    with col2:
                        if relevance > 0: st.write(f"**Relevance:** {relevance:.2f}")
                    if uri != "N/A" and uri != "#": st.markdown(f"**Link:** [View Document]({uri})")
        elif query_type == "dms_specific" and not sources:
            st.info("💡 This response was generated using general AI knowledge. For specific DMS procedures, try asking about particular features or processes.")

@st.cache_resource
def get_chatbot():
    return DMSChatbot()

chatbot = get_chatbot()

if not chatbot.auth_success:
    chatbot.auth_success = chatbot.setup_authentication_from_secrets()

if chatbot.auth_success and chatbot.genai_client is None:
    if not chatbot.initialize_clients():
        st.error("Failed to initialize the chatbot. Please check your configuration and secrets.")
        st.stop()

if chatbot.auth_success and chatbot.genai_client:
    if "messages" not in st.session_state:
        st.session_state.messages = []
        welcome_msg = "👋 **Welcome to the DMS Chatbot!** \n\nI'm here to help you with your Document Management System. You can ask me about:\n- How to create and manage documents\n- User roles and permissions\n- System navigation and features\n- Troubleshooting common issues\n- Best practices for document organization\n\nWhat would you like to know about the DMS?"
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask me anything about the DMS..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            query_type = chatbot.classify_query_type(prompt)
            spinner_text = "🤔 Thinking and searching the documentation..." if query_type == "dms_specific" else "💭 Thinking..."
            with st.spinner(spinner_text):
                response_text, sources = chatbot.process_query(prompt)
            st.markdown(response_text)
            chatbot.display_sources(sources, query_type)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 **Quick Actions**")
    quick_questions = ["How do I create a new document?", "What are the user roles in DMS?", "How do I search for documents?", "How do I share documents with others?", "What are the document approval workflows?"]
    for question in quick_questions:
        if st.sidebar.button(question, key=f"quick_{question}"):
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
else:
    st.error("❌ Authentication Failed. Please check your `.streamlit/secrets.toml` file.")
    st.markdown("Ensure your `secrets.toml` has a `[gcp_service_account]` section with the correct service account JSON values.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 **Configuration Info**")
st.sidebar.info(f"**Project:** {GCP_PROJECT_ID}")
st.sidebar.info(f"**Region:** {GCP_REGION}")
st.sidebar.info(f"**Model:** {chatbot.model_name}")

if "messages" in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 **Session Stats**")
    st.sidebar.metric("Messages", len(st.session_state.messages))
    st.sidebar.metric("User Queries", len([m for m in st.session_state.messages if m["role"] == "user"]))
