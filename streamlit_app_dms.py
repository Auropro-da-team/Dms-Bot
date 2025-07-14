import os
import streamlit as st
import vertexai
import json
import logging
from typing import List, Dict, Any, Optional
from google.cloud import storage
from google.oauth2 import service_account

# --- START OF FIX: Correct and Consolidated Imports ---
# REMOVED: from google import genai, from google.genai import types
# ADDED: The correct, modern SDK classes for Vertex AI
from vertexai.generative_models import (
    GenerativeModel,
    Content,
    Part,
    Tool,
    Retrieval,
    VertexRagStore,
    GenerationConfig,
    SafetySetting,
    HarmCategory,
    HarmBlockThreshold,
)
# --- END OF FIX ---

import google.auth

# Safely block credential fallback
def _disabled_creds(**kwargs):
    return None

google.auth._default._get_explicit_environ_credentials = _disabled_creds
google.auth._default._detect_gce = lambda: False

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
        # --- START OF FIX: Removed genai_client, as it's not used in the correct SDK pattern ---
        self.model = None # We will instantiate the model in process_query
        # --- END OF FIX ---
        self.tools_for_gemini = []
        self.model_name = "gemini-1.5-flash-001"
        self.credentials = None
        self.auth_success = False
        self.storage_client = None

        
    def setup_authentication(self) -> bool:
        """Setup authentication using Streamlit secrets"""
        try:
            st.sidebar.info("🔍 Checking authentication...")
            if "gcp_service_account" not in st.secrets:
                st.sidebar.error("❌ 'gcp_service_account' not found in Streamlit secrets")
                return False
            
            service_account_info = st.secrets["gcp_service_account"]
            available_keys = list(service_account_info.keys())
            st.sidebar.info(f"📋 Available keys in secrets: {available_keys}")
            
            required_keys = ["type", "project_id", "private_key_id", "private_key", "client_email", "client_id", "auth_uri", "token_uri", "auth_provider_x509_cert_url", "client_x509_cert_url", "universe_domain"]
            missing_keys = [key for key in required_keys if key not in service_account_info]
            if missing_keys:
                st.sidebar.error(f"❌ Missing required keys in secrets: {missing_keys}")
                return False
            
            service_account_dict = dict(service_account_info)
            self.credentials = service_account.Credentials.from_service_account_info(
                service_account_dict,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            self.storage_client = storage.Client(
                credentials=self.credentials, 
                project=self.credentials.project_id
            )
            st.sidebar.success("✅ Authentication successful using Streamlit secrets")
            return True
        except Exception as e:
            st.sidebar.error(f"❌ Authentication failed: {str(e)}")
            return False

    def initialize_clients(self) -> bool:
        """Initialize Vertex AI and define RAG tools."""
        try:
            vertexai.init(
                project=GCP_PROJECT_ID, 
                location=GCP_REGION, 
                credentials=self.credentials
            )
            
            # --- START OF FIX: Define tools using the correct SDK classes ---
            # This configures the RAG tool that will be used for DMS-specific queries.
            self.tools_for_gemini = [
                Tool(
                    retrieval=Retrieval(
                        vertex_rag_store=VertexRagStore(
                            rag_resources=[{"rag_corpus": RAG_CORPUS_RESOURCE_NAME}]
                        )
                    )
                )
            ]
            # --- END OF FIX ---

            st.sidebar.success("✅ Vertex AI initialized successfully")
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
        """Process user query with intelligent routing - RAG only when needed"""
        try:
            query_type = self.classify_query_type(user_query)
            
            # --- START OF FIX: Use correct Content and Part classes ---
            genai_contents = [
                Content(
                    role="user",
                    parts=[Part.from_text(self.get_enhanced_persona_prompt(user_query, query_type))]
                )
            ]
            # --- END OF FIX ---

            tools_to_use = None
            if query_type in ["greeting", "general"]:
                generation_config = GenerationConfig(temperature=0.7, top_p=0.9, top_k=40, max_output_tokens=2048)
            else:
                tools_to_use = self.tools_for_gemini
                generation_config = GenerationConfig(temperature=0.1, top_p=0.9, top_k=40, max_output_tokens=8192)

            # --- START OF FIX: Instantiate the model and generate content correctly ---
            self.model = GenerativeModel(
                model_name=self.model_name,
                tools=tools_to_use # This will be None for general chat, which is correct
            )
            
            response_stream = self.model.generate_content(
                contents=genai_contents,
                generation_config=generation_config,
                stream=True
            )
            # --- END OF FIX ---
            
            response_text = ""
            retrieved_sources = []
            sources_set = set()
            
            for chunk in response_stream:
                # --- START OF FIX: Correctly access grounding metadata from the Vertex AI SDK ---
                if query_type == "dms_specific" and chunk.grounding_metadata:
                    for ref in chunk.grounding_metadata.grounding_attributions:
                        source_key = (
                            getattr(ref.retrieved_context.resource, 'uri', 'N/A'),
                            getattr(ref.retrieved_context.resource, 'title', 'N/A')
                        )
                        if source_key not in sources_set:
                            sources_set.add(source_key)
                            retrieved_sources.append({
                                "uri": source_key[0],
                                "title": source_key[1],
                                "page_number": 'N/A', # Page number is not directly in this structure
                            })
                # --- END OF FIX ---

                if chunk.candidates and chunk.candidates[0].content.parts:
                    for part in chunk.candidates[0].content.parts:
                        if part.text:
                            response_text += part.text
            
            if not response_text:
                response_text = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
            
            if retrieved_sources:
                retrieved_sources.sort(key=lambda x: x.get('relevance_score', 0.0), reverse=True)
            
            return response_text, retrieved_sources
            
        except Exception as e:
            logger.error(f"Query processing error: {str(e)}", exc_info=True)
            error_msg = f"I encountered an error while processing your query: {str(e)}. Please try again or rephrase your question."
            return error_msg, []
    
    def display_sources(self, sources: List[Dict[str, Any]], query_type: str):
        if query_type == "dms_specific" and sources:
            st.markdown("---")
            st.markdown("### 🔍 **Sources Referenced:**")
            for i, source in enumerate(sources[:5]):
                title = source.get("title", "Unknown Document")
                uri = source.get("uri", "#")
                with st.expander(f"📄 **Source {i+1}:** {title}", expanded=False):
                    if uri != "N/A" and uri != "#":
                        st.markdown(f"**Link:** [View Document]({uri})")
        elif query_type == "dms_specific" and not sources:
            st.info("💡 This response was generated using general AI knowledge.")

# --- Main App Logic ---
@st.cache_resource
def get_chatbot():
    return DMSChatbot()

chatbot = get_chatbot()

if not chatbot.auth_success:
    chatbot.auth_success = chatbot.setup_authentication()

# --- START OF FIX: Check for initialization correctly ---
# We check if the tools list has been populated, which happens in initialize_clients.
if chatbot.auth_success and not chatbot.tools_for_gemini:
    if not chatbot.initialize_clients():
        st.error("Failed to initialize the chatbot. Please check your configuration.")
        st.stop()

if chatbot.auth_success and chatbot.tools_for_gemini:
# --- END OF FIX ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
        welcome_msg = "👋 **Welcome to the DMS Chatbot!** \n\nI'm here to help you with your Document Management System. What would you like to know?"
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask me anything about the DMS..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
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
    quick_questions = ["How do I create a new document?", "What are the user roles in DMS?", "How do I search for documents?"]
    for question in quick_questions:
        if st.sidebar.button(question, key=f"quick_{question}"):
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
else:
    st.error("❌ Authentication failed. Please check your Streamlit secrets configuration.")

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
