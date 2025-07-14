import os
import streamlit as st
import vertexai
import json
import logging
from typing import List, Dict, Any

# --- START OF FIX: Correct and consolidated imports from the official Vertex AI SDK ---
from vertexai.generative_models import (
    GenerativeModel,
    Content,
    Part,
    Tool,
    Retrieval,
    VertexRagStore,
    VertexRagStoreRagResource,
    GenerationConfig,
    SafetySetting,
    HarmCategory,
    HarmBlockThreshold,
)
# --- END OF FIX ---

# --- Authentication specific import ---
from google.oauth2 import service_account
from google.cloud import storage

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

# Custom CSS for better UI (No changes here)
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
        # --- START OF FIX: Use model objects, not a client object ---
        self.model_rag = None # Model with RAG (Retrieval-Augmented Generation) enabled
        self.model_general = None # Model for general chat
        # --- END OF FIX ---
        
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
            required_keys = ["type", "project_id", "private_key_id", "private_key", "client_email", "client_id", "auth_uri", "token_uri", "auth_provider_x509_cert_url", "client_x509_cert_url", "universe_domain"]
            missing_keys = [key for key in required_keys if key not in service_account_info]
            if missing_keys:
                st.sidebar.error(f"❌ Missing required keys in secrets: {missing_keys}")
                return False
            
            service_account_dict = dict(service_account_info)
            self.credentials = service_account.Credentials.from_service_account_info(service_account_dict, scopes=['https://www.googleapis.com/auth/cloud-platform'])
            self.storage_client = storage.Client(credentials=self.credentials, project=self.credentials.project_id)
            st.sidebar.success("✅ Authentication successful using Streamlit secrets")
            return True
            
        except Exception as e:
            st.sidebar.error(f"❌ Authentication failed: {str(e)}")
            return False

    def initialize_clients(self) -> bool:
        """Initialize Vertex AI and set up GenerativeModel objects"""
        try:
            vertexai.init(project=GCP_PROJECT_ID, location=GCP_REGION, credentials=self.credentials)
            
            # --- START OF FIX: Create the model objects correctly ---
            # 1. Define the RAG tool using the correct classes
            rag_tool = Tool(
                retrieval=Retrieval(
                    vertex_rag_store=VertexRagStore(
                        rag_resources=[
                            VertexRagStoreRagResource(
                                rag_corpus=RAG_CORPUS_RESOURCE_NAME,
                            )
                        ],
                        similarity_top_k=10,
                        vector_distance_threshold=0.5,
                    )
                )
            )

            # 2. Create a model instance WITH the RAG tool attached
            self.model_rag = GenerativeModel(self.model_name, tools=[rag_tool])
            
            # 3. Create a model instance WITHOUT tools for general conversation
            self.model_general = GenerativeModel(self.model_name)
            # --- END OF FIX ---

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
        # No changes here, this logic is perfect
        if query_type == "greeting": return f"""You are a friendly and helpful DMS (Document Management System) assistant. The user is greeting you...\nUser said: "{user_query}"\nRespond in a conversational, friendly manner."""
        elif query_type == "general": return f"""You are a helpful DMS (Document Management System) assistant...If the question is about your capabilities, mention that you specialize in helping with DMS-related questions...\nUser asked: "{user_query}"\nProvide a helpful, conversational response using your general knowledge."""
        else: return f"""You are an expert DMS (Document Management System) assistant...\n**Core Instructions:**\n1. **Primary Focus**: Always prioritize information from the provided DMS documentation...\n**Current User Query Context:**\nThe user is asking: "{user_query}" """

    def process_query(self, user_query: str) -> tuple[str, List[Dict[str, Any]]]:
        """Process user query using the correct GenerativeModel object"""
        try:
            query_type = self.classify_query_type(user_query)
            
            # --- START OF FIX: Logic now uses the correct model objects and classes ---
            contents = [Content(role="user", parts=[Part.from_text(self.get_enhanced_persona_prompt(user_query, query_type))])]
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }

            if query_type in ["greeting", "general"]:
                model_to_use = self.model_general
                generation_config = GenerationConfig(temperature=0.7, top_p=0.9, top_k=40, max_output_tokens=2048)
            else: # dms_specific
                model_to_use = self.model_rag
                generation_config = GenerationConfig(temperature=0.1, top_p=0.9, top_k=40, max_output_tokens=8192)

            if model_to_use is None:
                return "Error: The AI model is not initialized. Please contact support.", []

            response_stream = model_to_use.generate_content(
                contents,
                generation_config=generation_config,
                safety_settings=safety_settings,
                stream=True
            )
            # --- END OF FIX ---

            response_text = ""
            retrieved_sources = []
            
            # --- START OF FIX: Switched to modern grounding_metadata access ---
            for chunk in response_stream:
                response_text += chunk.text
                if chunk.grounding_metadata:
                    for ref in chunk.grounding_metadata.retrieval_queries[0].retrieved_references:
                        retrieved_sources.append({
                            "uri": ref.resource.uri,
                            "title": ref.resource.title,
                            "relevance_score": ref.relevance_score
                        })
            # --- END OF FIX ---

            if not response_text:
                if query_type in ["greeting", "general"]:
                    response_text = "Hello! I'm here to help you with your DMS. What would you like to know?"
                else:
                    response_text = "I apologize, but I couldn't generate a response for your query. Please try again."
            
            # Remove duplicates and sort
            unique_sources = list({v['uri']:v for v in retrieved_sources}.values())
            unique_sources.sort(key=lambda x: x.get('relevance_score', 0.0), reverse=True)
            
            return response_text, unique_sources
            
        except Exception as e:
            logger.error(f"Query processing error: {str(e)}", exc_info=True)
            return f"I encountered an error while processing your query: {str(e)}. Please try again.", []
    
    def display_sources(self, sources: List[Dict[str, Any]], query_type: str):
        if query_type == "dms_specific" and sources:
            st.markdown("---")
            st.markdown("### 🔍 **Sources Referenced:**")
            for i, source in enumerate(sources[:5]):
                title = source.get("title", "Unknown Document")
                uri = source.get("uri", "#")
                relevance = source.get("relevance_score", 0.0)
                with st.expander(f"📄 **Source {i+1}:** {title}", expanded=False):
                    if relevance > 0:
                        st.write(f"**Relevance:** {relevance:.2f}")
                    if uri != "N/A" and uri != "#":
                        st.markdown(f"**Link:** [View Document]({uri})")
        elif query_type == "dms_specific" and not sources:
            st.info("💡 This response was generated using general AI knowledge.")

# Initialize the chatbot
@st.cache_resource
def get_chatbot():
    return DMSChatbot()

chatbot = get_chatbot()

# --- START OF FIX: Check for the new model object, not the old client ---
if not chatbot.auth_success:
    chatbot.auth_success = chatbot.setup_authentication()

if chatbot.auth_success and chatbot.model_rag is None:
    if chatbot.initialize_clients():
        st.sidebar.success("🚀 DMS Chatbot is ready!")
    else:
        st.error("Failed to initialize the chatbot. Please check your configuration.")
        st.stop()

if chatbot.auth_success and chatbot.model_rag:
# --- END OF FIX ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
        welcome_msg = """
        👋 **Welcome to the DMS Chatbot!** 
        I'm here to help you with your Document Management System. You can ask me about:
        - How to create and manage documents, user roles, system features...
        What would you like to know about the DMS?
        """
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
    quick_questions = ["How do I create a new document?", "What are the user roles in DMS?", "How do I search for documents?", "How do I share documents with others?", "What are the document approval workflows?"]
    for question in quick_questions:
        if st.sidebar.button(question, key=f"quick_{question}"):
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()
    
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
else:
    st.error("❌ Authentication failed. Please check your Streamlit secrets configuration.")
    st.markdown("""
    ### 🔧 **Setup Instructions:**
    1. **Ensure your secrets.toml file is properly configured with the GCP service account**...
    """)

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
