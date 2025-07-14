import os
import streamlit as st
import vertexai
import json
import logging
from typing import List, Dict, Any

# --- CORRECTED IMPORTS: Use ONLY the Vertex AI (google-cloud-aiplatform) SDK ---
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
    HarmBlockThreshold
)

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
        # --- CORRECTED: Use a model object from the Vertex AI SDK ---
        self.model = None
        self.rag_tool = None
        self.model_name = "gemini-1.5-flash-001"
        self.credentials = None
        self.auth_success = False
        self.storage_client = None

    def setup_authentication(self) -> bool:
        """Setup authentication using Streamlit secrets"""
        try:
            st.sidebar.info("🔍 Checking authentication...")
            if "gcp_service_account" not in st.secrets:
                st.sidebar.error("❌ 'gcp_service_account' not found in Streamlit secrets.")
                return False
            
            service_account_info = dict(st.secrets["gcp_service_account"])
            
            self.credentials = service_account.Credentials.from_service_account_info(
                service_account_info,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            self.storage_client = storage.Client(
                credentials=self.credentials,
                project=self.credentials.project_id
            )
            st.sidebar.success("✅ Authentication successful using Streamlit secrets.")
            return True
        except Exception as e:
            st.sidebar.error(f"❌ Authentication failed: {str(e)}")
            return False

    def initialize_clients(self) -> bool:
        """Initialize Vertex AI clients"""
        try:
            vertexai.init(
                project=GCP_PROJECT_ID,
                location=GCP_REGION,
                credentials=self.credentials
            )
            
            # --- CORRECTED: Define the RAG tool using the Vertex AI SDK ---
            self.rag_tool = Tool.from_retrieval(
                Retrieval(
                    VertexRagStore(
                        rag_resources=[{"rag_corpus": RAG_CORPUS_RESOURCE_NAME}],
                        similarity_top_k=10,
                        vector_distance_threshold=0.5
                    )
                )
            )

            # --- CORRECTED: Initialize the model object from the Vertex AI SDK ---
            self.model = GenerativeModel(self.model_name)
            
            st.sidebar.success("✅ All clients initialized successfully.")
            return True
        except Exception as e:
            st.sidebar.error(f"❌ Client initialization failed: {str(e)}")
            logger.error(f"Client initialization error: {e}", exc_info=True)
            return False

    def classify_query_type(self, user_query: str) -> str:
        """Classify query type to determine if RAG search is needed"""
        query_lower = user_query.lower().strip()
        greeting_patterns = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'how are you']
        dms_keywords = ['document', 'file', 'upload', 'user', 'role', 'workflow', 'dms', 'system']
        if any(pattern in query_lower for pattern in greeting_patterns) and len(query_lower.split()) < 4:
            return "greeting"
        if any(keyword in query_lower for keyword in dms_keywords):
            return "dms_specific"
        return "general"

    def get_enhanced_persona_prompt(self, user_query: str) -> str:
        """Create context-aware persona prompt"""
        return f"""You are an expert DMS (Document Management System) assistant.
Your primary function is to answer questions based on the provided documentation using your retrieval tool.
For general conversation, like greetings, be friendly and brief.
For questions about the DMS, be detailed, accurate, and professional. Use step-by-step instructions when possible.
If the provided documents don't answer the question, state that clearly.

User query: "{user_query}"
"""

    def process_query(self, user_query: str) -> tuple[str, List[Dict[str, Any]]]:
        """Process user query using the correct Vertex AI SDK methods"""
        try:
            query_type = self.classify_query_type(user_query)
            
            # --- CORRECTED: Use Content and Part from the vertexai SDK ---
            contents = [Content(role="user", parts=[Part.from_text(self.get_enhanced_persona_prompt(user_query))])]
            
            tools_to_use = None
            if query_type == "dms_specific":
                tools_to_use = [self.rag_tool]
                temp = 0.1
            else: # greeting or general
                temp = 0.7

            # --- CORRECTED: Use GenerationConfig and SafetySettings from the vertexai SDK ---
            generation_config = GenerationConfig(
                temperature=temp,
                top_p=0.9,
                max_output_tokens=8192,
                candidate_count=1,
            )
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }

            # --- CORRECTED: This is the proper way to call the model in the vertexai SDK ---
            # This single call works for both RAG and general queries.
            response = self.model.generate_content(
                contents,
                generation_config=generation_config,
                safety_settings=safety_settings,
                tools=tools_to_use,
                stream=False  # Simpler to process non-streamed for now
            )
            
            response_text = response.text
            retrieved_sources = []

            # --- CORRECTED: Accessing grounding metadata from the vertexai SDK response ---
            if query_type == "dms_specific" and response.grounding_metadata.retrieval_queries:
                for query in response.grounding_metadata.retrieval_queries:
                    for ref in query.retrieved_references:
                        retrieved_sources.append({
                            "uri": ref.resource.uri,
                            "title": ref.resource.title,
                            "relevance_score": ref.relevance_score
                        })
                retrieved_sources.sort(key=lambda x: x.get('relevance_score', 0.0), reverse=True)

            return response_text, retrieved_sources
            
        except Exception as e:
            logger.error(f"Query processing error: {e}", exc_info=True)
            error_msg = f"I encountered an error while processing your query: {e}. Please try again."
            return error_msg, []

    def display_sources(self, sources: List[Dict[str, Any]]):
        """Display retrieved sources"""
        if sources:
            st.markdown("---")
            st.markdown("### 🔍 **Sources Referenced:**")
            for i, source in enumerate(sources[:5]):
                with st.expander(f"📄 **Source {i+1}:** {source.get('title', 'Unknown Document')}", expanded=False):
                    st.write(f"**Relevance:** {source.get('relevance_score', 0.0):.2f}")
                    if source.get('uri'):
                        st.markdown(f"**Link:** [View Document]({source['uri']})")

# --- Main Application Logic (Unchanged in principle, but now uses correct objects) ---

@st.cache_resource
def get_chatbot():
    return DMSChatbot()

chatbot = get_chatbot()

if not chatbot.auth_success:
    chatbot.auth_success = chatbot.setup_authentication()

if chatbot.auth_success and chatbot.model is None:
    if not chatbot.initialize_clients():
        st.error("Failed to initialize the chatbot. Please check configuration and logs.")
        st.stop()

if chatbot.auth_success and chatbot.model:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "👋 Welcome to the DMS Chatbot! How can I help you today?"}]
    
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
            chatbot.display_sources(sources)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 **Quick Actions**")
    quick_questions = [
        "How do I create a new document?",
        "What are the user roles in DMS?",
        "How do I search for documents?"
    ]
    for question in quick_questions:
        if st.sidebar.button(question, key=f"quick_{question}"):
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()
    
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

else:
    st.error("❌ Authentication failed or clients not initialized. Please check your `secrets.toml` file.")
    st.markdown("Ensure your `secrets.toml` is correctly formatted in the `.streamlit/` directory.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 **Configuration Info**")
st.sidebar.info(f"**Project:** {GCP_PROJECT_ID}")
st.sidebar.info(f"**Region:** {GCP_REGION}")
st.sidebar.info(f"**Model:** {chatbot.model_name}")
