import os
import streamlit as st
import vertexai
import json
import logging
from typing import List, Dict, Any, Optional
from google.cloud import storage
from google.oauth2 import service_account
from google import genai
from google.genai import types


import google.auth

# Safely block credential fallback
def _disabled_creds(**kwargs):
    return None

google.auth._default._get_explicit_environ_credentials = _disabled_creds
google.auth._default._detect_gce = lambda: False

# --- Authentication specific import ---


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
        self.model_name = "gemini-1.5-flash-001"  # Using a stable, recommended model
        self.credentials = None
        self.auth_success = False
        self.storage_client = None

        
    def setup_authentication(self) -> bool:
        """Setup authentication using Streamlit secrets"""
        try:
            # Debug: Check if secrets exist
            st.sidebar.info("🔍 Checking authentication...")
            
            # Check if gcp_service_account exists in secrets
            if "gcp_service_account" not in st.secrets:
                st.sidebar.error("❌ 'gcp_service_account' not found in Streamlit secrets")
                st.sidebar.error("Please ensure your secrets.toml file contains the [gcp_service_account] section")
                return False
            
            # Get the service account info from secrets
            service_account_info = st.secrets["gcp_service_account"]
            
            # Debug: Show what keys are available (without showing values)
            available_keys = list(service_account_info.keys())
            st.sidebar.info(f"📋 Available keys in secrets: {available_keys}")
            
            # Required keys for service account
            required_keys = [
                "type", "project_id", "private_key_id", "private_key", 
                "client_email", "client_id", "auth_uri", "token_uri", 
                "auth_provider_x509_cert_url", "client_x509_cert_url", 
                "universe_domain"
            ]
            
            # Check if all required keys are present
            missing_keys = [key for key in required_keys if key not in service_account_info]
            if missing_keys:
                st.sidebar.error(f"❌ Missing required keys in secrets: {missing_keys}")
                return False
            
            # Convert StreamlitSecrets to regular dict
            service_account_dict = dict(service_account_info)
          
            # Create credentials from service account info
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
            # Show more detailed error information
            st.sidebar.error("Please check your secrets.toml file format")
            return False

    def _auth_default(self) -> bool:
        st.sidebar.warning("⚠️ Default credentials disabled. Use secrets.toml instead.")
        return False


    def initialize_clients(self) -> bool:
        """Initialize Vertex AI and GenAI clients"""
        try:
            # Initialize Vertex AI with credentials
            vertexai.init(
                project=GCP_PROJECT_ID, 
                location=GCP_REGION, 
                credentials=self.credentials
            )
            
            # Initialize GenAI client
            self.genai_client = genai.Client(
                vertexai=True,
                project=GCP_PROJECT_ID,
                location=GCP_REGION,
            )
            
            # Configure RAG tools with enhanced settings
            self.tools_for_gemini = [
                types.Tool(
                    retrieval=types.Retrieval(
                        vertex_rag_store=types.VertexRagStore(
                            rag_resources=[
                                types.VertexRagStoreRagResource(
                                    rag_corpus=RAG_CORPUS_RESOURCE_NAME,
                                    # Add similarity threshold for better grounding
                                )
                            ],
                            # Enhanced retrieval configuration
                            similarity_top_k=10,  # Retrieve more candidates
                            vector_distance_threshold=0.5,  # Adjust based on your needs
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
        """Classify query type to determine if RAG search is needed"""
        query_lower = user_query.lower().strip()
        
        # Greetings and small talk patterns
        greeting_patterns = [
            'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
            'how are you', 'what\'s up', 'how\'s it going', 'nice to meet you',
            'thanks', 'thank you', 'bye', 'goodbye', 'see you', 'good day'
        ]
        
        # General conversation patterns
        general_patterns = [
            'what is your name', 'who are you', 'what can you do', 'how can you help',
            'tell me about yourself', 'what are your capabilities', 'how do you work',
            'what\'s the weather', 'how\'s the weather', 'what time is it',
            'tell me a joke', 'can you help me', 'what\'s new'
        ]
        
        # DMS-specific keywords that should trigger RAG search
        dms_keywords = [
            'document', 'file', 'upload', 'download', 'create', 'edit', 'delete',
            'user', 'role', 'permission', 'access', 'login', 'dashboard',
            'workflow', 'approval', 'review', 'version', 'share', 'folder',
            'search', 'filter', 'metadata', 'tag', 'category', 'archive',
            'backup', 'restore', 'export', 'import', 'template', 'form',
            'report', 'analytics', 'audit', 'security', 'compliance',
            'notification', 'alert', 'settings', 'configuration', 'admin',
            'dms', 'system', 'application', 'platform', 'software'
        ]
        
        # Check for exact greeting matches
        if any(pattern in query_lower for pattern in greeting_patterns):
            return "greeting"
        
        # Check for general conversation
        if any(pattern in query_lower for pattern in general_patterns):
            return "general"
        
        # Check for DMS-specific content
        if any(keyword in query_lower for keyword in dms_keywords):
            return "dms_specific"
        
        # If query is very short (less than 3 words), likely greeting/general
        if len(query_lower.split()) < 3:
            return "general"
        
        # Default to DMS-specific for longer queries
        return "dms_specific"
    
    def get_enhanced_persona_prompt(self, user_query: str, query_type: str) -> str:
        """Create context-aware persona prompt based on query type"""
        if query_type == "greeting":
            return f"""
You are a friendly and helpful DMS (Document Management System) assistant. The user is greeting you.

Respond naturally and warmly to their greeting. Keep it brief and welcoming. You can mention that you're here to help with DMS-related questions if appropriate, but don't force it.

User said: "{user_query}"

Respond in a conversational, friendly manner.
"""
        
        elif query_type == "general":
            return f"""
You are a helpful DMS (Document Management System) assistant. The user is asking a general question that doesn't require searching the DMS documentation.

Respond naturally and helpfully. If the question is about your capabilities, mention that you specialize in helping with DMS-related questions and can provide guidance on document management, user roles, workflows, and system features.

User asked: "{user_query}"

Provide a helpful, conversational response using your general knowledge.
"""
        
        else:  # dms_specific
            return f"""
You are an expert DMS (Document Management System) assistant with deep knowledge of the system's functionality and user manual. 

**Core Instructions:**
1. **Primary Focus**: Always prioritize information from the provided DMS documentation when answering DMS-related questions
2. **Accuracy**: Provide precise, step-by-step instructions based on the documentation
3. **Context Intelligence**: Use your understanding to connect related concepts and provide comprehensive answers
4. **Completeness**: Provide thorough answers that address the user's needs

**Response Guidelines:**
- Base your response on the documentation but enhance with logical connections and explanations
- Include relevant details about features, steps, and best practices
- If referencing images/figures from documentation, describe their purpose and location
- For unclear documentation, explain what you understand and what might need clarification
- Connect related features and workflows when relevant

**Current User Query Context:**
The user is asking: "{user_query}"

**Response Style:**
- Be conversational yet professional
- Use clear, numbered steps for processes
- Highlight important points with formatting
- Provide context for why certain steps are necessary
- Suggest related features or alternatives when appropriate

If the question cannot be answered from the documentation, clearly state this and explain what information would be needed.
"""

    def process_query(self, user_query: str) -> tuple[str, List[Dict[str, Any]]]:
        """Process user query with intelligent routing - RAG only when needed"""
        try:
            if self.genai_client is None:
                logger.error("GenAI client is not initialized.")
                return "❌ Internal error: GenAI client not initialized.", []

            # Classify query type to determine if RAG search is needed
            query_type = self.classify_query_type(user_query)
            
            # Prepare content with appropriate persona
            genai_contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part(text=self.get_enhanced_persona_prompt(user_query, query_type))
                    ]
                )
            ]
            
            # Adjust generation config based on query type
            if query_type in ["greeting", "general"]:
                generate_content_config = types.GenerateContentConfig(
                    temperature=0.7,
                    top_p=0.9,
                    top_k=40,
                    max_output_tokens=2048,
                    candidate_count=1,
                    safety_settings=[
                        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_MEDIUM_AND_ABOVE")
                    ],
                    response_mime_type="text/plain",
                )
                tools_to_use = None # No RAG tools for general chat
            else:
                generate_content_config = types.GenerateContentConfig(
                    temperature=0.1,
                    top_p=0.9,
                    top_k=40,
                    max_output_tokens=8192,
                    candidate_count=1,
                    safety_settings=[
                        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_MEDIUM_AND_ABOVE")
                    ],
                    response_mime_type="text/plain",
                )
                tools_to_use = self.tools_for_gemini # Use RAG tools for DMS queries
            
            # Generate response with streaming
            response_stream = self.genai_client.models.generate_content_stream(
                model=self.model_name,
                contents=genai_contents,
                config=generate_content_config,
                tools=tools_to_use
            )
            
            # --------------------- START OF FIX ---------------------
            # Add a safety check to ensure the API call returned a valid stream
            if response_stream is None:
                logger.error("API call to generate_content_stream returned None. This indicates a potential authentication or client configuration issue.")
                error_msg = "I'm sorry, I was unable to connect to the AI service. Please ask the administrator to check the application logs."
                return error_msg, []
            # ---------------------- END OF FIX ----------------------

            response_text = ""
            retrieved_sources = []
            sources_set = set()
            
            # Process streaming response
            for chunk in response_stream:
                if chunk.candidates and chunk.candidates[0].content:
                    # Extract grounding metadata ONLY for DMS-specific queries
                    if query_type == "dms_specific":
                        if hasattr(chunk.candidates[0].content, 'grounding_metadata') and \
                           chunk.candidates[0].content.grounding_metadata:
                            
                            metadata = chunk.candidates[0].content.grounding_metadata
                            if hasattr(metadata, 'retrieval_queries'):
                                for query in metadata.retrieval_queries:
                                    if hasattr(query, 'retrieved_references'):
                                        for ref in query.retrieved_references:
                                            source_key = (
                                                getattr(ref, 'uri', 'N/A'),
                                                getattr(ref, 'title', 'N/A'),
                                                getattr(ref, 'page_number', 'N/A')
                                            )
                                            if source_key not in sources_set:
                                                sources_set.add(source_key)
                                                retrieved_sources.append({
                                                    "uri": source_key[0],
                                                    "title": source_key[1],
                                                    "page_number": source_key[2],
                                                    "relevance_score": getattr(ref, 'relevance_score', 0.0)
                                                })
                    
                    # Extract text content
                    if hasattr(chunk.candidates[0].content, 'parts'):
                        for part in chunk.candidates[0].content.parts:
                            if hasattr(part, 'text') and part.text:
                                response_text += part.text
            
            # Fallback response if no content generated
            if not response_text:
                if query_type in ["greeting", "general"]:
                    response_text = "Hello! I'm here to help you with your DMS (Document Management System). What would you like to know?"
                else:
                    response_text = "I apologize, but I couldn't generate a response for your query. Please try rephrasing your question or check if it's related to the DMS system."
            
            # Sort sources by relevance if available (only for DMS queries)
            if retrieved_sources:
                retrieved_sources.sort(key=lambda x: x.get('relevance_score', 0.0), reverse=True)
            
            return response_text, retrieved_sources
            
        except Exception as e:
            logger.error(f"Query processing error: {str(e)}", exc_info=True)
            error_msg = f"I encountered an error while processing your query: {str(e)}. Please try again or rephrase your question."
            return error_msg, []
    
    def display_sources(self, sources: List[Dict[str, Any]], query_type: str):
        """Display retrieved sources with enhanced formatting - only for DMS queries"""
        if query_type == "dms_specific" and sources:
            st.markdown("---")
            st.markdown("### 🔍 **Sources Referenced:**")
            
            for i, source in enumerate(sources[:5]):  # Limit to top 5 sources
                title = source.get("title", "Unknown Document")
                uri = source.get("uri", "#")
                page = source.get("page_number", "N/A")
                relevance = source.get("relevance_score", 0.0)
                
                # Create expandable source info
                with st.expander(f"📄 **Source {i+1}:** {title}", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Page:** {page}")
                    with col2:
                        if relevance > 0:
                            st.write(f"**Relevance:** {relevance:.2f}")
                    
                    if uri != "N/A" and uri != "#":
                        st.markdown(f"**Link:** [View Document]({uri})")
        elif query_type == "dms_specific" and not sources:
            st.info("💡 This response was generated using general AI knowledge. For specific DMS procedures, try asking about particular features or processes.")
        # For greetings/general queries, don't show any source information

# Initialize the chatbot
@st.cache_resource
def get_chatbot():
    return DMSChatbot()

chatbot = get_chatbot()

# Authentication setup - automatically authenticate using Streamlit secrets
if not chatbot.auth_success:
    chatbot.auth_success = chatbot.setup_authentication()

# Initialize clients if authenticated
if chatbot.auth_success and chatbot.genai_client is None:
    if chatbot.initialize_clients():
        st.sidebar.success("🚀 DMS Chatbot is ready!")
    else:
        st.error("Failed to initialize the chatbot. Please check your configuration.")
        st.stop()

# Chat interface
if chatbot.auth_success and chatbot.genai_client:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        welcome_msg = """
        👋 **Welcome to the DMS Chatbot!** 
        
        I'm here to help you with your Document Management System. You can ask me about:
        - How to create and manage documents
        - User roles and permissions
        - System navigation and features
        - Troubleshooting common issues
        - Best practices for document organization
        
        What would you like to know about the DMS?
        """
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about the DMS..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            query_type = chatbot.classify_query_type(prompt)
            
            # --------------------- START OF FIX ---------------------
            # Restore the spinner and simplify the call to process_query
            spinner_text = "🤔 Thinking and searching the documentation..." if query_type == "dms_specific" else "💭 Thinking..."
            with st.spinner(spinner_text):
                try:
                    # The process_query method is designed to always return a tuple, even on error.
                    response_text, sources = chatbot.process_query(prompt)
                except Exception as e:
                    # This is a fallback for truly unexpected errors.
                    logger.error(f"A critical, unhandled error occurred in the chat loop: {e}", exc_info=True)
                    response_text = f"❌ A critical system error occurred. Please contact support."
                    sources = []
            # ---------------------- END OF FIX ----------------------

            # Display response
            st.markdown(response_text)
            
            # Display sources only for DMS-specific queries
            chatbot.display_sources(sources, query_type)
            
            # Add to chat history
            st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    # Sidebar features
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 **Quick Actions**")
    
    # Quick question buttons
    quick_questions = [
        "How do I create a new document?",
        "What are the user roles in DMS?",
        "How do I search for documents?",
        "How do I share documents with others?",
        "What are the document approval workflows?"
    ]
    
    for question in quick_questions:
        if st.sidebar.button(question, key=f"quick_{question}"):
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()
    
    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

else:
    st.error("❌ Authentication failed. Please check your Streamlit secrets configuration.")
    st.markdown("""
    ### 🔧 **Setup Instructions:**
    1. **Ensure your secrets.toml file is properly configured with the GCP service account**
    2. **Verify the service account has the required permissions**
    3. **Check that the RAG Corpus is configured correctly**
    
    **Your secrets.toml file should look like this:**
    ```toml
    [gcp_service_account]
    type = "service_account"
    project_id = "prj-auropro-dev"
    private_key_id = "your_private_key_id_here"
    private_key = "-----BEGIN PRIVATE KEY-----\\nYOUR_PRIVATE_KEY_HERE\\n-----END PRIVATE KEY-----\\n"
    client_email = "vertex@prj-auropro-dev.iam.gserviceaccount.com"
    client_id = "your_client_id_here"
    auth_uri = "https://accounts.google.com/o/oauth2/auth"
    token_uri = "https://oauth2.googleapis.com/token"
    auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
    client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/vertex%40prj-auropro-dev.iam.gserviceaccount.com"
    universe_domain = "googleapis.com"
    ```
    
    **Important Notes:**
    - Make sure there are no extra spaces or line breaks in your secrets.toml file
    - The private_key should be on a single line with \\n for line breaks
    - All values should be in quotes
    - The file should be placed in the .streamlit/ folder in your project root
    """)

# Footer information
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 **Configuration Info**")
st.sidebar.info(f"**Project:** {GCP_PROJECT_ID}")
st.sidebar.info(f"**Region:** {GCP_REGION}")
st.sidebar.info(f"**Model:** {chatbot.model_name}")

# Performance metrics (if available)
if "messages" in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 **Session Stats**")
    st.sidebar.metric("Messages", len(st.session_state.messages))
    st.sidebar.metric("User Queries", len([m for m in st.session_state.messages if m["role"] == "user"]))
