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
SERVICE_ACCOUNT_JSON_PATH = os.environ.get("SERVICE_ACCOUNT_JSON_PATH", "prj-auropro-dev-404fd024f226.json")

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
        self.model_name = "gemini-2.0-flash-exp"  # Using the latest model
        self.credentials = None
        self.auth_success = False
        
    def setup_authentication(self) -> bool:
        """Setup authentication using Streamlit secrets"""
        try:
            # Try to get credentials from Streamlit secrets first
            if "gcp_service_account" in st.secrets:
                # Use Streamlit secrets
                self.credentials = service_account.Credentials.from_service_account_info(
                    st.secrets["gcp_service_account"],
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                st.sidebar.success("✅ Using Streamlit secrets for authentication")
                return True
            else:
                # Fallback to other methods
                st.sidebar.header("🔐 Authentication")
                auth_method = st.sidebar.selectbox(
                    "Choose authentication method:",
                    ["Service Account JSON File", "Manual JSON Input", "Environment Variables", "Default Credentials"]
                )
                
                if auth_method == "Service Account JSON File":
                    return self._auth_from_file()
                elif auth_method == "Manual JSON Input":
                    return self._auth_from_input()
                elif auth_method == "Environment Variables":
                    return self._auth_from_env()
                else:
                    return self._auth_default()
        except Exception as e:
            st.sidebar.error(f"❌ Authentication failed: {str(e)}")
            return False
    
    def _auth_from_file(self) -> bool:
        """Authenticate using service account file"""
        if os.path.exists(SERVICE_ACCOUNT_JSON_PATH):
            try:
                self.credentials = service_account.Credentials.from_service_account_file(
                    SERVICE_ACCOUNT_JSON_PATH,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                st.sidebar.success(f"✅ Loaded service account from file")
                return True
            except Exception as e:
                st.sidebar.error(f"❌ Error loading service account: {str(e)}")
                return False
        else:
            st.sidebar.warning(f"⚠️ Service account file not found at: `{SERVICE_ACCOUNT_JSON_PATH}`")
            return False
    
    def _auth_from_input(self) -> bool:
        """Authenticate using manual JSON input"""
        json_key_input = st.sidebar.text_area(
            "Paste Service Account JSON:",
            height=150,
            help="Paste the entire contents of your service account JSON file here.",
            type="password"
        )
        if json_key_input:
            try:
                key_data = json.loads(json_key_input)
                self.credentials = service_account.Credentials.from_service_account_info(
                    key_data,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                st.sidebar.success("✅ Service account loaded from input")
                return True
            except Exception as e:
                st.sidebar.error(f"❌ Error parsing JSON: {str(e)}")
                return False
        return False
    
    def _auth_from_env(self) -> bool:
        """Authenticate using environment variables"""
        google_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if google_creds:
            try:
                if os.path.exists(google_creds):
                    self.credentials = service_account.Credentials.from_service_account_file(
                        google_creds,
                        scopes=['https://www.googleapis.com/auth/cloud-platform']
                    )
                else:
                    # Try parsing as JSON string
                    key_data = json.loads(google_creds)
                    self.credentials = service_account.Credentials.from_service_account_info(
                        key_data,
                        scopes=['https://www.googleapis.com/auth/cloud-platform']
                    )
                st.sidebar.success("✅ Using environment variable credentials")
                return True
            except Exception as e:
                st.sidebar.error(f"❌ Error with environment credentials: {str(e)}")
                return False
        else:
            st.sidebar.warning("⚠️ GOOGLE_APPLICATION_CREDENTIALS not found in environment")
            return False
    
    def _auth_default(self) -> bool:
        """Use default credentials"""
        st.sidebar.info("🔄 Attempting to use default credentials...")
        return True
    
    def initialize_clients(self) -> bool:
        """Initialize Vertex AI and GenAI clients"""
        try:
            # Initialize Vertex AI
            if self.credentials:
                vertexai.init(
                    project=GCP_PROJECT_ID, 
                    location=GCP_REGION, 
                    credentials=self.credentials
                )
            else:
                vertexai.init(project=GCP_PROJECT_ID, location=GCP_REGION)
            
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
                # For greetings/general: No RAG, higher temperature for more natural responses
                generate_content_config = types.GenerateContentConfig(
                    temperature=0.7,  # Higher for more natural conversation
                    top_p=0.9,
                    top_k=40,
                    max_output_tokens=2048,  # Shorter responses for greetings
                    candidate_count=1,
                    
                    safety_settings=[
                        types.SafetySetting(
                            category="HARM_CATEGORY_HATE_SPEECH",
                            threshold="BLOCK_MEDIUM_AND_ABOVE"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_DANGEROUS_CONTENT",
                            threshold="BLOCK_MEDIUM_AND_ABOVE"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            threshold="BLOCK_MEDIUM_AND_ABOVE"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_HARASSMENT",
                            threshold="BLOCK_MEDIUM_AND_ABOVE"
                        )
                    ],
                    
                    response_mime_type="text/plain",
                    # NO TOOLS for greetings/general queries
                )
            else:
                # For DMS-specific queries: Use RAG with lower temperature
                generate_content_config = types.GenerateContentConfig(
                    temperature=0.1,  # Lower for more consistent responses
                    top_p=0.9,
                    top_k=40,
                    max_output_tokens=8192,  # Increased for comprehensive responses
                    candidate_count=1,
                    
                    safety_settings=[
                        types.SafetySetting(
                            category="HARM_CATEGORY_HATE_SPEECH",
                            threshold="BLOCK_MEDIUM_AND_ABOVE"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_DANGEROUS_CONTENT",
                            threshold="BLOCK_MEDIUM_AND_ABOVE"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            threshold="BLOCK_MEDIUM_AND_ABOVE"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_HARASSMENT",
                            threshold="BLOCK_MEDIUM_AND_ABOVE"
                        )
                    ],
                    
                    response_mime_type="text/plain",
                    tools=self.tools_for_gemini  # Use RAG tools only for DMS queries
                )
            
            # Generate response with streaming
            response_stream = self.genai_client.models.generate_content_stream(
                model=self.model_name,
                contents=genai_contents,
                config=generate_content_config,
            )
            
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
            logger.error(f"Query processing error: {str(e)}")
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

# Authentication setup - automatically try to authenticate
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
            # Show different spinner messages based on query type
            query_type = chatbot.classify_query_type(prompt)
            
            if query_type in ["greeting", "general"]:
                with st.spinner("💭 Thinking..."):
                    response_text, sources = chatbot.process_query(prompt)
            else:
                with st.spinner("🤔 Thinking and searching the documentation..."):
                    response_text, sources = chatbot.process_query(prompt)
            
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
    st.warning("⚠️ Please configure authentication in the sidebar to use the chatbot.")
    st.markdown("""
    ### 🔧 **Setup Instructions:**
    1. **Choose an authentication method** from the sidebar
    2. **Provide your Google Cloud credentials**
    3. **Ensure your service account has the required permissions**
    4. **Verify your RAG Corpus is configured correctly**
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
