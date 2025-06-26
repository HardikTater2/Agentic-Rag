import os
import fitz
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Dict, TypedDict, List, Any
from langchain_community.vectorstores import FAISS
from textblob import TextBlob
from supabase import create_client, Client
from datetime import datetime
import uuid
import logging
import re

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
TICKET_TYPES = [
    "attendance_disparity",
    "leave_application",
    "evaluation",
    "missed_exam",
    "other"
]

ADMISSION_FIELDS = {
    "full_name": "What is your full name?",
    "email": "What is your email address?",
    "phone": "What is your phone number?",
    "interested_program": "Which program are you interested in? (e.g., B.Tech Computer Science, MBA, etc.)",
    "qualification": "What is your highest qualification?",
    "year_of_passing": "What year did you pass your qualifying examination?"
}

SUPPORT_PHONE = "+1-800-555-0199"  # Example support phone number

# Enhanced Exit/Cancel commands for better context clearing
EXIT_COMMANDS = [
    "exit", "quit", "cancel", "abort", "stop", "reset", 
    "start over", "new conversation", "clear", "restart",
    "nevermind", "forget it", "go back"
]

# Initialize models and services
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

# Enhanced Supabase initialization with better error handling
def init_supabase() -> Client:
    """Initialize Supabase client with comprehensive error handling"""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError("Supabase URL and key must be set in environment variables. Please check your .env file or deployment settings.")

    return create_client(supabase_url, supabase_key)

supabase = init_supabase()

# Enhanced document processing with better error handling
def load_and_process_document(file_path: str) -> FAISS:
    """Load and process PDF document into FAISS vector store with enhanced error handling"""
    try:
        if not os.path.exists(file_path):
            logger.error(f"Document file not found: {file_path}")
            raise FileNotFoundError(f"Document file not found: {file_path}")
        
        markdown_text = ""
        doc = fitz.open(file_path)
        
        for page in doc:
            text_dict = page.get_text("dict")
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["flags"] & 2 ** 1:  # Bold
                                markdown_text += f"**{span['text']}**"
                            elif span["flags"] & 2 ** 2:  # Italic
                                markdown_text += f"*{span['text']}*"
                            else:
                                markdown_text += span["text"]
                            markdown_text += "\n"
                        markdown_text += "\n"
        
        doc.close()
        
        if not markdown_text.strip():
            raise ValueError("Document appears to be empty or unreadable")
        
        chunks = text_splitter.split_text(markdown_text)
        logger.info(f"Document processed successfully: {len(chunks)} chunks created")
        return FAISS.from_texts(chunks, embeddings)
        
    except Exception as e:
        logger.error(f"Failed to process document: {str(e)}")
        raise RuntimeError(f"Failed to process document: {str(e)}")

# Initialize knowledge base with error handling
try:
    srb_vdb1 = load_and_process_document(r"C:\Users\AMISH\Downloads\SRB 2023-24_04.10.2023.pdf")
    logger.info("Knowledge base initialized successfully")
except Exception as e:
    logger.error(f"Error initializing knowledge base: {e}")
    srb_vdb1 = None

class AgentState(TypedDict):
    input: str
    sender: str
    chat_history: List[Dict[str, str]]
    current_agent: str
    agent_output: str
    metadata: Dict[str, Any]

# Enhanced helper functions
def generate_ticket_id() -> str:
    """Generate a unique ticket ID with timestamp"""
    return f"TKT-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"

def validate_email(email: str) -> bool:
    """Enhanced email validation with regex"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))

def validate_phone(phone: str) -> bool:
    """Enhanced phone number validation"""
    phone_clean = re.sub(r'[^\d]', '', phone)
    return len(phone_clean) >= 10 and len(phone_clean) <= 15

def analyze_sentiment(text: str) -> str:
    """Analyze sentiment with TextBlob and enhanced error handling"""
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.2:
            return "positive"
        elif polarity < -0.2:
            return "negative"
        else:
            return "neutral"
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        return "neutral"

def should_exit_context(user_input: str) -> bool:
    """Enhanced exit context detection with more patterns"""
    user_input_lower = user_input.lower().strip()
    return any(cmd in user_input_lower for cmd in EXIT_COMMANDS)

def should_switch_agent(user_input: str) -> bool:
    """Detect explicit agent switching requests"""
    switch_patterns = [
        "switch to", "change to", "i want to", "help me with",
        "talk to", "connect me to", "i need help with", "i want to ask about"
    ]
    return any(pattern in user_input.lower() for pattern in switch_patterns)

# Enhanced Router function with comprehensive state handling
def router(state: AgentState) -> str:
    """
    ENHANCED router with FIXED state handling and proper context management.
    
    Priority system:
    1. Exit commands (highest priority)
    2. Pending agent states 
    3. Explicit agent switching
    4. Keyword mapping
    5. LLM classification (fallback)
    """
    meta = state.get("metadata", {})
    user_input = state["input"].lower().strip()

    # PRIORITY 1: Handle exit commands - ALWAYS clear context first
    if should_exit_context(user_input):
        logger.info("Exit command detected, clearing context")
        state["metadata"] = {}
        return "faq"  # Default to FAQ after clearing context

    # PRIORITY 2: Check for ALL pending agent states (FIXED - now includes all states)
    if meta.get("awaiting_ticket_confirmation"):
        logger.info("Routing to ticket agent - awaiting confirmation")
        return "ticket"
    elif meta.get("collecting_feedback"):  # FIXED - was missing this check
        logger.info("Routing to feedback agent - collecting feedback")
        return "feedback"
    elif meta.get("current_field"):  # FIXED - was missing this check
        logger.info("Routing to admissions agent - collecting field data")
        return "admissions"

    # PRIORITY 3: Handle explicit agent switching commands
    if should_switch_agent(user_input):
        logger.info("Explicit agent switching detected, clearing context")
        state["metadata"] = {}  # Clear context when explicitly switching agents

        # Enhanced agent switching with better keyword detection
        if any(word in user_input for word in ["faq", "information", "question", "ask", "policy", "rule"]):
            return "faq"
        elif any(word in user_input for word in ["ticket", "issue", "problem", "help", "support"]):
            return "ticket"
        elif any(word in user_input for word in ["feedback", "suggestion", "opinion", "review", "complain"]):
            return "feedback"
        elif any(word in user_input for word in ["admission", "apply", "enroll", "join", "program"]):
            return "admissions"

    # PRIORITY 4: Enhanced keyword mapping with more comprehensive keywords
    keyword_mapping = {
        "faq": [
            "information", "what is", "how to", "policy", "rule", "procedure", 
            "when is", "where is", "explain", "define", "tell me about", 
            "details", "requirements", "guidelines", "handbook"
        ],
        "ticket": [
            "issue", "problem", "ticket", "attendance", "leave", "exam", 
            "evaluation", "missing", "wrong", "error", "help me with", 
            "bug", "technical", "system", "portal", "login", "access"
        ],
        "feedback": [
            "feedback", "suggest", "improve", "compliment", "complain", 
            "opinion", "review", "rate", "experience", "service", 
            "quality", "satisfied", "disappointed"
        ],
        "admissions": [
            "admission", "apply", "enroll", "enrollment", "program", 
            "course", "join", "application", "register", "eligibility",
            "fees", "scholarship", "placement", "campus"
        ],
    }

    for agent, keywords in keyword_mapping.items():
        if any(keyword in user_input for keyword in keywords):
            logger.info(f"Keyword match found, routing to {agent} agent")
            return agent

    # PRIORITY 5: LLM-based classification for ambiguous queries
    prompt = f"""
    Carefully analyze this college-related query and classify it into the most appropriate category:
    
    Query: "{state['input']}"

    Available categories:
    1. faq - General information requests about policies, rules, procedures, academic information
    2. ticket - Issues requiring administrative action (technical problems, attendance issues, leave applications, exam problems)
    3. feedback - Opinions, suggestions, complaints, or reviews about college services, faculty, infrastructure
    4. admissions - Questions about the application process, enrollment, program details, eligibility
    5. live - Complex issues requiring human intervention, grievances, or general conversation

    Respond with ONLY the category name (faq/ticket/feedback/admissions/live).
    """

    try:
        response = llm.invoke(prompt)
        agent = response.content.strip().lower()
        logger.info(f"LLM classification result: {agent}")
        
        if agent in ["faq", "ticket", "feedback", "admissions", "live"]:
            return agent
        else:
            logger.warning(f"Invalid LLM response: {agent}, defaulting to live")
            return "live"
            
    except Exception as e:
        logger.error(f"LLM routing error: {e}")
        return "live"

# Agent 1: FAQ Agent (unchanged but with better error handling)
def search_knowledge_base(query: str, k: int = 3) -> str:
    """Search college FAQ knowledge base with enhanced error handling"""
    if not srb_vdb1:
        logger.error("Knowledge base not available")
        return "Knowledge base is currently unavailable. Please try again later."
    
    try:
        faq_r = srb_vdb1.similarity_search(query, k=k)
        if not faq_r:
            logger.warning(f"No results found for query: {query}")
            return ""
        
        return "\n\n".join([doc.page_content for doc in faq_r])
    except Exception as e:
        logger.error(f"Knowledge base search error: {e}")
        return ""

def faq_agent(state: AgentState) -> Dict[str, Any]:
    """Handle FAQ queries with improved response generation"""
    try:
        result = search_knowledge_base(state["input"])
        
        if not result:
            return {
                "agent_output": "I couldn't find relevant information about your query. Please try rephrasing your question or ask about something else.",
                "metadata": {}
            }

        prompt = f"""
        You are a helpful college assistant. Answer the student's question using ONLY the provided context.
        If the answer isn't in the context, politely say you don't know and suggest they contact support.

        Context:
        {result}

        Question: {state['input']}

        Provide a clear, concise answer in simple language. If referring to rules/policies,
        mention the relevant section if available in the context.
        """

        response = llm.invoke(prompt)
        return {
            "agent_output": response.content,
            "metadata": {}
        }
        
    except Exception as e:
        logger.error(f"FAQ agent error: {e}")
        return {
            "agent_output": "I'm having trouble answering that right now. Please try again later or contact support.",
            "metadata": {}
        }

# Agent 2: Enhanced Ticket Agent with better state management
def ticket_agent(state: AgentState) -> Dict[str, Any]:
    """
    Handle ticket creation with ENHANCED state management and proper context clearing.
    Maintains exact same workflow while fixing exit handling issues.
    """
    meta = state.get("metadata", {})
    user_input = state["input"].strip().lower()

    # Enhanced: Check for cancellation at any point in the conversation
    if should_exit_context(user_input):
        logger.info("Ticket creation cancelled by user")
        return {
            "agent_output": "Ticket creation cancelled. How else can I help you?",
            "metadata": {}  # Clear all metadata to exit ticket mode
        }

    # Case 1: Awaiting confirmation for ticket creation
    if meta.get("awaiting_ticket_confirmation"):
        if user_input in ["yes", "y", "confirm", "create", "proceed"]:
            try:
                ticket_data = meta["pending_ticket"]
                # REMOVED ticket_id generation
                ticket_data.update({
                    "created_at": datetime.now().isoformat(),
                    "sender": state["sender"],
                    "original_query": meta.get("original_query", state["input"])
                })

                # Insert ticket data
                result = supabase.table("tickets").insert(ticket_data).execute()
                
                # Get ACTUAL ID from database response
                if result.data and len(result.data) > 0:
                    db_ticket_id = result.data[0]['id']
                else:
                    db_ticket_id = "TKT-UNKNOWN"

                logger.info(f"Ticket created successfully. DB ID: {db_ticket_id}")

                return {
                    "agent_output": (
                        f"✅ Ticket created successfully!\n\n"
                        f"• **Ticket ID:** {db_ticket_id}\n"
                        f"• **Type:** {ticket_data['ticket_type'].replace('_', ' ').title()}\n"
                        f"• **Status:** Open\n\n"
                        "We'll contact you within 24-48 hours."
                    ),
                    "metadata": {}
                }
                
            except Exception as e:
                logger.error(f"Ticket creation error: {e}")
                return {
                    "agent_output": (
                        "❌ Failed to create ticket due to a system error. "
                        "Would you like to try again? (yes/no)"
                    ),
                    "metadata": meta
                }

        elif user_input in ["no", "n", "cancel", "abort"]:
            logger.info("Ticket creation declined by user")
            return {
                "agent_output": "Understood. Ticket creation cancelled. What would you like to do instead?",
                "metadata": {}  # Clear all metadata
            }
        else:
            return {
                "agent_output": "Please confirm ticket creation with 'yes' or 'no'. You can also say 'cancel' to abort.",
                "metadata": meta  # Maintain current state
            }

    # Case 2: New ticket creation flow
    try:
        prompt = f"""
        Classify this support request into one of these categories:
        {", ".join(TICKET_TYPES)}
        
        Support Request: "{state['input']}"
        
        Respond with ONLY the ticket type from the options above.
        """

        response = llm.invoke(prompt)
        ticket_type = response.content.strip().lower()
        
        if ticket_type not in TICKET_TYPES:
            ticket_type = "other"
            
        logger.info(f"Ticket classified as: {ticket_type}")

    except Exception as e:
        logger.error(f"Ticket classification error: {e}")
        ticket_type = "other"

    # Prepare ticket data structure
    ticket_data = {
        "sender": state["sender"],
        "query": state["input"],
        "ticket_type": ticket_type,
        "status": "open",
        "urgency": "normal"
    }

    return {
        "agent_output": (
            f"I'll create a **{ticket_type.replace('_', ' ').title()}** ticket for you:\n\n"
            f"**Issue:** {state['input']}\n"
            f"**Type:** {ticket_type.replace('_', ' ').title()}\n\n"
            "Please confirm by typing 'yes' or cancel by typing 'no'."
        ),
        "metadata": {
            "awaiting_ticket_confirmation": True,
            "pending_ticket": ticket_data,
            "original_query": state["input"]
        }
    }

# Agent 3: FIXED Feedback Agent with authentication removed and database issues resolved
def feedback_agent(state: AgentState) -> Dict[str, Any]:
    """
    Handle feedback collection with authentication removed and database issues fixed.
    Key changes:
    1. Removed session_id causing database errors
    2. Removed authentication requirement
    3. Added default "anonymous" sender
    4. Simplified database insertion
    """
    user_input = state["input"].strip()
    
    # Enhanced: Check for exit commands
    if should_exit_context(user_input):
        logger.info("Feedback collection cancelled by user")
        return {
            "agent_output": "Feedback collection cancelled. What would you like to do instead?",
            "metadata": {}
        }

    # Initial feedback collection prompt
    if not state.get("metadata", {}).get("collecting_feedback"):
        logger.info("Starting feedback collection")
        return {
            "agent_output": (
                "We'd love to hear your feedback! 💭\n\n"
                "Please share your thoughts about:\n"
                "• Our college services\n"
                "• Faculty and teaching quality\n"
                "• Infrastructure and facilities\n"
                "• Any suggestions for improvement\n\n"
                "What would you like to tell us?"
            ),
            "metadata": {
                "collecting_feedback": True,
                "original_query": state["input"]
            }
        }

    # Process feedback with comprehensive error handling
    feedback_text = state["input"]
    
    try:
        # Enhanced sentiment analysis with error handling
        sentiment = analyze_sentiment(feedback_text)
        logger.info(f"Sentiment analyzed: {sentiment}")
        
        # Prepare feedback data with authentication removed
        feedback_data = {
            "sender": state.get("sender", "anonymous"),  # Default to anonymous
            "feedback_text": feedback_text,
            "sentiment": sentiment,
            "timestamp": datetime.now().isoformat(),
            "original_query": state["metadata"].get("original_query", "")
        }
        
        logger.info(f"Attempting to save feedback for sender: {feedback_data['sender']}")
        
        # ENHANCED database insertion with detailed error handling
        try:
            # Insert feedback
            result = supabase.table("feedback").insert(feedback_data).execute()
            logger.info(f"Feedback saved successfully")
            
            # Success response with sentiment-based messaging
            success_message = "Thank you for your valuable feedback! 🙏\n\n"
            
            if sentiment == "positive":
                success_message += "We're delighted to hear about your positive experience! Your encouragement motivates us to keep improving."
            elif sentiment == "negative":
                success_message += "We appreciate you bringing this to our attention. Your feedback helps us identify areas for improvement."
            else:
                success_message += "Your input is valuable to us and will help us serve you better."
            
            success_message += "\n\nWe'll review your feedback and take appropriate action where needed."
            
            return {
                "agent_output": success_message,
                "metadata": {}
            }
            
        except Exception as db_error:
            # Detailed error analysis for debugging
            error_msg = str(db_error).lower()
            logger.error(f"Database error details: {db_error}")
            
            # Specific error handling based on error type
            if "authentication" in error_msg or "jwt" in error_msg:
                user_message = (
                    "❌ Authentication issue detected. "
                    "Please try again or contact support if the problem persists."
                )
            elif "policy" in error_msg or "rls" in error_msg or "permission" in error_msg:
                user_message = (
                    "❌ Permission issue detected. Our team has been notified. "
                    "Please try again later or contact support."
                )
            elif "relation" in error_msg or "table" in error_msg or "column" in error_msg:
                user_message = (
                    "❌ Database configuration issue. Our technical team has been notified. "
                    "Please contact support for immediate assistance."
                )
            elif "network" in error_msg or "connection" in error_msg:
                user_message = (
                    "❌ Network connectivity issue. Please check your internet connection "
                    "and try again in a few moments."
                )
            else:
                user_message = (
                    "❌ We couldn't save your feedback due to a technical issue. "
                    "Please try again later or contact our support team."
                )
            
            user_message += f"\n\n📞 Support: {SUPPORT_PHONE}"
            
            return {
                "agent_output": user_message,
                "metadata": {}
            }
            
    except Exception as e:
        logger.error(f"General feedback processing error: {e}")
        return {
            "agent_output": (
                "❌ An unexpected error occurred while processing your feedback. "
                f"Please contact support at {SUPPORT_PHONE} for assistance."
            ),
            "metadata": {}
        }

# Agent 4: Enhanced Admission Agent with better validation and exit handling
def admission_agent(state: AgentState) -> Dict[str, Any]:
    """
    Handle admission inquiries with ENHANCED validation and exit handling.
    Maintains exact same workflow structure.
    """
    user_input = state["input"].strip()
    user_input_lower = user_input.lower()

    # Enhanced: Check for exit commands at any point
    if should_exit_context(user_input):
        logger.info("Admission inquiry cancelled by user")
        return {
            "agent_output": "Admission inquiry cancelled. How else can I help you?",
            "metadata": {}
        }

    metadata = state.get("metadata", {})
    admission_data = metadata.get("admission_data", {})
    current_field = metadata.get("current_field")

    # Process current field input if we're in the middle of collection
    if current_field:
        valid = True
        error_msg = ""

        # Enhanced field validation
        if current_field == "email":
            if not validate_email(state["input"]):
                valid = False
                error_msg = "Please enter a valid email address (e.g., name@example.com)."
        elif current_field == "phone":
            if not validate_phone(state["input"]):
                valid = False
                error_msg = "Please enter a valid phone number (10-15 digits)."
        elif current_field == "year_of_passing":
            year_input = state["input"].strip()
            if not year_input.isdigit() or len(year_input) != 4 or int(year_input) < 1950 or int(year_input) > 2030:
                valid = False
                error_msg = "Please enter a valid year (e.g., 2023)."
        elif current_field == "full_name":
            if len(state["input"].strip()) < 2:
                valid = False
                error_msg = "Please enter your full name (at least 2 characters)."

        if not valid:
            return {
                "agent_output": error_msg,
                "metadata": metadata
            }

        # Store the validated input
        admission_data[current_field] = state["input"].strip()
        metadata["admission_data"] = admission_data
        metadata["current_field"] = None
        logger.info(f"Admission field '{current_field}' collected successfully")

    # Find next field to collect
    next_field = None
    for field in ADMISSION_FIELDS:
        if field not in admission_data:
            next_field = field
            break

    # If all fields collected, submit the application
    if not next_field:
        try:
            admission_data.update({
                "sender": state["sender"],
                "submission_date": datetime.now().isoformat(),
                "original_query": state["metadata"].get("original_query", ""),
                "application_id": f"ADM-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
            })
            
            result = supabase.table("admissions").insert(admission_data).execute()
            logger.info(f"Admission inquiry submitted successfully: {admission_data['application_id']}")

            return {
                "agent_output": (
                    "✅ **Admission Inquiry Submitted Successfully!**\n\n"
                    f"**Application ID:** {admission_data['application_id']}\n"
                    f"**Program:** {admission_data['interested_program']}\n\n"
                    "**Next Steps:**\n"
                    "• Our admissions team will review your information\n"
                    "• You'll receive a follow-up email within 2-3 business days\n"
                    "• We'll provide detailed information about your chosen program\n\n"
                    "Keep an eye on your email for updates! 📧"
                ),
                "metadata": {}
            }
            
        except Exception as e:
            logger.error(f"Admission submission error: {e}")
            return {
                "agent_output": (
                    "❌ We couldn't submit your admission inquiry due to a technical issue. "
                    "Please try again later or contact our admissions office directly."
                ),
                "metadata": metadata
            }

    # Ask for the next field
    metadata["current_field"] = next_field
    
    # Enhanced field prompts with examples
    field_prompts = {
        "full_name": "What is your full name? (e.g., John Smith)",
        "email": "What is your email address? (e.g., john.smith@email.com)",
        "phone": "What is your phone number? (e.g., 9876543210)",
        "interested_program": "Which program are you interested in? (e.g., B.Tech Computer Science, MBA, etc.)",
        "qualification": "What is your highest qualification? (e.g., 12th Science, B.Com, etc.)",
        "year_of_passing": "What year did you pass your qualifying examination? (e.g., 2023)"
    }
    
    prompt = field_prompts.get(next_field, ADMISSION_FIELDS[next_field])
    
    return {
        "agent_output": prompt,
        "metadata": metadata
    }

def live_agent(state: AgentState) -> Dict[str, Any]:
    """Handle live agent requests with improved response"""
    return {
        "agent_output": (
            "I see you need assistance from a human agent. 👨‍💼\n\n"
            "**Contact Options:**\n"
            f"📞 **Phone:** {SUPPORT_PHONE}\n"
            "📧 **Email:** support@college.edu\n"
            "🕒 **Hours:** Monday-Friday, 9 AM - 6 PM\n\n"
            "**For urgent issues:**\n"
            "• Call during business hours for immediate assistance\n"
            "• Email for non-urgent queries - we respond within 24 hours\n\n"
            "Is there anything else I can help you with in the meantime?"
        ),
        "metadata": {}
    }

# ENHANCED Workflow function with FIXED routing and comprehensive error handling
def workflow(state: AgentState) -> Dict[str, Any]:
    """
    ENHANCED workflow with FIXED agent switching, context management, and error handling.
    
    COMPREHENSIVE FIXES:
    1. Proper handling of all exit commands
    2. Better context clearing when switching agents
    3. Improved state management for all agents
    4. Enhanced error handling with detailed logging
    5. Smart metadata preservation
    """
    try:
        # Initialize state components if missing
        state.setdefault("chat_history", [])
        state.setdefault("metadata", {})
        state.setdefault("current_agent", "")
        state.setdefault("sender", "anonymous")  # Default to anonymous
        state.setdefault("input", "")

        # Add user message to history with timestamp
        state["chat_history"].append({
            "role": "user", 
            "content": state["input"],
            "timestamp": datetime.now().isoformat(),
            "agent_context": state["current_agent"]
        })

        user_input = state["input"].lower().strip()
        logger.info(f"Processing message: {state['input'][:50]}...")

        # ENHANCED: Handle complete reset commands
        if user_input in ["reset", "start over", "new session", "clear all", "restart"]:
            logger.info("Complete session reset requested")
            return {
                "agent_output": "✨ Starting fresh! How can I help you today?",
                "metadata": {},
                "chat_history": []
            }

        # ENHANCED: Determine if we should route to a new agent
        meta = state.get("metadata", {})
        has_pending_state = any([
            meta.get("awaiting_ticket_confirmation"),
            meta.get("collecting_feedback"),
            meta.get("current_field")
        ])

        # Smart routing logic
        should_route = (
            not state["current_agent"] or  # No current agent
            should_exit_context(user_input) or  # User wants to exit
            should_switch_agent(user_input) or  # Explicit switching
            not has_pending_state  # No pending states
        )

        # Route to appropriate agent if needed
        if should_route:
            previous_agent = state["current_agent"]
            new_agent = router(state)
            
            # Enhanced context management
            if new_agent != previous_agent:
                logger.info(f"Agent switch: {previous_agent} → {new_agent}")
                state["current_agent"] = new_agent

                # Smart context clearing based on situation
                if should_exit_context(user_input) or should_switch_agent(user_input):
                    # Complete context clear for explicit exits/switches
                    state["metadata"] = {
                        "sender": state.get("sender", "anonymous"),
                        "session_start": datetime.now().isoformat(),
                        "last_active": datetime.now().isoformat()
                    }
                    logger.info("Context completely cleared due to explicit command")
                else:
                    # Preserve essential context for natural transitions
                    preserved_keys = ["sender", "session_start", "original_query", "last_active"]
                    new_metadata = {k: v for k, v in state["metadata"].items() if k in preserved_keys}
                    new_metadata["last_active"] = datetime.now().isoformat()
                    state["metadata"] = new_metadata
                    logger.info("Context partially preserved for natural transition")

        # Enhanced agent mapping with error handling
        agent_map = {
            "faq": faq_agent,
            "ticket": ticket_agent,
            "feedback": feedback_agent,
            "admissions": admission_agent,
            "live": live_agent
        }

        # Get the appropriate agent function
        current_agent = state.get("current_agent", "faq")
        agent_func = agent_map.get(current_agent, faq_agent)
        
        logger.info(f"Executing {current_agent} agent")

        # Execute agent function with error handling
        try:
            response = agent_func(state)
        except Exception as agent_error:
            logger.error(f"Agent execution error in {current_agent}: {agent_error}")
            response = {
                "agent_output": (
                    "❌ I encountered an error while processing your request. "
                    f"Please try again or contact support at {SUPPORT_PHONE}."
                ),
                "metadata": {}
            }

        # Update state with agent response
        state["agent_output"] = response.get("agent_output", "")

        # Enhanced metadata management - merge carefully
        if "metadata" in response:
            # Preserve important system metadata while updating agent-specific metadata
            system_keys = ["sender", "session_start", "last_active"]
            for key in system_keys:
                if key in state["metadata"]:
                    response["metadata"][key] = state["metadata"][key]
            
            state["metadata"] = response["metadata"]

        # Update activity timestamp
        state["metadata"]["last_active"] = datetime.now().isoformat()

        # Add agent response to history with comprehensive metadata
        state["chat_history"].append({
            "role": "agent",
            "content": state["agent_output"],
            "agent_type": current_agent,
            "metadata_snapshot": dict(state["metadata"]),
            "timestamp": datetime.now().isoformat()
        })

        logger.info(f"Successfully processed message, agent: {current_agent}")

        return {
            "agent_output": state["agent_output"],
            "metadata": state["metadata"],
            "chat_history": state["chat_history"]
        }

    except Exception as e:
        logger.error(f"Critical workflow error: {e}", exc_info=True)
        return {
            "agent_output": (
                "❌ I encountered a critical error processing your request. "
                f"Please contact support at {SUPPORT_PHONE} for immediate assistance."
            ),
            "metadata": state.get("metadata", {}),
            "chat_history": state.get("chat_history", [])
        }

# ... (rest of the code remains unchanged) ...


# Enhanced diagnostic function for troubleshooting feedback issues
def diagnose_feedback_system():
    """
    Comprehensive diagnostic tool for debugging feedback system issues.
    Run this function to identify specific problems with your setup.
    """
    print("🔍 Running Feedback System Diagnostics...\n")
    
    # 1. Check environment variables
    print("1. Environment Variables:")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if supabase_url and supabase_key:
        print("   ✅ Supabase credentials found")
        print(f"   📍 URL: {supabase_url[:30]}...")
        print(f"   🔑 Key: {supabase_key[:30]}...")
    else:
        print("   ❌ Missing Supabase credentials")
        return
    
    # 2. Test database connection
    print("\n2. Database Connection:")
    try:
        client = create_client(supabase_url, supabase_key)
        print("   ✅ Connection established")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return
    
    # 3. Test feedback table access
    print("\n3. Feedback Table Access:")
    try:
        result = client.table("feedback").select("*").limit(1).execute()
        print("   ✅ Table accessible")
        print(f"   📊 Current records: {len(result.data)}")
    except Exception as e:
        error_msg = str(e).lower()
        if "relation" in error_msg or "table" in error_msg:
            print("   ❌ Feedback table doesn't exist")
            print("   💡 Solution: Create the feedback table in Supabase")
        elif "policy" in error_msg or "rls" in error_msg:
            print("   ❌ RLS policy blocks access")
            print("   💡 Solution: Add INSERT policy for feedback table")
        else:
            print(f"   ❌ Access error: {e}")
        return
    
    # 4. Test data insertion
    print("\n4. Data Insertion Test:")
    test_data = {
        "sender": "diagnostic_test",
        "feedback_text": "Test feedback for diagnostics",
        "sentiment": "neutral",
        "timestamp": datetime.now().isoformat(),
        "session_id": str(uuid.uuid4())
    }
    
    try:
        result = client.table("feedback").insert(test_data).execute()
        print("   ✅ Test insertion successful")
        
        # Clean up test data
        client.table("feedback").delete().eq("sender", "diagnostic_test").execute()
        print("   🧹 Test data cleaned up")
        
    except Exception as e:
        print(f"   ❌ Insertion failed: {e}")
        error_msg = str(e).lower()
        if "authentication" in error_msg:
            print("   💡 Solution: Check authentication setup")
        elif "policy" in error_msg:
            print("   💡 Solution: Add INSERT policy allowing feedback submission")
        else:
            print("   💡 Solution: Check table schema and permissions")
    
    print("\n✅ Diagnostic complete!")

# Test function for the complete system
def test_system():
    """Test all agents with sample inputs"""
    print("🧪 Testing Multi-Agent System...\n")
    
    test_cases = [
        {"input": "What are the admission requirements?", "expected_agent": "faq"},
        {"input": "I have an attendance issue", "expected_agent": "ticket"}, 
        {"input": "I want to give feedback about faculty", "expected_agent": "feedback"},
        {"input": "How do I apply for B.Tech?", "expected_agent": "admissions"},
        {"input": "I need to speak to someone", "expected_agent": "live"}
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['input']}")
        
        state = {
            "input": test["input"],
            "sender": "test_user",
            "chat_history": [],
            "current_agent": "",
            "agent_output": "",
            "metadata": {}
        }
        
        result = workflow(state)
        agent_used = state.get("current_agent", "unknown")
        
        print(f"   Agent: {agent_used}")
        print(f"   Expected: {test['expected_agent']}")
        print(f"   Match: {'✅' if agent_used == test['expected_agent'] else '❌'}")
        print(f"   Response: {result['agent_output'][:100]}...")
        print()

