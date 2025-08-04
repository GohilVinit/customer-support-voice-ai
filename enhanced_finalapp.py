# Updated imports and OpenAI client initialization
import sys
import time
import os
import shutil
import json
import requests
import threading
from datetime import datetime
import streamlit as st
from fpdf import FPDF
from dotenv import load_dotenv
# Removed NVIDIA imports - using alternative approaches
# from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Updated imports for newer LangChain versions
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
# Removed outdated chain imports - will use direct API calls instead
from openai import OpenAI
import subprocess
import signal
import atexit

# Force UTF-8 encoding for printing
sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

# Set API keys (removed NVIDIA dependencies)
# os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')
os.environ['SER_AGENT'] = "streamlit-rag-demo"

# Initialize OpenAI client (updated approach)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# JSON file path for chat history
CHAT_HISTORY_FILE = "chat_history.json"

# Backend API configuration
BACKEND_API_URL = "http://localhost:5000"  # Updated to match new backend port
backend_process = None

# Pre-recorded responses
pre_recorded_responses = {
    "default": "I'm sorry, I couldn't find relevant information. Can I assist you with anything else?",
    "greeting": "Hello! How can I help you today?",
    "thanks": "You're welcome! Let me know if you need anything else.",
    "goodbye": "Goodbye! Have a great day!"
}

def start_backend_server():
    """Start the Flask backend server"""
    global backend_process
    try:
        backend_process = subprocess.Popen([
            sys.executable, "backend.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if server is running
        try:
            response = requests.get(f"{BACKEND_API_URL}/api/health", timeout=5)
            if response.status_code == 200:
                return True
        except:
            pass
        
        return False
    except Exception as e:
        st.error(f"Failed to start backend server: {e}")
        return False

def stop_backend_server():
    """Stop the Flask backend server"""
    global backend_process
    if backend_process:
        backend_process.terminate()
        backend_process.wait()

# Register cleanup function
atexit.register(stop_backend_server)

def check_backend_status():
    """Check if backend server is running"""
    try:
        response = requests.get(f"{BACKEND_API_URL}/api/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def upload_video_to_backend(file_path, language="en"):
    """Upload video file to backend for transcription"""
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'language': language}
            
            response = requests.post(
                f"{BACKEND_API_URL}/api/upload",
                files=files,
                data=data,
                timeout=300  # 5 minutes timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Upload failed: {response.json().get('error', 'Unknown error')}")
                return None
    except Exception as e:
        st.error(f"Error uploading video: {e}")
        return None

def add_documents_to_backend(documents):
    """Add documents to the unified vectorstore via backend API"""
    try:
        # Convert documents to the format expected by backend
        doc_data = []
        for doc in documents:
            doc_data.append({
                'content': doc.page_content,
                'metadata': doc.metadata
            })
        
        response = requests.post(
            f"{BACKEND_API_URL}/api/vectorstore/add_documents",
            json={'documents': doc_data},
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to add documents: {response.json().get('error', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error adding documents to backend: {e}")
        return None

def get_backend_retriever_info():
    """Get retriever information from backend"""
    try:
        response = requests.get(f"{BACKEND_API_URL}/api/vectorstore/retriever", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error getting retriever info: {e}")
        return None

def start_meeting_recording(meeting_url, bot_name="AI Notetaker", language="en"):
    """Start recording a meeting using MeetingBaas API"""
    try:
        response = requests.post(
            f"{BACKEND_API_URL}/api/meeting/start",
            json={
                'meeting_url': meeting_url,
                'bot_name': bot_name,
                'language': language
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Meeting recording failed: {response.json().get('error', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error starting meeting recording: {e}")
        return None

def get_meeting_status(bot_id):
    """Get meeting recording status"""
    try:
        response = requests.get(
            f"{BACKEND_API_URL}/api/meeting/status/{bot_id}",
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to get meeting status: {response.json().get('error', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error getting meeting status: {e}")
        return None

def stop_meeting_recording(bot_id):
    """Stop meeting recording"""
    try:
        response = requests.post(
            f"{BACKEND_API_URL}/api/meeting/stop/{bot_id}",
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to stop meeting: {response.json().get('error', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error stopping meeting: {e}")
        return None

def get_active_meetings():
    """Get all active meeting recordings"""
    try:
        response = requests.get(f"{BACKEND_API_URL}/api/meeting/active", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to get active meetings: {response.json().get('error', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error getting active meetings: {e}")
        return None

def get_transcripts_from_backend():
    """Get all transcripts from backend"""
    try:
        response = requests.get(f"{BACKEND_API_URL}/api/transcripts", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to get transcripts: {response.json().get('error', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error getting transcripts: {e}")
        return None

def chat_with_backend(question):
    """Chat with backend using unified embeddings"""
    try:
        response = requests.post(
            f"{BACKEND_API_URL}/api/chat",
            json={'question': question},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Chat failed: {response.json().get('error', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error chatting with backend: {e}")
        return None

def search_unified_data_backend(query, n_results=5):
    """Search unified data in backend"""
    try:
        response = requests.post(
            f"{BACKEND_API_URL}/api/search",
            json={'query': query, 'n_results': n_results},
            timeout=15
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Search failed: {response.json().get('error', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error searching unified data: {e}")
        return None

# Enhanced Analysis Functions
def comprehensive_analysis_backend(file_path, language="en", enable_diarization=True, 
                                enable_insights=True, enable_summary=True, 
                                enable_evaluation=True, enable_agent_analysis=True):
    """Perform comprehensive customer support call analysis"""
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {
                'language': language,
                'enable_diarization': str(enable_diarization).lower(),
                'enable_insights': str(enable_insights).lower(),
                'enable_summary': str(enable_summary).lower(),
                'enable_evaluation': str(enable_evaluation).lower(),
                'enable_agent_analysis': str(enable_agent_analysis).lower()
            }
            
            response = requests.post(
                f"{BACKEND_API_URL}/api/analysis/comprehensive",
                files=files,
                data=data,
                timeout=600  # 10 minutes timeout for comprehensive analysis
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Comprehensive analysis failed: {response.json().get('error', 'Unknown error')}")
                return None
    except Exception as e:
        st.error(f"Error performing comprehensive analysis: {e}")
        return None

def extract_insights_backend(transcript):
    """Extract insights from transcript"""
    try:
        response = requests.post(
            f"{BACKEND_API_URL}/api/analysis/insights",
            json={'transcript': transcript},
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Insights extraction failed: {response.json().get('error', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error extracting insights: {e}")
        return None

def generate_summary_backend(transcript, insights=None):
    """Generate summary from transcript"""
    try:
        data = {'transcript': transcript}
        if insights:
            data['insights'] = insights
            
        response = requests.post(
            f"{BACKEND_API_URL}/api/analysis/summary",
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Summary generation failed: {response.json().get('error', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None

def evaluate_call_backend(transcript, insights=None, summary=None):
    """Evaluate call quality"""
    try:
        data = {'transcript': transcript}
        if insights:
            data['insights'] = insights
        if summary:
            data['summary'] = summary
            
        response = requests.post(
            f"{BACKEND_API_URL}/api/analysis/evaluation",
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Call evaluation failed: {response.json().get('error', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error evaluating call: {e}")
        return None

def agent_analysis_backend(transcript, insights=None, summary=None):
    """Perform agent analysis and get recommendations"""
    try:
        data = {'transcript': transcript}
        if insights:
            data['insights'] = insights
        if summary:
            data['summary'] = summary
            
        response = requests.post(
            f"{BACKEND_API_URL}/api/analysis/agent",
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Agent analysis failed: {response.json().get('error', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error performing agent analysis: {e}")
        return None

def enhanced_chat_backend(query, transcript=None):
    """Enhanced chat with agentic capabilities"""
    try:
        data = {'query': query}
        if transcript:
            data['transcript'] = transcript
            
        response = requests.post(
            f"{BACKEND_API_URL}/api/analysis/chat",
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Enhanced chat failed: {response.json().get('error', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error with enhanced chat: {e}")
        return None

def perform_diarization_backend(file_path):
    """Perform speaker diarization on audio file"""
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            
            response = requests.post(
                f"{BACKEND_API_URL}/api/analysis/diarization",
                files=files,
                timeout=300  # 5 minutes timeout for diarization
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Diarization failed: {response.json().get('error', 'Unknown error')}")
                return None
    except Exception as e:
        st.error(f"Error performing diarization: {e}")
        return None

# Updated OpenAI Fallback function
def openai_fallback(user_input):
    try:
        # Updated to use the new OpenAI client and chat completions
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error with OpenAI API: {e}")
        return pre_recorded_responses["default"]

# Title
st.title("🎯 Enhanced Customer Support Call Analysis System")
st.markdown("**Unified RAG Demo - Documents, Videos, Live Meetings & Advanced Analysis**")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "backend_running" not in st.session_state:
    st.session_state.backend_running = False

if "active_meetings" not in st.session_state:
    st.session_state.active_meetings = {}

if "local_documents_processed" not in st.session_state:
    st.session_state.local_documents_processed = False

# Save chat history to JSON
def save_chat_to_json():
    chat_data = [{"question": q, "answer": a, "source": s} for q, a, s in st.session_state.chat_history]
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_data, f, indent=4, ensure_ascii=False)

# Backend Server Management
st.sidebar.header("Backend Server")
if st.sidebar.button("Start Backend Server"):
    with st.spinner("Starting backend server..."):
        if start_backend_server():
            st.session_state.backend_running = True
            st.sidebar.success("Backend server started!")
        else:
            st.sidebar.error("Failed to start backend server")

if st.sidebar.button("Check Backend Status"):
    if check_backend_status():
        st.session_state.backend_running = True
        st.sidebar.success("Backend server is running")
    else:
        st.session_state.backend_running = False
        st.sidebar.error("Backend server is not responding")

if st.sidebar.button("Stop Backend Server"):
    stop_backend_server()
    st.session_state.backend_running = False
    st.sidebar.info("Backend server stopped")

# Display backend status
backend_status = "Running" if st.session_state.backend_running else "Stopped"
st.sidebar.write(f"Backend Status: {backend_status}")

# Main Content Area - Enhanced Call Analysis (Front and Center)
if st.session_state.backend_running:
    st.header("🎯 Enhanced Call Analysis")
    st.markdown("Upload your customer support call recordings for comprehensive analysis including sentiment, tonality, intent, summarization, and quality evaluation.")
    
    # Analysis type selection
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Comprehensive Analysis", "Individual Components", "Speaker Diarization", "Enhanced Chat"]
    )
    
    if analysis_type == "Comprehensive Analysis":
        st.subheader("📊 Comprehensive Call Analysis")
        
        # File upload for comprehensive analysis
        analysis_file = st.file_uploader(
            "Upload Call Recording",
            type=["mp4", "avi", "mov", "mkv", "mp3", "wav", "flac", "aac", "ogg"],
            key="comprehensive_analysis"
        )
        
        if analysis_file:
            # Analysis options
            st.subheader("Analysis Options")
            col1, col2 = st.columns(2)
            
            with col1:
                enable_diarization = st.checkbox("Enable Speaker Diarization", value=True)
                enable_insights = st.checkbox("Enable Insights Extraction", value=True)
                enable_summary = st.checkbox("Enable Call Summarization", value=True)
            
            with col2:
                enable_evaluation = st.checkbox("Enable Quality Evaluation", value=True)
                enable_agent = st.checkbox("Enable Agent Analysis", value=True)
                language = st.selectbox("Language", ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"])
            
            if st.button("🚀 Start Comprehensive Analysis", type="primary"):
                # Save file temporarily
                temp_path = f"./temp_comprehensive_{analysis_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(analysis_file.read())
                
                with st.spinner("Performing comprehensive analysis... This may take several minutes."):
                    result = comprehensive_analysis_backend(
                        temp_path, language, enable_diarization, 
                        enable_insights, enable_summary, enable_evaluation, enable_agent
                    )
                    
                    if result:
                        st.session_state.comprehensive_analysis_result = result
                        st.success("✅ Comprehensive analysis completed!")
                    else:
                        st.error("❌ Analysis failed")
                
                # Clean up temp file
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    elif analysis_type == "Individual Components":
        st.subheader("🔍 Individual Analysis Components")
        
        # Transcript input for individual analysis
        transcript_text = st.text_area(
            "Enter transcript text for analysis:",
            height=150,
            placeholder="Paste your call transcript here..."
        )
        
        if transcript_text:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Extract Insights"):
                    with st.spinner("Extracting insights..."):
                        insights_result = extract_insights_backend(transcript_text)
                        if insights_result:
                            st.session_state.insights_result = insights_result
                            st.success("✅ Insights extracted!")
                
                if st.button("📝 Generate Summary"):
                    with st.spinner("Generating summary..."):
                        summary_result = generate_summary_backend(transcript_text)
                        if summary_result:
                            st.session_state.summary_result = summary_result
                            st.success("✅ Summary generated!")
                
                if st.button("⭐ Evaluate Quality"):
                    with st.spinner("Evaluating call quality..."):
                        evaluation_result = evaluate_call_backend(transcript_text)
                        if evaluation_result:
                            st.session_state.evaluation_result = evaluation_result
                            st.success("✅ Quality evaluated!")
            
            with col2:
                if st.button("🤖 Agent Analysis"):
                    with st.spinner("Performing agent analysis..."):
                        agent_result = agent_analysis_backend(transcript_text)
                        if agent_result:
                            st.session_state.agent_result = agent_result
                            st.success("✅ Agent analysis completed!")
    
    elif analysis_type == "Speaker Diarization":
        st.subheader("🎤 Speaker Diarization")
        
        diarization_file = st.file_uploader(
            "Upload Audio File for Speaker Separation",
            type=["mp3", "wav", "flac", "aac", "ogg"],
            key="diarization_analysis"
        )
        
        if diarization_file:
            if st.button("🎤 Start Speaker Diarization"):
                # Save file temporarily
                temp_path = f"./temp_diarization_{diarization_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(diarization_file.read())
                
                with st.spinner("Performing speaker diarization..."):
                    result = perform_diarization_backend(temp_path)
                    if result:
                        st.session_state.diarization_result = result
                        st.success("✅ Speaker diarization completed!")
                    else:
                        st.error("❌ Diarization failed")
                
                # Clean up temp file
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    elif analysis_type == "Enhanced Chat":
        st.subheader("💬 Enhanced Chat")
        
        query = st.text_input("Enter your question:", placeholder="Ask about the call analysis...")
        transcript = st.text_area("Optional: Provide transcript context:", height=100)
        
        if query:
            if st.button("💬 Send Enhanced Query"):
                with st.spinner("Processing enhanced query..."):
                    result = enhanced_chat_backend(query, transcript)
                    if result:
                        st.session_state.enhanced_chat_result = result
                        st.success("✅ Enhanced response generated!")
                    else:
                        st.error("❌ Query processing failed")

# Enhanced Analysis Results Display
if st.session_state.backend_running:
    st.header("📊 Analysis Results")
    
    # Comprehensive Analysis Results
    if hasattr(st.session_state, 'comprehensive_analysis_result') and st.session_state.comprehensive_analysis_result:
        st.subheader("📊 Comprehensive Analysis Results")
        
        result = st.session_state.comprehensive_analysis_result.get('results', {})
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📝 Basic Info", "🎤 Diarization", "📊 Insights", "📋 Summary", 
            "⭐ Evaluation", "🤖 Agent Analysis", "📈 Performance"
        ])
        
        with tab1:
            st.write("**File Information:**")
            file_info = result.get('file_info', {})
            st.json(file_info)
            
            st.write("**Overall Analysis Score:**")
            score = result.get('overall_analysis_score', 0)
            st.metric("Analysis Score", f"{score:.1f}/100")
        
        with tab2:
            diarization = result.get('diarization', {})
            if diarization:
                st.write("**Speaker Analysis:**")
                speaker_analysis = diarization.get('speaker_analysis', {})
                st.json(speaker_analysis)
                
                st.write("**Speaker Segments:**")
                segments = diarization.get('speaker_segments', [])
                if segments:
                    for i, segment in enumerate(segments[:10]):  # Show first 10 segments
                        st.write(f"Segment {i+1}: {segment}")
            else:
                st.info("Speaker diarization not performed or failed")
        
        with tab3:
            insights = result.get('insights', {})
            if insights:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    sentiment = insights.get('sentiment_analysis', {})
                    if sentiment:
                        st.write("**Sentiment Analysis:**")
                        st.metric("Overall Sentiment", sentiment.get('overall_sentiment', 'Unknown'))
                        st.metric("Confidence", f"{sentiment.get('confidence_score', 0)}%")
                
                with col2:
                    tonality = insights.get('tonality_analysis', {})
                    if tonality:
                        st.write("**Tonality Analysis:**")
                        st.metric("Primary Tonality", tonality.get('primary_tonality', 'Unknown'))
                        st.metric("Emotional Intensity", f"{tonality.get('emotional_intensity', 0)}/10")
                
                with col3:
                    intent = insights.get('intent_analysis', {})
                    if intent:
                        st.write("**Intent Analysis:**")
                        st.metric("Primary Intent", intent.get('primary_intent', 'Unknown'))
                        st.metric("Resolution Status", intent.get('resolution_status', 'Unknown'))
                
                st.write("**Call Quality Score:**")
                quality_score = insights.get('call_quality_score', 0)
                st.metric("Quality Score", f"{quality_score:.1f}/100")
            else:
                st.info("Insights extraction not performed or failed")
        
        with tab4:
            summary = result.get('summary', {})
            if summary:
                st.write("**Executive Summary:**")
                st.write(summary.get('executive_summary', 'No summary available'))
                
                st.write("**Main Issue:**")
                st.write(summary.get('main_issue', 'No issue identified'))
                
                st.write("**Key Points:**")
                key_points = summary.get('key_points', [])
                if key_points:
                    for point in key_points:
                        st.write(f"• {point}")
                
                st.write("**Brief Summary:**")
                st.write(result.get('brief_summary', 'No brief summary available'))
                
                st.write("**Key Phrases:**")
                key_phrases = result.get('key_phrases', [])
                if key_phrases:
                    for phrase in key_phrases:
                        st.write(f"• {phrase}")
            else:
                st.info("Summary generation not performed or failed")
        
        with tab5:
            evaluation = result.get('evaluation', {})
            if evaluation:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Quality Metrics:**")
                    st.metric("Overall Quality", f"{evaluation.get('overall_quality_score', 0)}/10")
                    st.metric("Communication", f"{evaluation.get('communication_effectiveness', 0)}/10")
                    st.metric("Problem Resolution", f"{evaluation.get('problem_resolution', 0)}/10")
                
                with col2:
                    st.write("**Customer Satisfaction:**")
                    st.metric("Satisfaction", f"{evaluation.get('customer_satisfaction', 0)}/10")
                    st.metric("Agent Professionalism", f"{evaluation.get('agent_professionalism', 0)}/10")
                    st.metric("Call Efficiency", f"{evaluation.get('call_efficiency', 0)}/10")
                
                st.write("**Areas for Improvement:**")
                improvements = evaluation.get('areas_for_improvement', [])
                if improvements:
                    for improvement in improvements:
                        st.write(f"• {improvement}")
                
                st.write("**Strengths:**")
                strengths = evaluation.get('strengths_highlighted', [])
                if strengths:
                    for strength in strengths:
                        st.write(f"• {strength}")
            else:
                st.info("Quality evaluation not performed or failed")
        
        with tab6:
            agent_analysis = result.get('agent_analysis', {})
            if agent_analysis:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Immediate Actions:**")
                    immediate_actions = agent_analysis.get('immediate_actions', [])
                    if immediate_actions:
                        for action in immediate_actions:
                            st.write(f"• {action}")
                
                with col2:
                    st.write("**Follow-up Actions:**")
                    follow_up_actions = agent_analysis.get('follow_up_actions', [])
                    if follow_up_actions:
                        for action in follow_up_actions:
                            st.write(f"• {action}")
                
                st.write("**Training Opportunities:**")
                training = agent_analysis.get('training_opportunities', [])
                if training:
                    for opportunity in training:
                        st.write(f"• {opportunity}")
                
                st.write("**Priority Level:**")
                st.metric("Priority", agent_analysis.get('priority_level', 'Unknown'))
            else:
                st.info("Agent analysis not performed or failed")
        
        with tab7:
            performance = result.get('performance_metrics', {})
            if performance:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Call Statistics:**")
                    word_count = performance.get('word_count', 0)
                    sentence_count = performance.get('sentence_count', 0)
                    avg_words = performance.get('avg_words_per_sentence', 0)
                    
                    st.metric("Word Count", word_count)
                    st.metric("Sentence Count", sentence_count)
                    st.metric("Avg Words/Sentence", f"{avg_words:.1f}" if avg_words is not None else "0.0")
                
                with col2:
                    st.write("**Performance Metrics:**")
                    call_duration = performance.get('call_duration_seconds', 0)
                    words_per_minute = performance.get('words_per_minute', 0)
                    efficiency_score = performance.get('efficiency_score', 0)
                    
                    # Fix the formatting error by adding proper null checks
                    st.metric("Call Duration", f"{call_duration:.1f}s" if call_duration is not None else "0.0s")
                    st.metric("Words per Minute", f"{words_per_minute:.1f}" if words_per_minute is not None else "0.0")
                    st.metric("Efficiency Score", f"{efficiency_score:.1f}" if efficiency_score is not None else "0.0")
            else:
                st.info("Performance metrics not available")
    
    # Individual Component Results
    if hasattr(st.session_state, 'insights_result') and st.session_state.insights_result:
        st.subheader("📊 Insights Analysis Results")
        insights = st.session_state.insights_result.get('insights', {})
        st.json(insights)
    
    if hasattr(st.session_state, 'summary_result') and st.session_state.summary_result:
        st.subheader("📋 Summary Results")
        summary = st.session_state.summary_result
        st.write("**Executive Summary:**")
        st.write(summary.get('summary', {}).get('executive_summary', 'No summary available'))
        st.write("**Brief Summary:**")
        st.write(summary.get('brief_summary', 'No brief summary available'))
        st.write("**Key Phrases:**")
        key_phrases = summary.get('key_phrases', [])
        for phrase in key_phrases:
            st.write(f"• {phrase}")
    
    if hasattr(st.session_state, 'evaluation_result') and st.session_state.evaluation_result:
        st.subheader("⭐ Quality Evaluation Results")
        evaluation = st.session_state.evaluation_result.get('evaluation', {})
        st.json(evaluation)
    
    if hasattr(st.session_state, 'agent_result') and st.session_state.agent_result:
        st.subheader("🤖 Agent Analysis Results")
        agent_analysis = st.session_state.agent_result.get('agent_analysis', {})
        st.json(agent_analysis)
    
    if hasattr(st.session_state, 'diarization_result') and st.session_state.diarization_result:
        st.subheader("🎤 Speaker Diarization Results")
        diarization = st.session_state.diarization_result.get('diarization', {})
        st.json(diarization)
    
    if hasattr(st.session_state, 'enhanced_chat_result') and st.session_state.enhanced_chat_result:
        st.subheader("💬 Enhanced Chat Results")
        chat_result = st.session_state.enhanced_chat_result.get('response', {})
        st.json(chat_result)

# Sidebar - Other Features
st.sidebar.header("📁 Other Features")

# Meeting Recording Section
if st.session_state.backend_running:
    st.sidebar.subheader("🎥 Meeting Recording")
    
    meeting_url = st.sidebar.text_input("Meeting URL (Zoom, Teams, Meet):", placeholder="https://zoom.us/j/123456789")
    bot_name = st.sidebar.text_input("Bot Name:", value="AI Notetaker")
    meeting_language = st.sidebar.selectbox("Meeting Language:", ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"])
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("Start Recording"):
            if meeting_url:
                with st.spinner("Starting meeting recording..."):
                    result = start_meeting_recording(meeting_url, bot_name, meeting_language)
                    if result:
                        bot_id = result.get('bot_id')
                        st.session_state.active_meetings[bot_id] = {
                            'meeting_url': meeting_url,
                            'bot_name': bot_name,
                            'status': 'starting',
                            'start_time': datetime.now()
                        }
                        st.sidebar.success(f"Recording started! Bot ID: {bot_id}")
            else:
                st.sidebar.error("Please enter a meeting URL")
    
    with col2:
        if st.button("Refresh Status"):
            active_meetings = get_active_meetings()
            if active_meetings:
                st.session_state.active_meetings = active_meetings.get('meetings', {})

    # Display active meetings
    if st.session_state.active_meetings:
        st.sidebar.subheader("Active Meetings")
        for bot_id, meeting_info in st.session_state.active_meetings.items():
            with st.sidebar.container():
                st.write(f"**Bot:** {meeting_info.get('bot_name', 'Unknown')}")
                st.write(f"**Status:** {meeting_info.get('status', 'Unknown')}")
                st.write(f"**Bot ID:** {bot_id}")
                
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    if st.button(f"Status", key=f"status_{bot_id}"):
                        status_result = get_meeting_status(bot_id)
                        if status_result:
                            st.json(status_result)
                
                with col2:
                    if st.button(f"Stop", key=f"stop_{bot_id}"):
                        with st.spinner("Stopping recording..."):
                            stop_result = stop_meeting_recording(bot_id)
                            if stop_result:
                                if bot_id in st.session_state.active_meetings:
                                    del st.session_state.active_meetings[bot_id]
                                st.sidebar.success("Recording stopped and processed!")
                st.write("---")

# Sidebar: Upload
st.sidebar.header("Upload Files")
upload_type = st.sidebar.selectbox("Select file type", ["PDF/TXT Documents", "Video/Audio Files"])

if upload_type == "PDF/TXT Documents":
    uploaded_files = st.sidebar.file_uploader("Choose PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    upload_folder = "./uploaded_files"
    os.makedirs(upload_folder, exist_ok=True)

    if uploaded_files:
        for file in uploaded_files:
            file_path = os.path.join(upload_folder, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
        st.sidebar.success("Document files uploaded successfully")

elif upload_type == "Video/Audio Files":
    if not st.session_state.backend_running:
        st.sidebar.warning("Backend server must be running to process video/audio files")
    else:
        video_files = st.sidebar.file_uploader(
            "Choose video/audio files", 
            type=["mp4", "avi", "mov", "mkv", "mp3", "wav", "flac", "aac", "ogg"],
            accept_multiple_files=True
        )
        
        if video_files:
            language = st.sidebar.selectbox("Select language", ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"])
            
            for video_file in video_files:
                # Save file temporarily
                temp_path = f"./temp_{video_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(video_file.read())
                
                # Upload to backend
                with st.spinner(f"Processing {video_file.name}..."):
                    result = upload_video_to_backend(temp_path, language)
                    
                    if result:
                        st.sidebar.success(f"{video_file.name} processed and added to unified embeddings!")
                        st.sidebar.json(result)
                    else:
                        st.sidebar.error(f"Failed to process {video_file.name}")
                
                # Clean up temp file
                try:
                    os.remove(temp_path)
                except:
                    pass

# Convert TXT to PDF
def txt_to_pdf(txt_file, pdf_file):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    with open(txt_file, "r", encoding="utf-8") as file:
        for line in file:
            pdf.multi_cell(0, 10, line)
    pdf.output(pdf_file)

if st.sidebar.button("Convert TXT to PDF"):
    upload_folder = "./uploaded_files"
    txt_files = [f for f in os.listdir(upload_folder) if f.endswith('.txt')]
    
    if txt_files:
        for txt_file in txt_files:
            txt_file_path = os.path.join(upload_folder, txt_file)
            pdf_file_path = txt_file_path.replace(".txt", ".pdf")
            txt_to_pdf(txt_file_path, pdf_file_path)
        st.sidebar.success(f"Converted {len(txt_files)} TXT files to PDF")
    else:
        st.sidebar.warning("No TXT files found to convert")

# Document Processing Section
st.sidebar.header("Document Processing")

if st.sidebar.button("Process Local Documents to Unified Embeddings"):
    if not st.session_state.backend_running:
        st.sidebar.error("Backend server must be running")
    else:
        try:
            upload_folder = "./uploaded_files"
            all_docs = []
            
            # Load uploaded files
            if os.path.exists(upload_folder):
                for file_name in os.listdir(upload_folder):
                    if file_name.endswith('.pdf'):
                        file_path = os.path.join(upload_folder, file_name)
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                        
                        # Add document metadata
                        for doc in docs:
                            doc.metadata["filename"] = file_name
                            doc.metadata["file_path"] = file_path
                        
                        all_docs.extend(docs)
            
            # Fallback to default directory if no uploads
            if not all_docs:
                if os.path.exists("./us_census"):
                    loader = PyPDFDirectoryLoader("./us_census")
                    all_docs = loader.load()
                    
                    # Add metadata for default docs
                    for doc in all_docs:
                        if "filename" not in doc.metadata:
                            doc.metadata["filename"] = "us_census_doc"

            if not all_docs:
                st.sidebar.warning("No documents found to process")
            else:
                # Split documents
                splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
                final_documents = splitter.split_documents(all_docs)
                
                st.sidebar.write(f"Loaded {len(all_docs)} documents")
                st.sidebar.write(f"Split into {len(final_documents)} chunks")
                
                # Send to backend
                with st.spinner("Adding documents to unified embeddings..."):
                    result = add_documents_to_backend(final_documents)
                    
                    if result:
                        st.sidebar.success("Documents added to unified embeddings successfully!")
                        st.session_state.local_documents_processed = True
                        st.sidebar.json(result)
                    else:
                        st.sidebar.error("Failed to add documents to unified embeddings")
                        
        except Exception as e:
            st.sidebar.error(f"Error processing documents: {e}")

# Transcript Management
if st.session_state.backend_running:
    st.sidebar.header("Transcript Management")
    
    if st.sidebar.button("View All Transcripts"):
        transcripts = get_transcripts_from_backend()
        if transcripts:
            st.sidebar.write(f"Total transcripts: {transcripts.get('count', 0)}")
            st.session_state.transcripts = transcripts.get('transcripts', [])

# Show Chat History
if st.session_state.chat_history:
    st.subheader("Chat History")
    for i, (q, a, s) in enumerate(st.session_state.chat_history):
        st.markdown(f"**Q{i+1}:** {q}")
        st.markdown(f"**A{i+1}** (from {s}): {a}")
        st.write("---")

# Admin Dashboard
if st.sidebar.checkbox("Admin Dashboard"):
    st.subheader("Admin: System Statistics")
    
    # Local processing stats
    st.write(f"Local documents processed: {st.session_state.local_documents_processed}")
    
    # Backend stats
    if st.session_state.backend_running:
        try:
            debug_response = requests.get(f"{BACKEND_API_URL}/api/debug", timeout=5)
            if debug_response.status_code == 200:
                debug_data = debug_response.json()
                st.write("**Backend Services:**")
                st.json(debug_data)
        except Exception as e:
            st.error(f"Error getting backend stats: {e}")
        
        # Retriever info
        retriever_info = get_backend_retriever_info()
        if retriever_info:
            st.write("**Unified Vectorstore Info:**")
            st.json(retriever_info)
    
    # Download Chat History
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            chat_json = f.read()
        st.download_button("Download Chat History", chat_json, file_name="chat_history.json", mime="application/json")

# Instructions
with st.expander("Instructions"):
    st.markdown("""
    ### 🎯 How to use this Enhanced Customer Support Call Analysis System:
    
    #### **1. Backend Setup**
    - **Start Backend Server**: Click "Start Backend Server" to enable all analysis features
    - **Check Status**: Verify backend is running before using advanced features
    
    #### **2. Enhanced Call Analysis** (New Features!)
    - **Comprehensive Analysis**: Upload call recordings for full analysis including:
      - Speech-to-text transcription
      - Speaker diarization (customer vs agent separation)
      - Sentiment, tonality, and intent analysis
      - Call summarization and key phrase extraction
      - Quality evaluation and performance metrics
      - Intelligent agent recommendations
    - **Individual Components**: Test specific analysis features separately
    - **Speaker Diarization**: Separate customer and agent speech
    - **Enhanced Chat**: AI-powered chat with call context
    
    #### **3. Meeting Recording**
    - Enter meeting URL (Zoom, Teams, Google Meet)
       - Configure bot name and language
    - Start recording and monitor active meetings
    - Stop recording to process and analyze
    
    #### **4. File Upload & Processing**
    - **Documents**: Upload PDF/TXT files for document Q&A
    - **Video/Audio**: Upload call recordings for transcription and analysis
    - **Process Documents**: Add to unified vectorstore for search
    
    #### **5. Unified Chat & Search**
    - Ask questions across all data (documents + transcripts)
    - Search specific information in your knowledge base
    - View chat history and download results
    
    ### 🚀 **New Enhanced Features:**
    
    #### **📊 Comprehensive Call Analysis**
    - **8-Step Analysis Pipeline**: Complete call evaluation
    - **Memory Optimized**: Designed for 4GB RAM systems
    - **Configurable Components**: Enable/disable analysis features
    - **Real-time Results**: Live progress tracking
    
    #### **🎤 Speaker Diarization**
    - **Customer vs Agent Separation**: Identify who's speaking
    - **Speaker Pattern Analysis**: Analyze speaking patterns
    - **Memory Efficient**: Optimized for limited RAM
    
    #### **📊 Advanced Insights**
    - **Sentiment Analysis**: Positive/Negative/Neutral detection
    - **Tonality Analysis**: Voice tone and emotional intensity
    - **Intent Classification**: Complaint/Query/Feedback detection
    - **Quality Scoring**: Automatic call quality assessment
    
    #### **📋 Intelligent Summarization**
    - **Executive Summaries**: High-level call overview
    - **Key Point Extraction**: Important issues and resolutions
    - **Brief Summaries**: Quick reference summaries
    - **Key Phrase Identification**: Important terms and concepts
    
    #### **⭐ Quality Evaluation**
    - **Communication Effectiveness**: Score agent communication
    - **Problem Resolution**: Assess issue resolution
    - **Customer Satisfaction**: Measure satisfaction indicators
    - **Performance Metrics**: Detailed call analytics
    
    #### **🤖 Agent Recommendations**
    - **Immediate Actions**: 24-hour follow-up tasks
    - **Training Opportunities**: Agent improvement suggestions
    - **Process Improvements**: Workflow optimization
    - **Priority Assessment**: Call priority determination
    
    ### 📁 **Supported File Types:**
    - **Meeting Platforms**: Zoom, Microsoft Teams, Google Meet, Webex
    - **Documents**: PDF, TXT
    - **Video/Audio**: MP4, AVI, MOV, MKV, MP3, WAV, FLAC, AAC, OGG
    
    ### 💾 **System Requirements:**
    - **RAM**: 4GB minimum (optimized for this configuration)
    - **Storage**: 2GB+ free space
    - **API Keys**: OpenAI (required), Gemini (required), HF Token (optional)
    
    ### 🔧 **Memory Management:**
    - **Automatic Monitoring**: Real-time memory usage tracking
    - **Efficient Processing**: Chunked file processing
    - **Resource Cleanup**: Automatic garbage collection
    - **Configurable Thresholds**: Adjustable memory limits
    
    ### 📈 **Analysis Results:**
    - **Comprehensive Reports**: Detailed analysis with scores
    - **Visual Metrics**: Charts and progress indicators
    - **Export Capabilities**: Download results and reports
    - **Historical Tracking**: Compare analysis over time
    """)