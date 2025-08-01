import os
import tempfile
import logging
import json
import hashlib
import requests
import time
import gc
import psutil
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Memory management for 4GB RAM system
class MemoryManager:
    """Memory management utility for efficient resource usage on 4GB RAM systems"""
    
    def __init__(self, threshold=80):
        self.threshold = threshold
        self.memory_warnings = 0
    
    def check_memory_usage(self) -> bool:
        """Check if system has sufficient memory for processing"""
        try:
            memory = psutil.virtual_memory()
            if memory.percent > self.threshold:
                self.memory_warnings += 1
                logger.warning(f"Memory usage high: {memory.percent}%")
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking memory: {e}")
            return True
    
    def force_cleanup(self):
        """Force garbage collection to free memory"""
        try:
            gc.collect()
            logger.info("Memory cleanup performed")
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")

# Initialize memory manager
memory_manager = MemoryManager()

# Try to import moviepy, fallback to pydub if not available
try:
    from moviepy.editor import VideoFileClip, AudioFileClip
    USE_MOVIEPY = True
    logger.info("Using moviepy for video/audio processing")
except ImportError:
    try:
        from pydub import AudioSegment
        USE_MOVIEPY = False
        logger.info("Using pydub for audio processing")
    except ImportError:
        raise ImportError("Neither moviepy nor pydub is installed. Please install one of them:\n"
                         "pip install moviepy\n"
                         "or\n"
                         "pip install pydub")

# OpenAI import for Whisper transcription
from openai import OpenAI

# Gemini API import for chat/completion and insights extraction
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    logger.info("Google Generative AI (Gemini) available for chat/completion and insights")
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google Generative AI not available. Install with: pip install google-generativeai")

# Speaker diarization imports (optional for 4GB RAM)
try:
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook
    DIARIZATION_AVAILABLE = True
    logger.info("Pyannote.audio available for speaker diarization")
except ImportError:
    DIARIZATION_AVAILABLE = False
    logger.warning("Pyannote.audio not available. Install with: pip install pyannote.audio")

# LangChain imports for agentic framework
try:
    from langchain.agents import initialize_agent, AgentType
    from langchain.tools import Tool
    from langchain.schema import Document
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
    logger.info("LangChain available for agentic framework")
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available. Install with: pip install langchain")

# Gemini embeddings class for vector operations
class GeminiEmbeddings:
    """Gemini embeddings for vector operations in customer support analysis"""
    
    def __init__(self, api_key=None):
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI not available. Install with: pip install google-generativeai")
        
        if api_key:
            genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel('embedding-001')
        logger.info("Gemini embeddings initialized for customer support analysis")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents for customer support context"""
        try:
            embeddings = []
            for text in texts:
                result = self.model.embed_content(text)
                embeddings.append(result.embedding)
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding documents with Gemini: {str(e)}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query for customer support search"""
        try:
            result = self.model.embed_content(text)
            return result.embedding
        except Exception as e:
            logger.error(f"Error embedding query with Gemini: {str(e)}")
            raise

# NVIDIA and FAISS imports for unified embeddings
try:
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    # Document is already imported above, no need to import again
    NVIDIA_EMBEDDINGS_AVAILABLE = True
    logger.info("NVIDIA Embeddings and FAISS available for unified embeddings")
except ImportError:
    NVIDIA_EMBEDDINGS_AVAILABLE = False
    logger.warning("NVIDIA Embeddings not available. Install with: pip install langchain-nvidia-ai-endpoints")

# Vector database and similarity search imports (keeping for backwards compatibility)
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
    logger.info("ChromaDB available for vector storage")
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("ChromaDB not available. Install with: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("SentenceTransformers available for embeddings")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("SentenceTransformers not available. Install with: pip install sentence-transformers")

# Google Drive imports
try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    import pickle
    from io import BytesIO
    GOOGLE_DRIVE_AVAILABLE = True
    logger.info("Google Drive API available")
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False
    logger.warning("Google Drive API not available. Install with: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'mp3', 'wav', 'flac', 'aac', 'ogg', 'm4a', 'webm', 'flv'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for services
transcriber = None
unified_vectorstore = None
meeting_baas_client = None

class UnifiedVectorStore:
    """Unified vector store for both documents and transcripts using Gemini embeddings and FAISS"""
    
    def __init__(self, vectorstore_path="./unified_vectorstore", gemini_api_key=None):
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI not available. Install with: pip install google-generativeai")
        
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required for embeddings")
        
        self.vectorstore_path = vectorstore_path
        self.embeddings = None
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        self.transcript_metadata = {}  # Store transcript metadata
        self.embedding_type = "gemini"
        
        # Initialize Gemini embeddings
        try:
            self.embeddings = GeminiEmbeddings(gemini_api_key)
            logger.info("Using Gemini embeddings for vector store")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini embeddings: {str(e)}")
            raise ImportError("Gemini embeddings failed to initialize")
        
        # Load existing vectorstore if it exists
        self.load_vectorstore()
        
        logger.info(f"Unified vector store initialized at: {vectorstore_path} with Gemini embeddings")
    
    def load_vectorstore(self):
        """Load existing vectorstore if available"""
        try:
            if os.path.exists(self.vectorstore_path) and self.embeddings:
                self.vectorstore = FAISS.load_local(self.vectorstore_path, self.embeddings)
                logger.info("Loaded existing unified vectorstore with Gemini embeddings")
                
                # Load transcript metadata if exists
                metadata_path = os.path.join(self.vectorstore_path, "transcript_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        self.transcript_metadata = json.load(f)
            else:
                logger.info("No existing vectorstore found")
        except Exception as e:
            logger.warning(f"Could not load existing vectorstore: {e}")
            self.vectorstore = None
    
    def save_vectorstore(self):
        """Save vectorstore and metadata to disk"""
        try:
            if self.vectorstore and self.embeddings:
                # Ensure directory exists
                os.makedirs(self.vectorstore_path, exist_ok=True)
                
                # Save FAISS vectorstore
                self.vectorstore.save_local(self.vectorstore_path)
                
                # Save transcript metadata
                metadata_path = os.path.join(self.vectorstore_path, "transcript_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(self.transcript_metadata, f, indent=2)
                
                logger.info("Unified vectorstore saved successfully")
        except Exception as e:
            logger.error(f"Error saving vectorstore: {e}")
            raise
    
    def add_transcript(self, transcript: str, metadata: Dict[str, Any]) -> str:
        """Add transcript to unified vectorstore"""
        try:
            if not self.embeddings:
                raise ValueError("No embeddings available for vector store operations")
            
            # Generate document ID
            doc_id = hashlib.md5(f"{metadata.get('filename', 'unknown')}_{datetime.now()}".encode()).hexdigest()
            
            # Create document with metadata
            full_metadata = {
                **metadata,
                "doc_id": doc_id,
                "doc_type": "transcript",
                "added_at": datetime.now().isoformat(),
                "embedding_type": self.embedding_type
            }
            
            # Split transcript into chunks
            chunks = self.text_splitter.split_text(transcript)
            
            # Create documents for each chunk
            documents = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    **full_metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                doc = Document(page_content=chunk, metadata=chunk_metadata)
                documents.append(doc)
            
            # Add to vectorstore
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            else:
                new_vectorstore = FAISS.from_documents(documents, self.embeddings)
                self.vectorstore.merge_from(new_vectorstore)
            
            # Store transcript metadata
            self.transcript_metadata[doc_id] = full_metadata
            
            # Save to disk
            self.save_vectorstore()
            
            logger.info(f"Added transcript to unified vectorstore: {doc_id} ({len(chunks)} chunks)")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding transcript to vectorstore: {e}")
            raise
    
    def add_documents(self, documents: List[Document]):
        """Add documents to unified vectorstore"""
        try:
            if not documents or not self.embeddings:
                return
            
            # Add document type to metadata
            for doc in documents:
                doc.metadata["doc_type"] = "document"
                doc.metadata["added_at"] = datetime.now().isoformat()
                doc.metadata["embedding_type"] = self.embedding_type
            
            # Add to vectorstore
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            else:
                new_vectorstore = FAISS.from_documents(documents, self.embeddings)
                self.vectorstore.merge_from(new_vectorstore)
            
            # Save to disk
            self.save_vectorstore()
            
            logger.info(f"Added {len(documents)} documents to unified vectorstore")
            
        except Exception as e:
            logger.error(f"Error adding documents to vectorstore: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar content in vectorstore"""
        try:
            if self.vectorstore is None or not self.embeddings:
                return []
            
            docs = self.vectorstore.similarity_search(query, k=k)
            
            results = []
            for doc in docs:
                result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []
    
    def get_all_transcripts(self) -> List[Dict[str, Any]]:
        """Get all transcript metadata"""
        return list(self.transcript_metadata.values())
    
    def delete_transcript(self, doc_id: str) -> bool:
        """Delete transcript from vectorstore - Note: FAISS doesn't support deletion easily"""
        try:
            if doc_id in self.transcript_metadata:
                # Remove from metadata
                del self.transcript_metadata[doc_id]
                
                # Save metadata (vectorstore will still contain the embeddings)
                metadata_path = os.path.join(self.vectorstore_path, "transcript_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(self.transcript_metadata, f, indent=2)
                
                logger.info(f"Removed transcript metadata: {doc_id}")
                logger.warning("Note: FAISS vectorstore still contains embeddings - full rebuild needed for complete removal")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting transcript: {e}")
            return False
    
    def get_retriever(self):
        """Get retriever for the vectorstore"""
        if self.vectorstore is None:
            return None
        return self.vectorstore.as_retriever()
    
    def get_embedding_type(self):
        """Get the type of embeddings being used"""
        return self.embedding_type

class UnifiedChatBot:
    """Chat interface for unified vector database using Gemini API"""
    
    def __init__(self, unified_vectorstore: UnifiedVectorStore, gemini_client=None):
        self.unified_vectorstore = unified_vectorstore
        self.gemini_client = gemini_client
        self.conversation_history = []
    
    def chat(self, user_question: str, n_context_results: int = 5) -> str:
        search_results = self.unified_vectorstore.similarity_search(user_question, n_context_results)
        
        context_chunks = []
        for result in search_results:
            metadata = result['metadata']
            doc_type = metadata.get('doc_type', 'unknown')
            source_info = f"[{doc_type.title()}: {metadata.get('filename', 'Unknown')}]"
            context_chunks.append(f"{source_info}\n{result['content']}")
        
        context = "\n\n".join(context_chunks)
        
        system_prompt = """You are an AI assistant that helps users find information from their documents, video/audio transcripts, and meeting recordings. 
        Use the provided context to answer the user's question. If the answer isn't in the context, say so.
        Be conversational and helpful. Always cite which source/file the information came from."""
        
        user_prompt = f"""Context from documents and transcripts:
        {context}
        
        User question: {user_question}
        
        Please answer based on the context provided above."""
        
        self.conversation_history.append({"role": "user", "content": user_question})
        
        # Prepare conversation history for Gemini
        conversation_parts = []
        conversation_parts.append(f"{system_prompt}\n\n")
        
        # Add recent conversation history (last 6 exchanges)
        if len(self.conversation_history) > 1:
            recent_history = self.conversation_history[-6:]
            for msg in recent_history:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    conversation_parts.append(f"User: {content}\n")
                elif role == "assistant":
                    conversation_parts.append(f"Assistant: {content}\n")
        
        # Add current user question
        conversation_parts.append(f"User: {user_question}\n")
        conversation_parts.append("Assistant: ")
        
        full_prompt = "".join(conversation_parts)
        
        try:
            if self.gemini_client:
                # Use Gemini API
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(full_prompt)
                
                if response.text:
                    ai_response = response.text.strip()
                else:
                    ai_response = "I apologize, but I couldn't generate a response at this time."
                
            else:
                # Fallback response if Gemini is not available
                ai_response = "I apologize, but the AI service is currently unavailable. Please check your Gemini API configuration."
            
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error getting AI response: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"

class MeetingBaasClient:
    """Client for MeetingBaas API integration"""
    
    def __init__(self, api_key, base_url="https://api.meetingbaas.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "x-meeting-baas-api-key": api_key,
        }
        self.active_bots = {}  # Track active bots
        logger.info("MeetingBaas client initialized")
    
    def start_recording(self, meeting_url, bot_name="AI Notetaker", language="en", 
                       recording_mode="speaker_view", bot_image=None, entry_message=None):
        """Start recording a meeting"""
        try:
            config = {
                "meeting_url": meeting_url,
                "bot_name": bot_name,
                "recording_mode": recording_mode,
                "reserved": False,
                "speech_to_text": {
                    "provider": "Default",
                    "language": language
                },
                "automatic_leave": {
                    "waiting_room_timeout": 600  # 10 minutes in seconds
                }
            }
            
            if bot_image:
                config["bot_image"] = bot_image
            
            if entry_message:
                config["entry_message"] = entry_message
            else:
                config["entry_message"] = f"Hi! I'm {bot_name}, an AI assistant here to record this meeting."
            
            logger.info(f"Starting meeting recording for: {meeting_url}")
            response = requests.post(
                f"{self.base_url}/bots",
                json=config,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                bot_id = result.get('bot_id')
                
                # Store bot info
                self.active_bots[bot_id] = {
                    'meeting_url': meeting_url,
                    'bot_name': bot_name,
                    'language': language,
                    'status': 'starting',
                    'start_time': datetime.now().isoformat(),
                    'config': config
                }
                
                logger.info(f"Meeting recording started successfully. Bot ID: {bot_id}")
                return result
            else:
                error_msg = f"Failed to start recording: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {"error": error_msg}
                
        except Exception as e:
            error_msg = f"Error starting meeting recording: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_bot_status(self, bot_id):
        """Get status of a specific bot"""
        try:
            logger.info(f"Getting status for bot: {bot_id}")
            response = requests.get(
                f"{self.base_url}/bots/{bot_id}",
                headers=self.headers,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Update stored bot info
                if bot_id in self.active_bots:
                    self.active_bots[bot_id]['status'] = result.get('status', 'unknown')
                    self.active_bots[bot_id]['last_updated'] = datetime.now().isoformat()
                
                logger.info(f"Bot {bot_id} status: {result.get('status', 'unknown')}")
                return result
            else:
                error_msg = f"Failed to get bot status: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {"error": error_msg}
                
        except Exception as e:
            error_msg = f"Error getting bot status: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def stop_recording(self, bot_id):
        """Stop recording for a specific bot"""
        try:
            logger.info(f"Stopping recording for bot: {bot_id}")
            response = requests.delete(
                f"{self.base_url}/bots/{bot_id}",
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Update bot status
                if bot_id in self.active_bots:
                    self.active_bots[bot_id]['status'] = 'stopped'
                    self.active_bots[bot_id]['end_time'] = datetime.now().isoformat()
                
                logger.info(f"Recording stopped for bot: {bot_id}")
                return result
            else:
                error_msg = f"Failed to stop recording: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {"error": error_msg}
                
        except Exception as e:
            error_msg = f"Error stopping recording: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_recording(self, bot_id):
        """Get the recording file for a bot"""
        try:
            logger.info(f"Getting recording for bot: {bot_id}")
            response = requests.get(
                f"{self.base_url}/bots/{bot_id}/recording",
                headers=self.headers,
                timeout=60
            )
            
            if response.status_code == 200:
                # Save the recording file
                filename = f"meeting_recording_{bot_id}_{int(time.time())}.mp4"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"Recording saved: {filepath}")
                return {
                    "success": True,
                    "filepath": filepath,
                    "filename": filename,
                    "size": len(response.content)
                }
            else:
                error_msg = f"Failed to get recording: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {"error": error_msg}
                
        except Exception as e:
            error_msg = f"Error getting recording: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_transcript(self, bot_id):
        """Get the transcript for a bot"""
        try:
            logger.info(f"Getting transcript for bot: {bot_id}")
            response = requests.get(
                f"{self.base_url}/bots/{bot_id}/transcript",
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Transcript retrieved for bot: {bot_id}")
                return result
            else:
                error_msg = f"Failed to get transcript: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {"error": error_msg}
                
        except Exception as e:
            error_msg = f"Error getting transcript: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def list_active_bots(self):
        """List all active bots"""
        return self.active_bots
    
    def remove_bot_from_active(self, bot_id):
        """Remove bot from active list"""
        if bot_id in self.active_bots:
            del self.active_bots[bot_id]
            logger.info(f"Removed bot {bot_id} from active list")

class GoogleDriveUploader:
    """Handle Google Drive uploads"""
    
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    
    def __init__(self, credentials_file='credentials.json', token_file='token.pickle'):
        if not GOOGLE_DRIVE_AVAILABLE:
            raise ImportError("Google Drive API libraries not installed")
        
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = self._authenticate()
    
    def _authenticate(self):
        creds = None
        
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_file):
                    raise FileNotFoundError(f"Google Cloud credentials file not found: {self.credentials_file}")
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, self.SCOPES)
                creds = flow.run_local_server(port=8000)
            
            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)
        
        return build('drive', 'v3', credentials=creds)
    
    def upload_text_file(self, content, filename, folder_id=None):
        try:
            file_metadata = {
                'name': filename,
                'mimeType': 'text/plain'
            }
            
            if folder_id:
                file_metadata['parents'] = [folder_id]
            
            media = MediaIoBaseUpload(
                BytesIO(content.encode('utf-8')), 
                mimetype='text/plain'
            )
            
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id,name,webViewLink'
            ).execute()
            
            logger.info(f"File uploaded to Google Drive: {file.get('name')}")
            return file.get('id')
            
        except Exception as e:
            logger.error(f"Error uploading to Google Drive: {str(e)}")
            raise

class VideoTranscriber:
    """Enhanced video transcriber with customer support call analysis capabilities"""
    
    def __init__(self, openai_api_key, gemini_api_key=None, google_drive_uploader=None, unified_vectorstore=None):
        # Initialize OpenAI client for Whisper transcription
        try:
            # Try different initialization methods for compatibility
            try:
                self.openai_client = OpenAI(api_key=openai_api_key)
            except TypeError:
                # Fallback for older versions
                self.openai_client = OpenAI(api_key=openai_api_key, proxies=None)
            logger.info("OpenAI client initialized for Whisper transcription")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise
        
        # Initialize Gemini client for chat/completion and enhanced analysis
        self.gemini_client = None
        if gemini_api_key and GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=gemini_api_key)
                self.gemini_client = genai
                logger.info("Gemini client initialized for chat/completion and enhanced analysis")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {str(e)}")
                # Don't raise here, as we can still use OpenAI for transcription
        
        self.drive_uploader = google_drive_uploader
        self.unified_vectorstore = unified_vectorstore
        
        # Initialize enhanced analysis components
        self.insights_extractor = None
        self.conversation_summarizer = None
        self.support_agent = None
        self.call_evaluator = None
        self.call_metadata = CallMetadata()
        self.speaker_diarizer = None
        
        # Initialize enhanced components if Gemini is available
        if self.gemini_client:
            try:
                self.insights_extractor = InsightsExtractor(gemini_client=self.gemini_client)
                self.conversation_summarizer = ConversationSummarizer(gemini_client=self.gemini_client)
                self.support_agent = SupportAgent(gemini_client=self.gemini_client, vectorstore=unified_vectorstore)
                self.call_evaluator = CallEvaluator(gemini_client=self.gemini_client)
                logger.info("Enhanced analysis components initialized successfully")
            except Exception as e:
                logger.warning(f"Some enhanced components failed to initialize: {str(e)}")
        
        # Initialize speaker diarization if available
        try:
            hf_token = os.getenv("HF_TOKEN")
            if DIARIZATION_AVAILABLE and hf_token:
                self.speaker_diarizer = SpeakerDiarizer(hf_token=hf_token)
                logger.info("Speaker diarization initialized for customer support analysis")
            else:
                logger.warning("Speaker diarization not available (requires HF_TOKEN and pyannote.audio)")
        except Exception as e:
            logger.warning(f"Speaker diarization failed to initialize: {str(e)}")
        
        # Initialize chat bot for basic RAG functionality
        if self.unified_vectorstore:
            self.chat_bot = UnifiedChatBot(self.unified_vectorstore, self.gemini_client)
        else:
            self.chat_bot = None
    
    def extract_audio_from_video(self, video_path, output_audio_path=None):
        try:
            video_path = os.path.normpath(video_path)
            logger.info(f"Processing file: {video_path}")
            
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"File not found: {video_path}")
            
            if output_audio_path is None:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_audio_path = os.path.join(os.path.dirname(video_path), f"{base_name}_audio.mp3")
                output_audio_path = os.path.normpath(output_audio_path)
            
            file_extension = os.path.splitext(video_path)[1].lower()
            
            # Check if it's already an audio file
            audio_extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
            
            if file_extension in audio_extensions:
                logger.info(f"File is already an audio file: {file_extension}")
                
                if USE_MOVIEPY:
                    # Use MoviePy's AudioFileClip for audio files
                    logger.info("Using moviepy AudioFileClip to process audio")
                    
                    try:
                        audio = AudioFileClip(video_path)
                        logger.info(f"Audio duration: {audio.duration} seconds")
                        audio.write_audiofile(output_audio_path, verbose=False, logger=None)
                        audio.close()
                    except Exception as moviepy_error:
                        logger.warning(f"MoviePy failed for audio file: {moviepy_error}")
                        # Fallback to pydub for audio files
                        try:
                            from pydub import AudioSegment
                            logger.info("Falling back to pydub for audio processing")
                            audio = AudioSegment.from_file(video_path)
                            audio.export(output_audio_path, format="mp3")
                        except ImportError:
                            raise ImportError("Both moviepy and pydub failed. Please install at least one of them.")
                else:
                    # Use pydub for audio files
                    logger.info("Using pydub to process audio file")
                    try:
                        from pydub import AudioSegment
                        audio = AudioSegment.from_file(video_path)
                        audio.export(output_audio_path, format="mp3")
                    except ImportError:
                        raise ImportError("Pydub not available. Please install with: pip install pydub")
            
            elif file_extension in video_extensions:
                logger.info(f"File is a video file: {file_extension}")
                
                if USE_MOVIEPY:
                    # Use MoviePy's VideoFileClip for video files
                    logger.info("Using moviepy VideoFileClip to extract audio from video")
                    
                    video = VideoFileClip(video_path)
                    
                    if video.audio is None:
                        raise ValueError("Video file has no audio track")
                    
                    audio = video.audio
                    logger.info(f"Video duration: {video.duration} seconds")
                    logger.info(f"Extracting audio to: {output_audio_path}")
                    
                    audio.write_audiofile(output_audio_path, verbose=False, logger=None)
                    
                    audio.close()
                    video.close()
                else:
                    raise ImportError("Video file detected but moviepy is not installed. Please install moviepy for video processing.")
            
            else:
                raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: {audio_extensions + video_extensions}")
            
            logger.info("Audio extraction completed successfully")
            return output_audio_path
            
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            raise

    def transcribe_audio(self, audio_path, language=None, prompt=None):
        try:
            logger.info(f"Starting transcription for: {audio_path}")
            
            # Check file size
            file_size = os.path.getsize(audio_path)
            max_size = 25 * 1024 * 1024  # 25MB limit for OpenAI Whisper
            
            if file_size > max_size:
                raise ValueError(f"Audio file is too large ({file_size / (1024*1024):.1f}MB). OpenAI Whisper API has a 25MB limit.")
            
            logger.info(f"File size: {file_size / (1024*1024):.1f}MB")
            
            with open(audio_path, "rb") as audio_file:
                transcription_params = {
                    "file": audio_file,
                    "model": "whisper-1",
                    "response_format": "text"
                }
                
                if language:
                    transcription_params["language"] = language
                if prompt:
                    transcription_params["prompt"] = prompt
                
                logger.info("Sending request to OpenAI Whisper API...")
                transcript = self.openai_client.audio.transcriptions.create(**transcription_params)
                
                logger.info("Transcription completed successfully")
                logger.info(f"Transcript length: {len(transcript)} characters")
                
                return transcript
                
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            raise

    def process_video_to_transcript(self, video_path, language=None, prompt=None, keep_audio=False, 
                                   upload_to_drive=False, drive_folder_id=None, add_to_vectorstore=True):
        temp_audio_path = None
        
        try:
            logger.info(f"Starting processing for: {video_path}")
            
            # Check if input file exists
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Input file not found: {video_path}")
            
            # Get file info
            file_size = os.path.getsize(video_path)
            file_extension = os.path.splitext(video_path)[1].lower()
            logger.info(f"Input file size: {file_size / (1024*1024):.1f}MB")
            logger.info(f"Input file extension: {file_extension}")
            
            # Extract or prepare audio
            temp_audio_path = self.extract_audio_from_video(video_path)
            
            # Transcribe audio
            transcript = self.transcribe_audio(temp_audio_path, language, prompt)
            
            if not transcript or len(transcript.strip()) == 0:
                raise ValueError("Transcription resulted in empty text")
            
            result = {
                "transcript": transcript,
                "video_path": video_path,
                "input_file_size": file_size,
                "input_file_extension": file_extension
            }
            
            # Save transcript locally
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            transcript_filename = f"{base_name}_transcript.txt"
            local_transcript_path = os.path.join(os.path.dirname(video_path), transcript_filename)
            
            with open(local_transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            
            result["local_transcript_path"] = local_transcript_path
            result["transcript_filename"] = transcript_filename
            logger.info(f"Transcript saved locally: {local_transcript_path}")
            
            # Add to unified vectorstore
            if add_to_vectorstore and self.unified_vectorstore:
                try:
                    metadata = {
                        "filename": os.path.basename(video_path),
                        "file_path": video_path,
                        "transcript_path": local_transcript_path,
                        "language": language or "unknown",
                        "processing_date": datetime.now().isoformat(),
                        "file_size": file_size,
                        "file_extension": file_extension
                    }
                    
                    doc_id = self.unified_vectorstore.add_transcript(transcript, metadata)
                    result["vector_db_id"] = doc_id
                    result["added_to_vectorstore"] = True
                    logger.info(f"Transcript added to unified vectorstore: {doc_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to add to unified vectorstore: {str(e)}")
                    result["vectorstore_error"] = str(e)
                    result["added_to_vectorstore"] = False
            else:
                result["added_to_vectorstore"] = False
                if not self.unified_vectorstore:
                    result["vectorstore_error"] = "Unified vectorstore not initialized"
            
            # Upload to Google Drive
            if upload_to_drive and self.drive_uploader:
                try:
                    file_id = self.drive_uploader.upload_text_file(
                        content=transcript,
                        filename=transcript_filename,
                        folder_id=drive_folder_id
                    )
                    result["drive_file_id"] = file_id
                    result["uploaded_to_drive"] = True
                    logger.info("Transcript uploaded to Google Drive successfully")
                except Exception as e:
                    logger.error(f"Failed to upload to Google Drive: {str(e)}")
                    result["drive_error"] = str(e)
                    result["uploaded_to_drive"] = False
            else:
                result["uploaded_to_drive"] = False
                if not self.drive_uploader:
                    result["drive_error"] = "Google Drive uploader not initialized"
            
            # Handle audio file cleanup
            if keep_audio and temp_audio_path:
                result["audio_path"] = temp_audio_path
                logger.info(f"Audio file preserved: {temp_audio_path}")
            else:
                if temp_audio_path and os.path.exists(temp_audio_path) and temp_audio_path != video_path:
                    try:
                        os.remove(temp_audio_path)
                        logger.info("Temporary audio file removed")
                    except Exception as cleanup_error:
                        logger.warning(f"Could not remove temporary audio file: {cleanup_error}")
            
            logger.info("Processing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in process_video_to_transcript: {str(e)}")
            
            # Cleanup on error
            if temp_audio_path and os.path.exists(temp_audio_path) and temp_audio_path != video_path and not keep_audio:
                try:
                    os.remove(temp_audio_path)
                    logger.info("Cleaned up temporary audio file after error")
                except:
                    pass
            
            raise
    
    def perform_comprehensive_analysis(self, video_path: str, language: str = None, 
                                    enable_diarization: bool = True, 
                                    enable_insights: bool = True,
                                    enable_summary: bool = True,
                                    enable_evaluation: bool = True,
                                    enable_agent_analysis: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive customer support call analysis including:
        - Speech-to-text transcription
        - Speaker diarization (if enabled)
        - Sentiment, tonality, and intent analysis
        - Call summarization
        - Quality evaluation
        - Agent recommendations
        - Metadata generation
        
        Args:
            video_path: Path to the video/audio file
            language: Language for transcription
            enable_diarization: Whether to perform speaker diarization
            enable_insights: Whether to extract insights
            enable_summary: Whether to generate summary
            enable_evaluation: Whether to evaluate call quality
            enable_agent_analysis: Whether to perform agent analysis
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        try:
            if not memory_manager.check_memory_usage():
                raise MemoryError("Insufficient memory for comprehensive analysis")
            
            logger.info(f"Starting comprehensive analysis for: {video_path}")
            
            # Step 1: Basic transcription
            logger.info("Step 1: Performing speech-to-text transcription")
            basic_result = self.process_video_to_transcript(
                video_path=video_path,
                language=language,
                add_to_vectorstore=True,
                upload_to_drive=False,
                keep_audio=True
            )
            
            transcript = basic_result.get("transcript", "")
            audio_path = basic_result.get("audio_path", video_path)
            
            if not transcript:
                raise ValueError("Failed to generate transcript")
            
            # Initialize results structure
            analysis_results = {
                "basic_transcription": basic_result,
                "transcript": transcript,
                "audio_path": audio_path,
                "analysis_timestamp": datetime.now().isoformat(),
                "file_info": {
                    "filename": os.path.basename(video_path),
                    "file_size": os.path.getsize(video_path),
                    "file_type": os.path.splitext(video_path)[1].lower()
                }
            }
            
            # Step 2: Speaker diarization (if enabled and available)
            if enable_diarization and self.speaker_diarizer and audio_path:
                try:
                    logger.info("Step 2: Performing speaker diarization")
                    diarization_result = self.speaker_diarizer.perform_diarization(audio_path)
                    analysis_results["diarization"] = diarization_result
                    
                    # Combine transcript with speaker labels
                    labeled_transcript = self.speaker_diarizer.combine_transcript_with_speakers(
                        transcript, diarization_result
                    )
                    analysis_results["labeled_transcript"] = labeled_transcript
                    
                    logger.info("Speaker diarization completed successfully")
                except Exception as e:
                    logger.warning(f"Speaker diarization failed: {str(e)}")
                    analysis_results["diarization_error"] = str(e)
            else:
                logger.info("Speaker diarization skipped (not enabled or not available)")
            
            # Step 3: Insights extraction (if enabled and available)
            if enable_insights and self.insights_extractor:
                try:
                    logger.info("Step 3: Extracting insights (sentiment, tonality, intent)")
                    insights = self.insights_extractor.extract_all_insights(transcript)
                    analysis_results["insights"] = insights
                    logger.info("Insights extraction completed successfully")
                except Exception as e:
                    logger.warning(f"Insights extraction failed: {str(e)}")
                    analysis_results["insights_error"] = str(e)
            else:
                logger.info("Insights extraction skipped (not enabled or not available)")
            
            # Step 4: Call summarization (if enabled and available)
            if enable_summary and self.conversation_summarizer:
                try:
                    logger.info("Step 4: Generating call summary")
                    insights_for_summary = analysis_results.get("insights", {})
                    summary = self.conversation_summarizer.generate_summary(transcript, insights_for_summary)
                    analysis_results["summary"] = summary
                    
                    # Also generate brief summary and key phrases
                    brief_summary = self.conversation_summarizer.generate_brief_summary(transcript)
                    key_phrases = self.conversation_summarizer.extract_key_phrases(transcript)
                    
                    analysis_results["brief_summary"] = brief_summary
                    analysis_results["key_phrases"] = key_phrases
                    
                    logger.info("Call summarization completed successfully")
                except Exception as e:
                    logger.warning(f"Call summarization failed: {str(e)}")
                    analysis_results["summary_error"] = str(e)
            else:
                logger.info("Call summarization skipped (not enabled or not available)")
            
            # Step 5: Call quality evaluation (if enabled and available)
            if enable_evaluation and self.call_evaluator:
                try:
                    logger.info("Step 5: Evaluating call quality")
                    insights_for_evaluation = analysis_results.get("insights", {})
                    summary_for_evaluation = analysis_results.get("summary", {})
                    
                    evaluation = self.call_evaluator.evaluate_call_quality(
                        transcript, insights_for_evaluation, summary_for_evaluation
                    )
                    analysis_results["evaluation"] = evaluation
                    
                    # Calculate performance metrics
                    performance_metrics = self.call_evaluator.calculate_performance_metrics(transcript)
                    analysis_results["performance_metrics"] = performance_metrics
                    
                    logger.info("Call quality evaluation completed successfully")
                except Exception as e:
                    logger.warning(f"Call quality evaluation failed: {str(e)}")
                    analysis_results["evaluation_error"] = str(e)
            else:
                logger.info("Call quality evaluation skipped (not enabled or not available)")
            
            # Step 6: Agent analysis and recommendations (if enabled and available)
            if enable_agent_analysis and self.support_agent:
                try:
                    logger.info("Step 6: Performing agent analysis and generating recommendations")
                    insights_for_agent = analysis_results.get("insights", {})
                    summary_for_agent = analysis_results.get("summary", {})
                    
                    agent_analysis = self.support_agent.analyze_call_for_actions(
                        transcript, insights_for_agent, summary_for_agent
                    )
                    analysis_results["agent_analysis"] = agent_analysis
                    
                    # Find similar cases
                    similar_cases = self.support_agent.suggest_similar_cases(transcript)
                    analysis_results["similar_cases"] = similar_cases
                    
                    logger.info("Agent analysis completed successfully")
                except Exception as e:
                    logger.warning(f"Agent analysis failed: {str(e)}")
                    analysis_results["agent_analysis_error"] = str(e)
            else:
                logger.info("Agent analysis skipped (not enabled or not available)")
            
            # Step 7: Generate comprehensive metadata
            try:
                logger.info("Step 7: Generating comprehensive metadata")
                metadata = self.call_metadata.create_call_metadata(
                    analysis_results["file_info"], analysis_results
                )
                analysis_results["metadata"] = metadata
                logger.info("Metadata generation completed successfully")
            except Exception as e:
                logger.warning(f"Metadata generation failed: {str(e)}")
                analysis_results["metadata_error"] = str(e)
            
            # Step 8: Calculate overall analysis score
            try:
                analysis_score = self._calculate_overall_analysis_score(analysis_results)
                analysis_results["overall_analysis_score"] = analysis_score
                logger.info(f"Overall analysis score: {analysis_score}/100")
            except Exception as e:
                logger.warning(f"Analysis score calculation failed: {str(e)}")
                analysis_results["analysis_score_error"] = str(e)
            
            # Cleanup memory
            memory_manager.force_cleanup()
            
            logger.info("Comprehensive analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            raise
    
    def _calculate_overall_analysis_score(self, analysis_results: Dict) -> float:
        """
        Calculate overall analysis score based on completed components
        
        Args:
            analysis_results: Results from comprehensive analysis
            
        Returns:
            Overall score (0-100)
        """
        try:
            score = 0
            max_score = 0
            
            # Basic transcription (required)
            if analysis_results.get("transcript"):
                score += 20
                max_score += 20
            
            # Speaker diarization
            if analysis_results.get("diarization"):
                score += 15
                max_score += 15
            
            # Insights extraction
            if analysis_results.get("insights"):
                score += 20
                max_score += 20
            
            # Call summary
            if analysis_results.get("summary"):
                score += 15
                max_score += 15
            
            # Quality evaluation
            if analysis_results.get("evaluation"):
                score += 15
                max_score += 15
            
            # Agent analysis
            if analysis_results.get("agent_analysis"):
                score += 15
                max_score += 15
            
            # Calculate percentage
            if max_score > 0:
                return (score / max_score) * 100
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Error calculating analysis score: {str(e)}")
            return 50  # Default score

# Speaker diarization class for customer support call analysis
class SpeakerDiarizer:
    """Speaker diarization for separating customer and agent speech in support calls"""
    
    def __init__(self, hf_token=None):
        if not DIARIZATION_AVAILABLE:
            raise ImportError("Pyannote.audio not available. Install with: pip install pyannote.audio")
        
        self.hf_token = hf_token
        self.pipeline = None
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the speaker diarization pipeline with memory management"""
        try:
            if not memory_manager.check_memory_usage():
                raise MemoryError("Insufficient memory for diarization pipeline")
            
            # Initialize pipeline with memory-efficient settings
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization@2.1",
                use_auth_token=self.hf_token
            )
            
            # Configure for memory efficiency
            self.pipeline.instantiate({
                "segmentation": {
                    "min_duration_off": 0.5,  # Minimum silence duration
                    "min_duration_on": 0.5,   # Minimum speech duration
                },
                "clustering": {
                    "method": "centroid",
                    "min_clusters": 2,  # Expect at least 2 speakers (customer + agent)
                    "max_clusters": 4,  # Maximum 4 speakers
                }
            })
            
            logger.info("Speaker diarization pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize diarization pipeline: {str(e)}")
            raise
    
    def perform_diarization(self, audio_file: str) -> Dict[str, Any]:
        """
        Perform speaker diarization on customer support call audio
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            Dictionary containing speaker segments and metadata
        """
        try:
            if not memory_manager.check_memory_usage():
                raise MemoryError("Insufficient memory for diarization")
            
            logger.info(f"Starting speaker diarization for: {audio_file}")
            
            # Perform diarization
            diarization = self.pipeline(audio_file)
            
            # Extract speaker segments
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = {
                    "speaker": speaker,
                    "start_time": turn.start,
                    "end_time": turn.end,
                    "duration": turn.end - turn.start
                }
                speaker_segments.append(segment)
            
            # Analyze speaker patterns (customer vs agent identification)
            speaker_analysis = self._analyze_speaker_patterns(speaker_segments)
            
            result = {
                "speaker_segments": speaker_segments,
                "speaker_analysis": speaker_analysis,
                "total_speakers": len(set(seg["speaker"] for seg in speaker_segments)),
                "total_duration": sum(seg["duration"] for seg in speaker_segments)
            }
            
            logger.info(f"Diarization completed: {result['total_speakers']} speakers identified")
            
            # Cleanup memory
            memory_manager.force_cleanup()
            
            return result
            
        except Exception as e:
            logger.error(f"Error during speaker diarization: {str(e)}")
            raise
    
    def _analyze_speaker_patterns(self, speaker_segments: List[Dict]) -> Dict[str, Any]:
        """
        Analyze speaker patterns to identify customer vs agent
        
        Args:
            speaker_segments: List of speaker segments from diarization
            
        Returns:
            Analysis of speaker patterns and roles
        """
        try:
            # Group segments by speaker
            speaker_stats = {}
            for segment in speaker_segments:
                speaker = segment["speaker"]
                if speaker not in speaker_stats:
                    speaker_stats[speaker] = {
                        "total_duration": 0,
                        "segment_count": 0,
                        "avg_duration": 0
                    }
                
                speaker_stats[speaker]["total_duration"] += segment["duration"]
                speaker_stats[speaker]["segment_count"] += 1
            
            # Calculate averages
            for speaker in speaker_stats:
                stats = speaker_stats[speaker]
                stats["avg_duration"] = stats["total_duration"] / stats["segment_count"]
            
            # Identify likely customer vs agent based on speaking patterns
            speakers = list(speaker_stats.keys())
            if len(speakers) >= 2:
                # Sort by total duration (agent usually speaks more)
                speakers.sort(key=lambda s: speaker_stats[s]["total_duration"], reverse=True)
                
                # First speaker (most talkative) is likely the agent
                agent_speaker = speakers[0]
                customer_speaker = speakers[1] if len(speakers) > 1 else None
                
                analysis = {
                    "agent_speaker": agent_speaker,
                    "customer_speaker": customer_speaker,
                    "speaker_stats": speaker_stats,
                    "confidence": "high" if len(speakers) == 2 else "medium"
                }
            else:
                analysis = {
                    "agent_speaker": None,
                    "customer_speaker": None,
                    "speaker_stats": speaker_stats,
                    "confidence": "low"
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing speaker patterns: {str(e)}")
            return {"error": str(e)}
    
    def combine_transcript_with_speakers(self, transcript: str, diarization_result: Dict) -> str:
        """
        Combine transcript with speaker labels for customer support analysis
        
        Args:
            transcript: Raw transcript from Whisper
            diarization_result: Result from speaker diarization
            
        Returns:
            Speaker-labeled transcript
        """
        try:
            # This is a simplified version - in practice, you'd need more sophisticated
            # alignment between transcript timestamps and speaker segments
            
            speaker_segments = diarization_result.get("speaker_segments", [])
            speaker_analysis = diarization_result.get("speaker_analysis", {})
            
            # Create speaker-labeled transcript
            labeled_transcript = f"Speaker Analysis:\n"
            labeled_transcript += f"Agent Speaker: {speaker_analysis.get('agent_speaker', 'Unknown')}\n"
            labeled_transcript += f"Customer Speaker: {speaker_analysis.get('customer_speaker', 'Unknown')}\n\n"
            
            labeled_transcript += "Speaker Segments:\n"
            for segment in speaker_segments:
                speaker = segment["speaker"]
                start_time = segment["start_time"]
                end_time = segment["end_time"]
                
                # Determine if this is likely agent or customer
                if speaker == speaker_analysis.get("agent_speaker"):
                    speaker_label = "Agent"
                elif speaker == speaker_analysis.get("customer_speaker"):
                    speaker_label = "Customer"
                else:
                    speaker_label = f"Speaker_{speaker}"
                
                labeled_transcript += f"[{start_time:.1f}s - {end_time:.1f}s] {speaker_label}: [Speech segment]\n"
            
            labeled_transcript += f"\nFull Transcript:\n{transcript}"
            
            return labeled_transcript
            
        except Exception as e:
            logger.error(f"Error combining transcript with speakers: {str(e)}")
            return transcript  # Return original transcript if combination fails

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize services
def initialize_services():
    global transcriber, unified_vectorstore, meeting_baas_client
    
    # Get API keys from environment
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MEETINGBAAS_API_KEY = os.getenv("MEETINGBAAS_API_KEY")
    
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not found in environment variables")
        return False
    
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not found in environment variables. Chat functionality will be limited.")
    
    if not MEETINGBAAS_API_KEY:
        logger.warning("MEETINGBAAS_API_KEY not found in environment variables. Meeting recording will be disabled.")
    
    # Initialize MeetingBaas client
    if MEETINGBAAS_API_KEY:
        try:
            meeting_baas_client = MeetingBaasClient(MEETINGBAAS_API_KEY)
            logger.info("MeetingBaas client initialized successfully")
        except Exception as e:
            logger.error(f"Could not initialize MeetingBaas client: {str(e)}")
            meeting_baas_client = None
    else:
        meeting_baas_client = None
    
    # Initialize unified vectorstore
    try:
        if not GEMINI_AVAILABLE:
            logger.error("Google Generative AI not available. Install with: pip install google-generativeai")
            unified_vectorstore = None
        elif not GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY is required for embeddings")
            unified_vectorstore = None
        else:
            unified_vectorstore = UnifiedVectorStore(vectorstore_path="./unified_vectorstore", gemini_api_key=GEMINI_API_KEY)
            logger.info("Unified vectorstore initialized successfully with Gemini embeddings")
    except Exception as e:
        logger.error(f"Could not initialize unified vectorstore: {str(e)}")
        unified_vectorstore = None
    
    # Initialize Google Drive uploader
    drive_uploader = None
    if GOOGLE_DRIVE_AVAILABLE:
        try:
            drive_uploader = GoogleDriveUploader('credentials.json')
            logger.info("Google Drive uploader initialized")
        except Exception as e:
            logger.warning(f"Could not initialize Google Drive uploader: {str(e)}")
            drive_uploader = None
    
    # Initialize transcriber
    try:
        transcriber = VideoTranscriber(OPENAI_API_KEY, GEMINI_API_KEY, drive_uploader, unified_vectorstore)
        logger.info("Video transcriber initialized successfully")
    except Exception as e:
        logger.error(f"Could not initialize transcriber: {str(e)}")
        transcriber = None
    
    return True

# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/api/upload', methods=['POST'])
def upload_video():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if file and allowed_file(file.filename):
            # Check if transcriber is available before processing
            if not transcriber:
                return jsonify({"error": "Transcriber service not available. Please check server logs and ensure OpenAI API key is valid for Whisper transcription."}), 500
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the uploaded file
            try:
                file.save(filepath)
                logger.info(f"File saved to: {filepath}")
                
                # Check file size
                file_size = os.path.getsize(filepath)
                logger.info(f"File size: {file_size / (1024*1024):.1f}MB")
                
                # Check if file is too large (we'll let the transcriber handle the 25MB limit)
                if file_size > MAX_CONTENT_LENGTH:
                    os.remove(filepath)
                    return jsonify({"error": f"File too large. Maximum size is {MAX_CONTENT_LENGTH / (1024*1024):.0f}MB"}), 413
                
            except Exception as save_error:
                logger.error(f"Error saving file: {str(save_error)}")
                return jsonify({"error": f"Failed to save file: {str(save_error)}"}), 500
            
            # Get optional parameters
            language = request.form.get('language', 'en')
            upload_to_drive = request.form.get('upload_to_drive', 'false').lower() == 'true'
            keep_audio = request.form.get('keep_audio', 'false').lower() == 'true'
            
            # Process the file
            try:
                logger.info(f"Starting processing for: {filename}")
                logger.info(f"Parameters: language={language}, upload_to_drive={upload_to_drive}, keep_audio={keep_audio}")
                
                result = transcriber.process_video_to_transcript(
                    video_path=filepath,
                    language=language,
                    upload_to_drive=upload_to_drive,
                    add_to_vectorstore=True if unified_vectorstore else False,
                    keep_audio=keep_audio
                )
                
                # Clean up the uploaded file after processing (unless it's kept as audio)
                if not keep_audio or result.get("audio_path") != filepath:
                    try:
                        os.remove(filepath)
                        logger.info(f"Cleaned up uploaded file: {filepath}")
                    except Exception as cleanup_error:
                        logger.warning(f"Could not clean up file {filepath}: {str(cleanup_error)}")
                
                return jsonify({
                    "message": "File processed successfully",
                    "filename": filename,
                    "result": result,
                    "processing": False,
                    "success": True
                })
                
            except Exception as process_error:
                logger.error(f"Error processing file {filename}: {str(process_error)}")
                
                # Clean up the uploaded file on error
                try:
                    os.remove(filepath)
                except:
                    pass
                
                return jsonify({
                    "error": f"Processing failed: {str(process_error)}",
                    "filename": filename,
                    "success": False
                }), 500
        
        return jsonify({"error": "Invalid file type. Supported formats: " + ", ".join(ALLOWED_EXTENSIONS)}), 400
        
    except Exception as e:
        logger.error(f"Error in upload endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Meeting Recording Endpoints
@app.route('/api/meeting/start', methods=['POST'])
def start_meeting_recording():
    try:
        if not meeting_baas_client:
            return jsonify({"error": "MeetingBaas service not available. Please check API key configuration."}), 500
        
        data = request.get_json()
        if not data or 'meeting_url' not in data:
            return jsonify({"error": "Meeting URL is required"}), 400
        
        meeting_url = data['meeting_url']
        bot_name = data.get('bot_name', 'AI Notetaker')
        language = data.get('language', 'en')
        bot_image = data.get('bot_image')
        entry_message = data.get('entry_message')
        
        result = meeting_baas_client.start_recording(
            meeting_url=meeting_url,
            bot_name=bot_name,
            language=language,
            bot_image=bot_image,
            entry_message=entry_message
        )
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify({
            "message": "Meeting recording started successfully",
            "bot_id": result.get('bot_id'),
            "meeting_url": meeting_url,
            "bot_name": bot_name,
            "status": "starting",
            "result": result
        })
        
    except Exception as e:
        logger.error(f"Error starting meeting recording: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/meeting/status/<bot_id>', methods=['GET'])
def get_meeting_status(bot_id):
    try:
        if not meeting_baas_client:
            return jsonify({"error": "MeetingBaas service not available"}), 500
        
        result = meeting_baas_client.get_bot_status(bot_id)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting meeting status: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/meeting/stop/<bot_id>', methods=['POST'])
def stop_meeting_recording(bot_id):
    try:
        if not meeting_baas_client:
            return jsonify({"error": "MeetingBaas service not available"}), 500
        
        # Stop the recording
        stop_result = meeting_baas_client.stop_recording(bot_id)
        
        if 'error' in stop_result:
            return jsonify(stop_result), 500
        
        # Try to get the recording and process it
        try:
            # Wait a moment for the recording to be ready
            time.sleep(2)
            
            recording_result = meeting_baas_client.get_recording(bot_id)
            
            if 'error' not in recording_result and recording_result.get('success'):
                # Process the recording file for transcription
                if transcriber and unified_vectorstore:
                    try:
                        filepath = recording_result['filepath']
                        
                        # Get bot info for metadata
                        bot_info = meeting_baas_client.active_bots.get(bot_id, {})
                        language = bot_info.get('language', 'en')
                        
                        # Process the recording
                        transcript_result = transcriber.process_video_to_transcript(
                            video_path=filepath,
                            language=language,
                            add_to_vectorstore=True,
                            keep_audio=False
                        )
                        
                        logger.info(f"Meeting recording processed and transcribed: {bot_id}")
                        
                        # Remove from active bots
                        meeting_baas_client.remove_bot_from_active(bot_id)
                        
                        return jsonify({
                            "message": "Meeting stopped and processed successfully",
                            "bot_id": bot_id,
                            "recording": recording_result,
                            "transcript": transcript_result,
                            "stop_result": stop_result
                        })
                        
                    except Exception as process_error:
                        logger.error(f"Error processing meeting recording: {str(process_error)}")
                        return jsonify({
                            "message": "Meeting stopped but processing failed",
                            "bot_id": bot_id,
                            "recording": recording_result,
                            "processing_error": str(process_error),
                            "stop_result": stop_result
                        })
                else:
                    return jsonify({
                        "message": "Meeting stopped, recording retrieved but transcription not available",
                        "bot_id": bot_id,
                        "recording": recording_result,
                        "stop_result": stop_result
                    })
            else:
                return jsonify({
                    "message": "Meeting stopped but recording retrieval failed",
                    "bot_id": bot_id,
                    "recording_error": recording_result.get('error', 'Unknown error'),
                    "stop_result": stop_result
                })
                
        except Exception as recording_error:
            logger.error(f"Error retrieving meeting recording: {str(recording_error)}")
            return jsonify({
                "message": "Meeting stopped but recording retrieval failed",
                "bot_id": bot_id,
                "recording_error": str(recording_error),
                "stop_result": stop_result
            })
        
    except Exception as e:
        logger.error(f"Error stopping meeting recording: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/meeting/active', methods=['GET'])
def get_active_meetings():
    try:
        if not meeting_baas_client:
            return jsonify({"error": "MeetingBaas service not available"}), 500
        
        active_bots = meeting_baas_client.list_active_bots()
        
        return jsonify({
            "meetings": active_bots,
            "count": len(active_bots),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting active meetings: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/meeting/recording/<bot_id>', methods=['GET'])
def get_meeting_recording(bot_id):
    try:
        if not meeting_baas_client:
            return jsonify({"error": "MeetingBaas service not available"}), 500
        
        result = meeting_baas_client.get_recording(bot_id)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting meeting recording: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/meeting/transcript/<bot_id>', methods=['GET'])
def get_meeting_transcript(bot_id):
    try:
        if not meeting_baas_client:
            return jsonify({"error": "MeetingBaas service not available"}), 500
        
        result = meeting_baas_client.get_transcript(bot_id)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting meeting transcript: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/transcripts', methods=['GET'])
def get_transcripts():
    try:
        if not unified_vectorstore:
            return jsonify({"error": "Unified vectorstore not available"}), 500
        
        try:
            transcripts = unified_vectorstore.get_all_transcripts()
            return jsonify({
                "transcripts": transcripts,
                "count": len(transcripts),
                "timestamp": datetime.now().isoformat()
            })
        except Exception as db_error:
            logger.error(f"Database error in get_transcripts: {str(db_error)}")
            return jsonify({
                "transcripts": [],
                "count": 0,
                "error": f"Database error: {str(db_error)}",
                "timestamp": datetime.now().isoformat()
            })
    
    except Exception as e:
        logger.error(f"Error getting transcripts: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat_with_unified_data():
    try:
        if not transcriber:
            return jsonify({"error": "Transcriber service not available"}), 500
        
        if not transcriber.chat_bot:
            return jsonify({"error": "Chat bot not available - unified vectorstore may not be initialized or Gemini API not configured"}), 500
        
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Question is required"}), 400
        
        question = data['question']
        
        # Check if there are any embeddings to chat with
        if not unified_vectorstore or not unified_vectorstore.vectorstore:
            return jsonify({
                "question": question,
                "response": "I don't have any documents or transcripts to chat about yet. Please upload some files or record a meeting first.",
                "timestamp": datetime.now().isoformat()
            })
        
        try:
            response = transcriber.chat_bot.chat(question)
            
            return jsonify({
                "question": question,
                "response": response,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as chat_error:
            logger.error(f"Chat processing error: {str(chat_error)}")
            return jsonify({
                "question": question,
                "response": f"I encountered an error while processing your question: {str(chat_error)}",
                "timestamp": datetime.now().isoformat()
            })
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": f"Chat service error: {str(e)}"}), 500

@app.route('/api/search', methods=['POST'])
def search_unified_data():
    try:
        if not unified_vectorstore:
            return jsonify({"error": "Unified vectorstore not available"}), 500
        
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Query is required"}), 400
        
        query = data['query']
        n_results = data.get('n_results', 5)
        
        # Check if there are any embeddings to search
        if not unified_vectorstore.vectorstore:
            return jsonify({
                "query": query,
                "results": [],
                "message": "No data available to search. Please upload some files or record a meeting first.",
                "timestamp": datetime.now().isoformat()
            })
        
        try:
            results = unified_vectorstore.similarity_search(query, n_results)
            
            return jsonify({
                "query": query,
                "results": results,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as search_error:
            logger.error(f"Search error: {str(search_error)}")
            return jsonify({
                "query": query,
                "results": [],
                "error": f"Search error: {str(search_error)}",
                "timestamp": datetime.now().isoformat()
            })
    
    except Exception as e:
        logger.error(f"Error searching unified data: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/transcript/<doc_id>', methods=['DELETE'])
def delete_transcript(doc_id):
    try:
        if not unified_vectorstore:
            return jsonify({"error": "Unified vectorstore not available"}), 500
        
        success = unified_vectorstore.delete_transcript(doc_id)
        
        if success:
            return jsonify({"message": f"Transcript {doc_id} deleted successfully"})
        else:
            return jsonify({"error": f"Transcript {doc_id} not found"}), 404
    
    except Exception as e:
        logger.error(f"Error deleting transcript: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/vectorstore/add_documents', methods=['POST'])
def add_documents_to_vectorstore():
    """Add documents to the unified vectorstore"""
    try:
        if not unified_vectorstore:
            return jsonify({"error": "Unified vectorstore not available"}), 500
        
        data = request.get_json()
        if not data or 'documents' not in data:
            return jsonify({"error": "Documents are required"}), 400
        
        # Convert document data to Document objects
        documents = []
        for doc_data in data['documents']:
            doc = Document(
                page_content=doc_data.get('content', ''),
                metadata=doc_data.get('metadata', {})
            )
            documents.append(doc)
        
        unified_vectorstore.add_documents(documents)
        
        return jsonify({
            "message": f"Added {len(documents)} documents to unified vectorstore",
            "count": len(documents),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error adding documents to vectorstore: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/vectorstore/retriever', methods=['GET'])
def get_vectorstore_retriever():
    """Get retriever information"""
    try:
        if not unified_vectorstore:
            return jsonify({"error": "Unified vectorstore not available"}), 500
        
        retriever = unified_vectorstore.get_retriever()
        
        if retriever:
            return jsonify({
                "retriever_available": True,
                "vectorstore_path": unified_vectorstore.vectorstore_path,
                "transcript_count": len(unified_vectorstore.transcript_metadata),
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "retriever_available": False,
                "message": "No vectorstore available",
                "timestamp": datetime.now().isoformat()
            })
        
    except Exception as e:
        logger.error(f"Error getting retriever info: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Enhanced customer support analysis endpoints
@app.route('/api/analysis/comprehensive', methods=['POST'])
def comprehensive_analysis():
    """
    Perform comprehensive customer support call analysis including:
    - Speech-to-text transcription
    - Speaker diarization
    - Sentiment, tonality, and intent analysis
    - Call summarization
    - Quality evaluation
    - Agent recommendations
    """
    try:
        if not video_transcriber:
            return jsonify({"error": "Video transcriber not initialized"}), 500
        
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400
        
        # Get analysis options from request
        enable_diarization = request.form.get('enable_diarization', 'true').lower() == 'true'
        enable_insights = request.form.get('enable_insights', 'true').lower() == 'true'
        enable_summary = request.form.get('enable_summary', 'true').lower() == 'true'
        enable_evaluation = request.form.get('enable_evaluation', 'true').lower() == 'true'
        enable_agent_analysis = request.form.get('enable_agent_analysis', 'true').lower() == 'true'
        language = request.form.get('language', None)
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        logger.info(f"Starting comprehensive analysis for file: {filename}")
        
        # Perform comprehensive analysis
        analysis_results = video_transcriber.perform_comprehensive_analysis(
            video_path=file_path,
            language=language,
            enable_diarization=enable_diarization,
            enable_insights=enable_insights,
            enable_summary=enable_summary,
            enable_evaluation=enable_evaluation,
            enable_agent_analysis=enable_agent_analysis
        )
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass
        
        return jsonify({
            "success": True,
            "message": "Comprehensive analysis completed successfully",
            "results": analysis_results
        })
        
    except MemoryError as e:
        logger.error(f"Memory error during analysis: {str(e)}")
        return jsonify({
            "error": "Insufficient memory for analysis. Please try with a smaller file or fewer analysis options.",
            "details": str(e)
        }), 500
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {str(e)}")
        return jsonify({
            "error": f"Analysis failed: {str(e)}"
        }), 500

@app.route('/api/analysis/insights', methods=['POST'])
def extract_insights():
    """Extract insights from existing transcript"""
    try:
        if not video_transcriber or not video_transcriber.insights_extractor:
            return jsonify({"error": "Insights extractor not available"}), 500
        
        data = request.get_json()
        if not data or 'transcript' not in data:
            return jsonify({"error": "Transcript text required"}), 400
        
        transcript = data['transcript']
        
        # Extract insights
        insights = video_transcriber.insights_extractor.extract_all_insights(transcript)
        
        return jsonify({
            "success": True,
            "insights": insights
        })
        
    except Exception as e:
        logger.error(f"Error extracting insights: {str(e)}")
        return jsonify({
            "error": f"Insights extraction failed: {str(e)}"
        }), 500

@app.route('/api/analysis/summary', methods=['POST'])
def generate_summary():
    """Generate summary from existing transcript"""
    try:
        if not video_transcriber or not video_transcriber.conversation_summarizer:
            return jsonify({"error": "Conversation summarizer not available"}), 500
        
        data = request.get_json()
        if not data or 'transcript' not in data:
            return jsonify({"error": "Transcript text required"}), 400
        
        transcript = data['transcript']
        insights = data.get('insights', None)
        
        # Generate summary
        summary = video_transcriber.conversation_summarizer.generate_summary(transcript, insights)
        brief_summary = video_transcriber.conversation_summarizer.generate_brief_summary(transcript)
        key_phrases = video_transcriber.conversation_summarizer.extract_key_phrases(transcript)
        
        return jsonify({
            "success": True,
            "summary": summary,
            "brief_summary": brief_summary,
            "key_phrases": key_phrases
        })
        
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return jsonify({
            "error": f"Summary generation failed: {str(e)}"
        }), 500

@app.route('/api/analysis/evaluation', methods=['POST'])
def evaluate_call():
    """Evaluate call quality from existing transcript"""
    try:
        if not video_transcriber or not video_transcriber.call_evaluator:
            return jsonify({"error": "Call evaluator not available"}), 500
        
        data = request.get_json()
        if not data or 'transcript' not in data:
            return jsonify({"error": "Transcript text required"}), 400
        
        transcript = data['transcript']
        insights = data.get('insights', None)
        summary = data.get('summary', None)
        
        # Evaluate call quality
        evaluation = video_transcriber.call_evaluator.evaluate_call_quality(transcript, insights, summary)
        performance_metrics = video_transcriber.call_evaluator.calculate_performance_metrics(transcript)
        
        return jsonify({
            "success": True,
            "evaluation": evaluation,
            "performance_metrics": performance_metrics
        })
        
    except Exception as e:
        logger.error(f"Error evaluating call: {str(e)}")
        return jsonify({
            "error": f"Call evaluation failed: {str(e)}"
        }), 500

@app.route('/api/analysis/agent', methods=['POST'])
def agent_analysis():
    """Perform agent analysis and get recommendations"""
    try:
        if not video_transcriber or not video_transcriber.support_agent:
            return jsonify({"error": "Support agent not available"}), 500
        
        data = request.get_json()
        if not data or 'transcript' not in data:
            return jsonify({"error": "Transcript text required"}), 400
        
        transcript = data['transcript']
        insights = data.get('insights', None)
        summary = data.get('summary', None)
        
        # Perform agent analysis
        agent_analysis = video_transcriber.support_agent.analyze_call_for_actions(transcript, insights, summary)
        similar_cases = video_transcriber.support_agent.suggest_similar_cases(transcript)
        
        return jsonify({
            "success": True,
            "agent_analysis": agent_analysis,
            "similar_cases": similar_cases
        })
        
    except Exception as e:
        logger.error(f"Error in agent analysis: {str(e)}")
        return jsonify({
            "error": f"Agent analysis failed: {str(e)}"
        }), 500

@app.route('/api/analysis/chat', methods=['POST'])
def enhanced_chat():
    """Enhanced chat with agentic capabilities"""
    try:
        if not video_transcriber or not video_transcriber.support_agent:
            return jsonify({"error": "Support agent not available"}), 500
        
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Query required"}), 400
        
        query = data['query']
        transcript = data.get('transcript', None)
        
        # Create enhanced RAG response
        enhanced_response = video_transcriber.support_agent.create_enhanced_rag_response(query, transcript)
        
        return jsonify({
            "success": True,
            "response": enhanced_response
        })
        
    except Exception as e:
        logger.error(f"Error in enhanced chat: {str(e)}")
        return jsonify({
            "error": f"Enhanced chat failed: {str(e)}"
        }), 500

@app.route('/api/analysis/diarization', methods=['POST'])
def perform_diarization():
    """Perform speaker diarization on audio file"""
    try:
        if not video_transcriber or not video_transcriber.speaker_diarizer:
            return jsonify({"error": "Speaker diarization not available"}), 500
        
        if 'file' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Perform diarization
        diarization_result = video_transcriber.speaker_diarizer.perform_diarization(file_path)
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass
        
        return jsonify({
            "success": True,
            "diarization": diarization_result
        })
        
    except Exception as e:
        logger.error(f"Error in speaker diarization: {str(e)}")
        return jsonify({
            "error": f"Speaker diarization failed: {str(e)}"
        }), 500

@app.route('/api/debug', methods=['GET'])
def debug_info():
    try:
        debug_data = {
            "services": {
                "transcriber": transcriber is not None,
                "unified_vectorstore": unified_vectorstore is not None,
                "meeting_baas_client": meeting_baas_client is not None,
                "chat_bot": transcriber.chat_bot is not None if transcriber else False
            },
            "dependencies": {
                "nvidia_embeddings_available": NVIDIA_EMBEDDINGS_AVAILABLE,
                "chroma_available": CHROMA_AVAILABLE,
                "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
                "google_drive_available": GOOGLE_DRIVE_AVAILABLE,
                "moviepy_available": USE_MOVIEPY,
                "gemini_available": GEMINI_AVAILABLE
            },
            "api_keys": {
                "openai_api_key": "Available" if os.getenv("OPENAI_API_KEY") else "Missing",
                "gemini_api_key": "Available" if os.getenv("GEMINI_API_KEY") else "Missing",
                "meetingbaas_api_key": "Available" if os.getenv("MEETINGBAAS_API_KEY") else "Missing"
            },
            "vectorstore_info": {},
            "meeting_info": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Get vectorstore info if available
        if unified_vectorstore:
            try:
                transcripts = unified_vectorstore.get_all_transcripts()
                debug_data["vectorstore_info"] = {
                    "transcript_count": len(transcripts),
                    "vectorstore_exists": unified_vectorstore.vectorstore is not None,
                    "vectorstore_path": unified_vectorstore.vectorstore_path,
                    "embedding_type": unified_vectorstore.get_embedding_type(),
                    "embeddings_available": unified_vectorstore.embeddings is not None
                }
            except Exception as vs_error:
                debug_data["vectorstore_info"] = {
                    "error": str(vs_error),
                    "vectorstore_exists": False,
                    "embedding_type": None
                }
        
        # Get meeting info if available
        if meeting_baas_client:
            try:
                active_bots = meeting_baas_client.list_active_bots()
                debug_data["meeting_info"] = {
                    "active_meetings": len(active_bots),
                    "meeting_baas_available": True
                }
            except Exception as meeting_error:
                debug_data["meeting_info"] = {
                    "error": str(meeting_error),
                    "meeting_baas_available": False
                }
        
        return jsonify(debug_data)
    
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    try:
        status_data = {
            "unified_vectorstore_available": unified_vectorstore is not None,
            "transcriber_available": transcriber is not None,
            "meeting_baas_available": meeting_baas_client is not None,
            "chat_bot_available": transcriber.chat_bot is not None if transcriber else False,
            "dependencies": {
                "nvidia_embeddings_available": NVIDIA_EMBEDDINGS_AVAILABLE,
                "chroma_available": CHROMA_AVAILABLE,
                "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
                "google_drive_available": GOOGLE_DRIVE_AVAILABLE,
                "moviepy_available": USE_MOVIEPY,
                "gemini_available": GEMINI_AVAILABLE
            },
            "api_keys": {
                "openai_api_key": "Available" if os.getenv("OPENAI_API_KEY") else "Missing",
                "gemini_api_key": "Available" if os.getenv("GEMINI_API_KEY") else "Missing",
                "meetingbaas_api_key": "Available" if os.getenv("MEETINGBAAS_API_KEY") else "Missing",
                "nvidia_api_key": "Available" if os.getenv("NVIDIA_API_KEY") else "Missing"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Add vectorstore info if available
        if unified_vectorstore:
            status_data["vectorstore_info"] = {
                "embedding_type": unified_vectorstore.get_embedding_type(),
                "embeddings_available": unified_vectorstore.embeddings is not None,
                "transcript_count": len(unified_vectorstore.get_all_transcripts())
            }
        
        return jsonify(status_data)
    
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/routes', methods=['GET'])
def list_routes():
    """List all available routes for debugging"""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'rule': str(rule)
        })
    return jsonify(routes)

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        "message": "Enhanced RAG Backend API with Unified Embedding",
        "status": "running",
        "available_endpoints": [
            "/api/health",
            "/api/status", 
            "/api/debug",
            "/api/upload",
            "/api/meeting/start",
            "/api/meeting/status/<bot_id>",
            "/api/meeting/stop/<bot_id>",
            "/api/meeting/active",
            "/api/meeting/recording/<bot_id>",
            "/api/meeting/transcript/<bot_id>",
            "/api/transcripts",
            "/api/chat",
            "/api/search",
            "/api/transcript/<doc_id>",
            "/api/vectorstore/add_documents",
            "/api/vectorstore/retriever",
            "/routes"
        ]
    })

# Insights extraction class for customer support call analysis
class InsightsExtractor:
    """Extract structured insights from customer support call transcripts"""
    
    def __init__(self, gemini_client=None):
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI not available. Install with: pip install google-generativeai")
        
        self.gemini_client = gemini_client
        self.model = genai.GenerativeModel('gemini-pro')
        logger.info("Insights extractor initialized for customer support analysis")
    
    def extract_sentiment(self, transcript: str) -> Dict[str, Any]:
        """
        Extract sentiment analysis from customer support call transcript
        
        Args:
            transcript: Call transcript text
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        try:
            prompt = f"""
            Analyze the sentiment of this customer support conversation.
            
            Transcript: {transcript}
            
            Please provide:
            1. Overall sentiment (Positive/Negative/Neutral)
            2. Confidence score (0-100)
            3. Key emotional indicators
            4. Sentiment changes throughout the call
            
            Format your response as JSON with these fields:
            - overall_sentiment
            - confidence_score
            - emotional_indicators
            - sentiment_timeline
            """
            
            response = self.model.generate_content(prompt)
            
            # Try to parse JSON response
            try:
                import json
                result = json.loads(response.text)
            except:
                # Fallback if JSON parsing fails
                result = {
                    "overall_sentiment": "Neutral",
                    "confidence_score": 70,
                    "emotional_indicators": ["neutral"],
                    "sentiment_timeline": "stable",
                    "raw_response": response.text
                }
            
            logger.info(f"Sentiment analysis completed: {result.get('overall_sentiment', 'Unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting sentiment: {str(e)}")
            return {"error": str(e), "overall_sentiment": "Unknown"}
    
    def extract_tonality(self, transcript: str) -> Dict[str, Any]:
        """
        Extract tonality analysis from customer support call transcript
        
        Args:
            transcript: Call transcript text
            
        Returns:
            Dictionary containing tonality analysis results
        """
        try:
            prompt = f"""
            Analyze the tonality in this customer support conversation.
            
            Transcript: {transcript}
            
            Please identify:
            1. Primary tonality (Calm/Angry/Polite/Frustrated/Neutral)
            2. Tonality changes throughout the call
            3. Customer tonality vs Agent tonality
            4. Emotional intensity level (1-10)
            
            Format your response as JSON with these fields:
            - primary_tonality
            - customer_tonality
            - agent_tonality
            - emotional_intensity
            - tonality_changes
            """
            
            response = self.model.generate_content(prompt)
            
            try:
                import json
                result = json.loads(response.text)
            except:
                result = {
                    "primary_tonality": "Neutral",
                    "customer_tonality": "Neutral",
                    "agent_tonality": "Polite",
                    "emotional_intensity": 5,
                    "tonality_changes": "stable",
                    "raw_response": response.text
                }
            
            logger.info(f"Tonality analysis completed: {result.get('primary_tonality', 'Unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting tonality: {str(e)}")
            return {"error": str(e), "primary_tonality": "Unknown"}
    
    def extract_intent(self, transcript: str) -> Dict[str, Any]:
        """
        Extract intent classification from customer support call transcript
        
        Args:
            transcript: Call transcript text
            
        Returns:
            Dictionary containing intent analysis results
        """
        try:
            prompt = f"""
            Classify the intent of this customer support conversation.
            
            Transcript: {transcript}
            
            Please identify:
            1. Primary intent (Complaint/Query/Feedback/Technical Support/Billing/General)
            2. Secondary intents if multiple
            3. Intent confidence score (0-100)
            4. Key intent indicators
            5. Resolution status (Resolved/Partially Resolved/Unresolved/Escalated)
            
            Format your response as JSON with these fields:
            - primary_intent
            - secondary_intents
            - confidence_score
            - intent_indicators
            - resolution_status
            """
            
            response = self.model.generate_content(prompt)
            
            try:
                import json
                result = json.loads(response.text)
            except:
                result = {
                    "primary_intent": "Query",
                    "secondary_intents": [],
                    "confidence_score": 75,
                    "intent_indicators": ["question"],
                    "resolution_status": "Unknown",
                    "raw_response": response.text
                }
            
            logger.info(f"Intent analysis completed: {result.get('primary_intent', 'Unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting intent: {str(e)}")
            return {"error": str(e), "primary_intent": "Unknown"}
    
    def extract_all_insights(self, transcript: str) -> Dict[str, Any]:
        """
        Extract all insights (sentiment, tonality, intent) from customer support call
        
        Args:
            transcript: Call transcript text
            
        Returns:
            Dictionary containing all insights analysis
        """
        try:
            if not memory_manager.check_memory_usage():
                raise MemoryError("Insufficient memory for insights extraction")
            
            logger.info("Starting comprehensive insights extraction")
            
            # Extract all insights
            sentiment = self.extract_sentiment(transcript)
            tonality = self.extract_tonality(transcript)
            intent = self.extract_intent(transcript)
            
            # Combine results
            insights = {
                "sentiment_analysis": sentiment,
                "tonality_analysis": tonality,
                "intent_analysis": intent,
                "extraction_timestamp": datetime.now().isoformat(),
                "transcript_length": len(transcript)
            }
            
            # Calculate overall call quality score
            insights["call_quality_score"] = self._calculate_call_quality_score(insights)
            
            logger.info("Comprehensive insights extraction completed")
            
            # Cleanup memory
            memory_manager.force_cleanup()
            
            return insights
            
        except Exception as e:
            logger.error(f"Error extracting all insights: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_call_quality_score(self, insights: Dict) -> float:
        """
        Calculate overall call quality score based on insights
        
        Args:
            insights: Dictionary containing all insights
            
        Returns:
            Call quality score (0-100)
        """
        try:
            score = 50  # Base score
            
            # Sentiment scoring
            sentiment = insights.get("sentiment_analysis", {})
            if sentiment.get("overall_sentiment") == "Positive":
                score += 20
            elif sentiment.get("overall_sentiment") == "Neutral":
                score += 10
            
            # Tonality scoring
            tonality = insights.get("tonality_analysis", {})
            if tonality.get("primary_tonality") in ["Calm", "Polite"]:
                score += 15
            elif tonality.get("primary_tonality") == "Neutral":
                score += 10
            
            # Intent scoring
            intent = insights.get("intent_analysis", {})
            if intent.get("resolution_status") == "Resolved":
                score += 15
            elif intent.get("resolution_status") == "Partially Resolved":
                score += 10
            
            return min(score, 100)  # Cap at 100
            
        except Exception as e:
            logger.error(f"Error calculating call quality score: {str(e)}")
            return 50  # Default score

# Conversation summarizer class for customer support call analysis
class ConversationSummarizer:
    """Generate comprehensive summaries of customer support calls"""
    
    def __init__(self, gemini_client=None):
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI not available. Install with: pip install google-generativeai")
        
        self.gemini_client = gemini_client
        self.model = genai.GenerativeModel('gemini-pro')
        logger.info("Conversation summarizer initialized for customer support analysis")
    
    def generate_summary(self, transcript: str, insights: Dict = None) -> Dict[str, Any]:
        """
        Generate comprehensive summary of customer support call
        
        Args:
            transcript: Call transcript text
            insights: Optional insights from sentiment/tonality/intent analysis
            
        Returns:
            Dictionary containing call summary and key points
        """
        try:
            if not memory_manager.check_memory_usage():
                raise MemoryError("Insufficient memory for summary generation")
            
            # Create context from insights if available
            context = ""
            if insights:
                sentiment = insights.get("sentiment_analysis", {})
                tonality = insights.get("tonality_analysis", {})
                intent = insights.get("intent_analysis", {})
                
                context = f"""
                Call Analysis Context:
                - Sentiment: {sentiment.get('overall_sentiment', 'Unknown')}
                - Tonality: {tonality.get('primary_tonality', 'Unknown')}
                - Intent: {intent.get('primary_intent', 'Unknown')}
                - Resolution: {intent.get('resolution_status', 'Unknown')}
                """
            
            prompt = f"""
            Generate a comprehensive summary of this customer support call.
            
            Transcript: {transcript}
            
            {context}
            
            Please provide a structured summary including:
            
            1. Executive Summary (2-3 sentences)
            2. Main Issue/Request
            3. Key Points Discussed
            4. Resolution Status
            5. Customer Sentiment
            6. Agent Performance Highlights
            7. Follow-up Actions Required
            8. Call Quality Assessment
            
            Format your response as JSON with these fields:
            - executive_summary
            - main_issue
            - key_points
            - resolution_status
            - customer_sentiment
            - agent_performance
            - follow_up_actions
            - call_quality_assessment
            - summary_timestamp
            """
            
            response = self.model.generate_content(prompt)
            
            try:
                import json
                result = json.loads(response.text)
            except:
                # Fallback summary if JSON parsing fails
                result = {
                    "executive_summary": "Customer support call processed successfully",
                    "main_issue": "General inquiry",
                    "key_points": ["Call processed", "Transcript available"],
                    "resolution_status": "Processed",
                    "customer_sentiment": "Neutral",
                    "agent_performance": "Standard",
                    "follow_up_actions": ["Review transcript", "Analyze insights"],
                    "call_quality_assessment": "Standard",
                    "summary_timestamp": datetime.now().isoformat(),
                    "raw_response": response.text
                }
            
            logger.info("Call summary generated successfully")
            
            # Cleanup memory
            memory_manager.force_cleanup()
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {"error": str(e)}
    
    def generate_brief_summary(self, transcript: str) -> str:
        """
        Generate a brief summary for quick reference
        
        Args:
            transcript: Call transcript text
            
        Returns:
            Brief summary string
        """
        try:
            prompt = f"""
            Generate a brief 2-3 sentence summary of this customer support call:
            
            {transcript}
            
            Focus on the main issue and outcome.
            """
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error generating brief summary: {str(e)}")
            return "Call summary unavailable"
    
    def extract_key_phrases(self, transcript: str) -> List[str]:
        """
        Extract key phrases from the call transcript
        
        Args:
            transcript: Call transcript text
            
        Returns:
            List of key phrases
        """
        try:
            prompt = f"""
            Extract 5-10 key phrases from this customer support call transcript:
            
            {transcript}
            
            Focus on:
            - Main issues mentioned
            - Technical terms
            - Customer concerns
            - Resolution steps
            
            Return as a simple list, one phrase per line.
            """
            
            response = self.model.generate_content(prompt)
            phrases = [phrase.strip() for phrase in response.text.split('\n') if phrase.strip()]
            return phrases[:10]  # Limit to 10 phrases
            
        except Exception as e:
            logger.error(f"Error extracting key phrases: {str(e)}")
            return []

# Agentic framework class for intelligent customer support actions
class SupportAgent:
    """Intelligent agent for customer support call analysis and follow-up actions"""
    
    def __init__(self, gemini_client=None, vectorstore=None):
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI not available. Install with: pip install google-generativeai")
        
        self.gemini_client = gemini_client
        self.vectorstore = vectorstore
        self.model = genai.GenerativeModel('gemini-pro')
        logger.info("Support agent initialized for intelligent customer support actions")
    
    def analyze_call_for_actions(self, transcript: str, insights: Dict = None, summary: Dict = None) -> Dict[str, Any]:
        """
        Analyze call to suggest intelligent follow-up actions
        
        Args:
            transcript: Call transcript text
            insights: Optional insights from sentiment/tonality/intent analysis
            summary: Optional call summary
            
        Returns:
            Dictionary containing suggested actions and recommendations
        """
        try:
            if not memory_manager.check_memory_usage():
                raise MemoryError("Insufficient memory for agent analysis")
            
            # Create context from insights and summary
            context = ""
            if insights:
                sentiment = insights.get("sentiment_analysis", {})
                intent = insights.get("intent_analysis", {})
                context += f"Sentiment: {sentiment.get('overall_sentiment', 'Unknown')}, "
                context += f"Intent: {intent.get('primary_intent', 'Unknown')}, "
                context += f"Resolution: {intent.get('resolution_status', 'Unknown')}"
            
            if summary:
                context += f", Summary: {summary.get('executive_summary', '')}"
            
            prompt = f"""
            Analyze this customer support call and suggest intelligent follow-up actions.
            
            Transcript: {transcript}
            Context: {context}
            
            Please provide recommendations for:
            
            1. Immediate Actions Required (within 24 hours)
            2. Follow-up Actions (within 1 week)
            3. Training Opportunities for Agent
            4. Process Improvements
            5. Customer Satisfaction Actions
            6. Escalation Requirements
            7. Documentation Needs
            8. Quality Assurance Review
            
            Format your response as JSON with these fields:
            - immediate_actions
            - follow_up_actions
            - training_opportunities
            - process_improvements
            - customer_satisfaction_actions
            - escalation_requirements
            - documentation_needs
            - quality_assurance_review
            - priority_level
            """
            
            response = self.model.generate_content(prompt)
            
            try:
                import json
                result = json.loads(response.text)
            except:
                result = {
                    "immediate_actions": ["Review call transcript"],
                    "follow_up_actions": ["Monitor customer satisfaction"],
                    "training_opportunities": ["Standard call handling"],
                    "process_improvements": ["Continue current processes"],
                    "customer_satisfaction_actions": ["Standard follow-up"],
                    "escalation_requirements": ["None"],
                    "documentation_needs": ["Standard documentation"],
                    "quality_assurance_review": ["Standard review"],
                    "priority_level": "Medium",
                    "raw_response": response.text
                }
            
            logger.info("Agent analysis completed successfully")
            
            # Cleanup memory
            memory_manager.force_cleanup()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in agent analysis: {str(e)}")
            return {"error": str(e)}
    
    def suggest_similar_cases(self, transcript: str, limit: int = 5) -> List[Dict]:
        """
        Find similar cases from vector store for reference
        
        Args:
            transcript: Current call transcript
            limit: Maximum number of similar cases to return
            
        Returns:
            List of similar cases with relevance scores
        """
        try:
            if not self.vectorstore:
                logger.warning("Vector store not available for similar case search")
                return []
            
            # Search for similar cases
            similar_docs = self.vectorstore.similarity_search(transcript, k=limit)
            
            similar_cases = []
            for i, doc in enumerate(similar_docs):
                case = {
                    "rank": i + 1,
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": 1.0 - (i * 0.1)  # Simple relevance scoring
                }
                similar_cases.append(case)
            
            logger.info(f"Found {len(similar_cases)} similar cases")
            return similar_cases
            
        except Exception as e:
            logger.error(f"Error finding similar cases: {str(e)}")
            return []
    
    def generate_agent_response(self, user_query: str, context: Dict = None) -> str:
        """
        Generate intelligent agent response for customer support queries
        
        Args:
            user_query: User's question or request
            context: Optional context from previous analysis
            
        Returns:
            Intelligent response from the agent
        """
        try:
            context_str = ""
            if context:
                context_str = f"Context: {json.dumps(context, indent=2)}"
            
            prompt = f"""
            You are an intelligent customer support agent. Respond to this query:
            
            Query: {user_query}
            {context_str}
            
            Provide a helpful, professional response that addresses the user's needs.
            If you have access to similar cases or insights, reference them appropriately.
            """
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error generating agent response: {str(e)}")
            return "I apologize, but I'm unable to process your request at the moment. Please try again later."
    
    def create_enhanced_rag_response(self, query: str, transcript: str = None) -> Dict[str, Any]:
        """
        Create enhanced RAG response with agentic capabilities
        
        Args:
            query: User's question
            transcript: Optional transcript for context
            
        Returns:
            Enhanced RAG response with multiple components
        """
        try:
            # Get similar cases
            similar_cases = self.suggest_similar_cases(query if transcript is None else transcript)
            
            # Generate agent response
            agent_response = self.generate_agent_response(query, {
                "similar_cases": similar_cases,
                "transcript_available": transcript is not None
            })
            
            # Create enhanced response
            enhanced_response = {
                "agent_response": agent_response,
                "similar_cases": similar_cases,
                "confidence_score": self._calculate_response_confidence(query, similar_cases),
                "response_timestamp": datetime.now().isoformat(),
                "response_type": "enhanced_rag"
            }
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Error creating enhanced RAG response: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_response_confidence(self, query: str, similar_cases: List[Dict]) -> float:
        """
        Calculate confidence score for agent response
        
        Args:
            query: User's question
            similar_cases: Similar cases found
            
        Returns:
            Confidence score (0-100)
        """
        try:
            base_confidence = 70
            
            # Adjust based on number of similar cases
            if len(similar_cases) > 0:
                base_confidence += min(len(similar_cases) * 5, 20)
            
            # Adjust based on query complexity
            if len(query.split()) > 10:
                base_confidence -= 10
            
            return min(max(base_confidence, 0), 100)
            
        except Exception as e:
            logger.error(f"Error calculating response confidence: {str(e)}")
            return 70  # Default confidence

# Call evaluator class for customer support call quality assessment
class CallEvaluator:
    """Evaluate customer support call quality and performance metrics"""
    
    def __init__(self, gemini_client=None):
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI not available. Install with: pip install google-generativeai")
        
        self.gemini_client = gemini_client
        self.model = genai.GenerativeModel('gemini-pro')
        logger.info("Call evaluator initialized for customer support quality assessment")
    
    def evaluate_call_quality(self, transcript: str, insights: Dict = None, summary: Dict = None) -> Dict[str, Any]:
        """
        Evaluate overall call quality and performance
        
        Args:
            transcript: Call transcript text
            insights: Optional insights from sentiment/tonality/intent analysis
            summary: Optional call summary
            
        Returns:
            Dictionary containing call quality evaluation
        """
        try:
            if not memory_manager.check_memory_usage():
                raise MemoryError("Insufficient memory for call evaluation")
            
            # Create evaluation context
            context = ""
            if insights:
                sentiment = insights.get("sentiment_analysis", {})
                tonality = insights.get("tonality_analysis", {})
                intent = insights.get("intent_analysis", {})
                context = f"Sentiment: {sentiment.get('overall_sentiment', 'Unknown')}, "
                context += f"Tonality: {tonality.get('primary_tonality', 'Unknown')}, "
                context += f"Intent: {intent.get('primary_intent', 'Unknown')}"
            
            prompt = f"""
            Evaluate the quality of this customer support call.
            
            Transcript: {transcript}
            Context: {context}
            
            Please evaluate:
            
            1. Communication Effectiveness (1-10)
            2. Problem Resolution (1-10)
            3. Customer Satisfaction Indicators (1-10)
            4. Agent Professionalism (1-10)
            5. Call Efficiency (1-10)
            6. Overall Quality Score (1-10)
            7. Areas for Improvement
            8. Strengths Highlighted
            9. Compliance Assessment
            10. Training Recommendations
            
            Format your response as JSON with these fields:
            - communication_effectiveness
            - problem_resolution
            - customer_satisfaction
            - agent_professionalism
            - call_efficiency
            - overall_quality_score
            - areas_for_improvement
            - strengths_highlighted
            - compliance_assessment
            - training_recommendations
            """
            
            response = self.model.generate_content(prompt)
            
            try:
                import json
                result = json.loads(response.text)
            except:
                result = {
                    "communication_effectiveness": 7,
                    "problem_resolution": 7,
                    "customer_satisfaction": 7,
                    "agent_professionalism": 7,
                    "call_efficiency": 7,
                    "overall_quality_score": 7,
                    "areas_for_improvement": ["Standard call handling"],
                    "strengths_highlighted": ["Professional approach"],
                    "compliance_assessment": "Compliant",
                    "training_recommendations": ["Standard training"],
                    "raw_response": response.text
                }
            
            logger.info("Call quality evaluation completed")
            
            # Cleanup memory
            memory_manager.force_cleanup()
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating call quality: {str(e)}")
            return {"error": str(e)}
    
    def calculate_performance_metrics(self, transcript: str, call_duration: float = None) -> Dict[str, Any]:
        """
        Calculate performance metrics for the call
        
        Args:
            transcript: Call transcript text
            call_duration: Optional call duration in seconds
            
        Returns:
            Dictionary containing performance metrics
        """
        try:
            # Basic metrics calculation
            word_count = len(transcript.split())
            sentence_count = len(transcript.split('.'))
            
            metrics = {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "avg_words_per_sentence": word_count / max(sentence_count, 1),
                "call_duration_seconds": call_duration,
                "words_per_minute": (word_count * 60) / max(call_duration or 60, 1),
                "transcript_length": len(transcript)
            }
            
            # Add quality indicators
            if call_duration:
                metrics["efficiency_score"] = min(100, (word_count / max(call_duration / 60, 1)) * 10)
            
            logger.info("Performance metrics calculated")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {"error": str(e)}

# Call metadata class for customer support call information management
class CallMetadata:
    """Manage metadata for customer support calls"""
    
    def __init__(self):
        self.metadata_template = {
            "call_id": None,
            "timestamp": None,
            "duration": None,
            "file_size": None,
            "file_type": None,
            "customer_id": None,
            "agent_id": None,
            "call_type": None,
            "priority": None,
            "status": None,
            "tags": [],
            "notes": "",
            "quality_score": None,
            "resolution_status": None,
            "escalation_level": None,
            "follow_up_required": False,
            "training_flagged": False
        }
        logger.info("Call metadata manager initialized")
    
    def create_call_metadata(self, file_info: Dict, analysis_results: Dict = None) -> Dict[str, Any]:
        """
        Create comprehensive call metadata
        
        Args:
            file_info: Information about the uploaded file
            analysis_results: Optional results from call analysis
            
        Returns:
            Dictionary containing call metadata
        """
        try:
            # Generate unique call ID
            call_id = self._generate_call_id(file_info)
            
            # Create base metadata
            metadata = self.metadata_template.copy()
            metadata.update({
                "call_id": call_id,
                "timestamp": datetime.now().isoformat(),
                "file_size": file_info.get("file_size", 0),
                "file_type": file_info.get("file_type", "unknown"),
                "status": "processed"
            })
            
            # Add analysis results if available
            if analysis_results:
                insights = analysis_results.get("insights", {})
                summary = analysis_results.get("summary", {})
                evaluation = analysis_results.get("evaluation", {})
                
                # Extract key information
                intent_analysis = insights.get("intent_analysis", {})
                sentiment_analysis = insights.get("sentiment_analysis", {})
                
                metadata.update({
                    "call_type": intent_analysis.get("primary_intent", "General"),
                    "priority": self._determine_priority(insights),
                    "resolution_status": intent_analysis.get("resolution_status", "Unknown"),
                    "quality_score": evaluation.get("overall_quality_score", 7),
                    "tags": self._generate_tags(insights, summary),
                    "follow_up_required": self._check_follow_up_required(insights),
                    "training_flagged": self._check_training_required(evaluation)
                })
            
            logger.info(f"Call metadata created for call ID: {call_id}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error creating call metadata: {str(e)}")
            return self.metadata_template.copy()
    
    def _generate_call_id(self, file_info: Dict) -> str:
        """Generate unique call ID based on file information"""
        try:
            filename = file_info.get("filename", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
            return f"CALL_{timestamp}_{file_hash}"
        except Exception as e:
            logger.error(f"Error generating call ID: {str(e)}")
            return f"CALL_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _determine_priority(self, insights: Dict) -> str:
        """Determine call priority based on insights"""
        try:
            sentiment = insights.get("sentiment_analysis", {})
            intent = insights.get("intent_analysis", {})
            
            # High priority indicators
            if sentiment.get("overall_sentiment") == "Negative":
                return "High"
            if intent.get("primary_intent") in ["Complaint", "Technical Support"]:
                return "Medium"
            if intent.get("resolution_status") == "Unresolved":
                return "High"
            
            return "Low"
        except Exception as e:
            logger.error(f"Error determining priority: {str(e)}")
            return "Medium"
    
    def _generate_tags(self, insights: Dict, summary: Dict) -> List[str]:
        """Generate tags for the call based on analysis"""
        try:
            tags = []
            
            # Add sentiment tag
            sentiment = insights.get("sentiment_analysis", {})
            if sentiment.get("overall_sentiment"):
                tags.append(f"sentiment_{sentiment['overall_sentiment'].lower()}")
            
            # Add intent tag
            intent = insights.get("intent_analysis", {})
            if intent.get("primary_intent"):
                tags.append(f"intent_{intent['primary_intent'].lower().replace(' ', '_')}")
            
            # Add resolution tag
            if intent.get("resolution_status"):
                tags.append(f"resolution_{intent['resolution_status'].lower().replace(' ', '_')}")
            
            # Add summary tags
            if summary:
                tags.append("has_summary")
            
            return tags
            
        except Exception as e:
            logger.error(f"Error generating tags: {str(e)}")
            return ["general"]
    
    def _check_follow_up_required(self, insights: Dict) -> bool:
        """Check if follow-up is required based on insights"""
        try:
            intent = insights.get("intent_analysis", {})
            sentiment = insights.get("sentiment_analysis", {})
            
            # Follow-up required indicators
            if intent.get("resolution_status") in ["Unresolved", "Partially Resolved"]:
                return True
            if sentiment.get("overall_sentiment") == "Negative":
                return True
            if intent.get("primary_intent") == "Complaint":
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking follow-up requirement: {str(e)}")
            return False
    
    def _check_training_required(self, evaluation: Dict) -> bool:
        """Check if training is required based on evaluation"""
        try:
            quality_score = evaluation.get("overall_quality_score", 7)
            
            # Training required if quality score is low
            if quality_score < 6:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking training requirement: {str(e)}")
            return False
    
    def update_metadata(self, call_id: str, updates: Dict) -> Dict[str, Any]:
        """
        Update existing call metadata
        
        Args:
            call_id: Call ID to update
            updates: Dictionary of updates to apply
            
        Returns:
            Updated metadata
        """
        try:
            # In a real implementation, you would load existing metadata
            # For now, we'll create a new metadata entry
            metadata = self.metadata_template.copy()
            metadata.update(updates)
            metadata["call_id"] = call_id
            metadata["last_updated"] = datetime.now().isoformat()
            
            logger.info(f"Metadata updated for call ID: {call_id}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error updating metadata: {str(e)}")
            return {"error": str(e)}

if __name__ == '__main__':
    print("🎯 Enhanced Customer Support Call Analysis System")
    print("=" * 70)
    
    # Check dependencies
    print("🔧 Checking dependencies...")
    print(f"   MoviePy: {'✓ Available' if USE_MOVIEPY else '✗ Not Available'}")
    print(f"   ChromaDB: {'✓ Available' if CHROMA_AVAILABLE else '✗ Not Available'}")
    print(f"   Google Drive: {'✓ Available' if GOOGLE_DRIVE_AVAILABLE else '✗ Not Available'}")
    print(f"   Gemini API: {'✓ Available' if GEMINI_AVAILABLE else '✗ Not Available'}")
    print(f"   Speaker Diarization: {'✓ Available' if DIARIZATION_AVAILABLE else '✗ Not Available'}")
    print(f"   LangChain: {'✓ Available' if LANGCHAIN_AVAILABLE else '✗ Not Available'}")
    
    # Check environment variables
    print("\n🔑 Checking API keys...")
    print(f"   OpenAI API Key: {'✓ Available' if os.getenv('OPENAI_API_KEY') else '✗ Missing'} (for Whisper transcription)")
    print(f"   Gemini API Key: {'✓ Available' if os.getenv('GEMINI_API_KEY') else '✗ Missing'} (for chat/completion and insights)")
    print(f"   HF Token: {'✓ Available' if os.getenv('HF_TOKEN') else '✗ Missing'} (for speaker diarization)")
    print(f"   MeetingBaas API Key: {'✓ Available' if os.getenv('MEETINGBAAS_API_KEY') else '✗ Missing'}")
    
    # Initialize services
    if not initialize_services():
        print("\n❌ Critical Error: Failed to initialize services")
        exit(1)
    
    # Check if critical services are available
    if not transcriber:
        print("\n❌ Critical Error: Transcriber service failed to initialize")
        print("   This is likely due to:")
        print("   1. Invalid OpenAI API key")
        print("   2. OpenAI client compatibility issues")
        print("   3. Network connectivity issues")
        print("   Please check the logs above for more details")
        print("\n🔧 Troubleshooting steps:")
        print("   1. Verify your OpenAI API key is valid")
        print("   2. Update OpenAI library: pip install --upgrade openai")
        print("   3. Check internet connection")
        exit(1)
    
    # Check enhanced analysis services
    print("\n🧠 Enhanced Analysis Services:")
    if transcriber.insights_extractor:
        print("   ✓ Insights Extraction (Sentiment, Tonality, Intent)")
    else:
        print("   ✗ Insights Extraction (requires GEMINI_API_KEY)")
    
    if transcriber.conversation_summarizer:
        print("   ✓ Conversation Summarizer")
    else:
        print("   ✗ Conversation Summarizer (requires GEMINI_API_KEY)")
    
    if transcriber.call_evaluator:
        print("   ✓ Call Quality Evaluator")
    else:
        print("   ✗ Call Quality Evaluator (requires GEMINI_API_KEY)")
    
    if transcriber.support_agent:
        print("   ✓ Support Agent (Intelligent Recommendations)")
    else:
        print("   ✗ Support Agent (requires GEMINI_API_KEY)")
    
    if transcriber.speaker_diarizer:
        print("   ✓ Speaker Diarization")
    else:
        print("   ✗ Speaker Diarization (requires HF_TOKEN)")
    
    # Check memory management
    print(f"\n💾 Memory Management: ✓ Active (Threshold: {memory_manager.threshold}%)")
    
    # Check other services
    if not meeting_baas_client:
        print("\n⚠️  Warning: MeetingBaas client not available")
        print("   Reason: Missing or invalid MEETINGBAAS_API_KEY")
        print("   Meeting recording functionality will be disabled")
    
    if not unified_vectorstore:
        print("\n⚠️  Warning: Unified vectorstore not available")
        print("   Reason: Missing Gemini API key or dependencies")
        print("   Install with: pip install google-generativeai")
        print("   Add GEMINI_API_KEY to your .env file")
        print("   Chat and search functionality will be limited")
    else:
        print("\n✓ Unified vectorstore available with Gemini embeddings")
    
    if not transcriber.chat_bot or not transcriber.gemini_client:
        print("\n⚠️  Warning: Gemini chat service not available")
        print("   Reason: Missing or invalid GEMINI_API_KEY")
        print("   Install with: pip install google-generativeai")
        print("   Chat functionality will be limited")
    
    print("\n✅ Backend API started successfully")
    print("📊 Core Services:")
    print("   ✓ Basic transcription service (using OpenAI Whisper)")
    if meeting_baas_client:
        print("   ✓ Meeting recording service")
    if unified_vectorstore:
        print("   ✓ Unified vector store (using Gemini embeddings)")
    if transcriber.chat_bot and transcriber.gemini_client:
        print("   ✓ Gemini chat service")
    
    print("\n🎯 Enhanced Analysis Services:")
    if transcriber.insights_extractor:
        print("   ✓ Sentiment, tonality, and intent analysis")
    if transcriber.conversation_summarizer:
        print("   ✓ Call summarization and key phrase extraction")
    if transcriber.call_evaluator:
        print("   ✓ Call quality evaluation and performance metrics")
    if transcriber.support_agent:
        print("   ✓ Intelligent agent recommendations")
    if transcriber.speaker_diarizer:
        print("   ✓ Speaker diarization and separation")
    
    print("\n🌐 API will be available at: http://localhost:5000")
    print("📡 Available endpoints:")
    print("   Basic Services:")
    print("   - GET  /api/health")
    print("   - GET  /api/status") 
    print("   - GET  /api/debug")
    print("   - POST /api/upload")
    
    if meeting_baas_client:
        print("   Meeting Recording:")
        print("   - POST /api/meeting/start")
        print("   - GET  /api/meeting/status/{bot_id}")
        print("   - POST /api/meeting/stop/{bot_id}")
        print("   - GET  /api/meeting/active")
        print("   - GET  /api/meeting/recording/{bot_id}")
        print("   - GET  /api/meeting/transcript/{bot_id}")
    
    if unified_vectorstore:
        print("   RAG & Chat:")
        print("   - GET  /api/transcripts")
        print("   - POST /api/chat")
        print("   - POST /api/search")
        print("   - DELETE /api/transcript/{doc_id}")
        print("   - POST /api/vectorstore/add_documents")
        print("   - GET  /api/vectorstore/retriever")
    
    print("   Enhanced Analysis:")
    print("   - POST /api/analysis/comprehensive")
    print("   - POST /api/analysis/insights")
    print("   - POST /api/analysis/summary")
    print("   - POST /api/analysis/evaluation")
    print("   - POST /api/analysis/agent")
    print("   - POST /api/analysis/chat")
    print("   - POST /api/analysis/diarization")
    
    print("   - GET  /routes (for debugging)")
    print("\n🎯 Optimized for 4GB RAM Systems")
    print("🚀 Starting Flask server...")
    print("=" * 70)
    
    app.run(debug=False, host='0.0.0.0', port=5000)
