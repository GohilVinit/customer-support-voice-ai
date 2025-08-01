# Customer Support Voice Analytics System

## 🎯 Problem Statement

Customer support centers generate vast amounts of voice data every day. Extracting meaningful insights from these conversations can improve customer experience, operational efficiency, and agent performance.

This project implements an end-to-end pipeline that converts customer-agent audio calls into structured insights using Speech-to-Text (STT) tools, LLM APIs, and agentic/retrieval-augmented generation (RAG) frameworks.

## 🚀 Features

### 1. **Multilingual Audio Transcription**
- **Speech-to-Text**: OpenAI Whisper for high-quality transcription
- **Multilingual Support**: English, Hindi, and Hinglish (Hindi-English mixed)
- **Speaker Diarization**: Automatic speaker separation and identification
- **Audio/Video Processing**: Support for various audio and video formats

### 2. **Structured Insights Extraction**
- **Sentiment Analysis**: Positive/Negative/Neutral classification
- **Tonality Detection**: Calm, Angry, Polite, Frustrated, etc.
- **Intent Classification**: Complaint, Query, Feedback, General Inquiry
- **Call Quality Metrics**: Response time, resolution efficiency, customer satisfaction

### 3. **Conversation Analysis**
- **Smart Summarization**: AI-powered conversation summaries
- **Key Phrase Extraction**: Important topics and action items
- **Call Evaluation**: Performance metrics and quality scoring
- **Metadata Generation**: Call ID, timestamps, priority levels, tags

### 4. **Agentic Framework & RAG**
- **LangChain Integration**: Intelligent document retrieval and processing
- **Vector Database**: ChromaDB for efficient similarity search
- **Contextual Responses**: AI-powered follow-up suggestions
- **Knowledge Base**: FAQ and policy document integration

### 5. **Real-time Processing**
- **Live Meeting Recording**: Integration with MeetingBaas API
- **Streamlit Frontend**: User-friendly web interface
- **RESTful API**: Backend services for scalable deployment
- **File Upload Support**: Drag-and-drop audio/video processing

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.12**: Main programming language
- **Flask**: Backend web framework
- **Streamlit**: Frontend web interface
- **OpenAI Whisper**: Speech-to-text transcription
- **Google Gemini**: LLM for insights and analysis
- **LangChain**: Agentic framework and RAG implementation

### Audio/Video Processing
- **MoviePy**: Video processing and audio extraction
- **PyDub**: Audio manipulation and format conversion
- **Librosa**: Audio analysis and feature extraction

### Vector Database & Embeddings
- **ChromaDB**: Vector store for similarity search
- **Sentence Transformers**: Text embeddings
- **Google Generative AI**: Alternative embedding solution

### Additional Libraries
- **Pandas/NumPy**: Data processing
- **Scikit-learn**: Machine learning utilities
- **FPDF**: PDF report generation
- **Google Drive API**: Cloud storage integration

## 📋 Prerequisites

### System Requirements
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: 2GB free space
- **Python**: 3.11 or 3.12
- **OS**: Windows 10/11, macOS, or Linux

### API Keys Required
1. **OpenAI API Key**: For Whisper transcription
2. **Google Gemini API Key**: For LLM analysis
3. **MeetingBaas API Key** (Optional): For live meeting recording

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd csinfocomm
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the project root:
```env
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_GENERATIVEAI_API_KEY=your_gemini_api_key_here

# Optional: MeetingBaas API
MEETINGBAAS_API_KEY=your_meetingbaas_api_key_here

# Optional: Google Drive Integration
GOOGLE_DRIVE_CREDENTIALS_FILE=credentials.json
GOOGLE_DRIVE_TOKEN_FILE=token.pickle

# Optional: HuggingFace Token (for speaker diarization)
HUGGINGFACE_TOKEN=your_hf_token_here
```

### 5. Download Sample Data
```bash
# Create directories for sample data
mkdir -p sample_audio
mkdir -p sample_documents
```

## 🎯 Usage

### Method 1: Streamlit Web Interface (Recommended)

1. **Start the Backend Server**:
```bash
python backend.py
```

2. **Launch the Frontend**:
```bash
streamlit run enhanced_finalapp.py
```

3. **Access the Application**:
   - Open your browser to `http://localhost:8501`
   - Upload audio/video files or start live meeting recording
   - View comprehensive analysis results

### Method 2: Direct API Usage

1. **Start Backend Only**:
```bash
python backend.py
```

2. **API Endpoints Available**:
   - `POST /api/upload`: Upload and transcribe audio/video
   - `POST /api/analysis/comprehensive`: Full analysis pipeline
   - `POST /api/chat`: Chat with processed data
   - `GET /api/transcripts`: Retrieve all transcripts

### Method 3: Jupyter Notebook

Create a new notebook and import the modules:
```python
from backend import VideoTranscriber, InsightsExtractor
from enhanced_finalapp import comprehensive_analysis_backend

# Process audio file
results = comprehensive_analysis_backend("path/to/audio.mp3")
```

## 📊 Sample Data

### Audio Files
The system includes sample audio files for testing:
- `temp_first_2_min_trimmed.mp3`: Sample customer support call

### Sample Documents for RAG
Create sample FAQ and policy documents in the `sample_documents/` folder:
- Customer service policies
- Product information
- Common troubleshooting guides
- Escalation procedures

## 🔍 Analysis Pipeline

### 1. Audio Processing
```
Audio/Video Input → Audio Extraction → Whisper Transcription → Speaker Diarization
```

### 2. Insights Extraction
```
Transcript → Sentiment Analysis → Tonality Detection → Intent Classification
```

### 3. Summary Generation
```
Transcript + Insights → AI Summary → Key Phrases → Action Items
```

### 4. RAG Integration
```
Query → Vector Search → Context Retrieval → AI Response Generation
```

## 📈 Evaluation Metrics

### Transcription Quality
- **Word Error Rate (WER)**: Accuracy of transcription
- **Speaker Identification**: Diarization accuracy
- **Language Detection**: Multilingual support accuracy

### Analysis Quality
- **Sentiment Accuracy**: Comparison with human annotations
- **Intent Classification**: Precision, Recall, F1-Score
- **Summary Relevance**: ROUGE scores for summarization

### System Performance
- **Processing Time**: End-to-end pipeline speed
- **Memory Usage**: Resource efficiency
- **API Response Time**: Real-time performance

## 🎯 Deliverables

### ✅ Completed Features

1. **✅ Multilingual Transcription**
   - OpenAI Whisper integration
   - Support for English, Hindi, Hinglish
   - Speaker diarization (optional)

2. **✅ Structured Insights**
   - Sentiment analysis (Positive/Negative/Neutral)
   - Tonality detection (Calm, Angry, Polite, Frustrated)
   - Intent classification (Complaint, Query, Feedback)

3. **✅ Conversation Summary**
   - AI-powered summarization
   - Key phrase extraction
   - Action item identification

4. **✅ Evaluation Strategy**
   - Quality metrics calculation
   - Performance benchmarking
   - Comparative analysis

5. **✅ Agentic Framework**
   - LangChain integration
   - RAG implementation
   - Document retrieval system

6. **✅ Code Delivery**
   - Python scripts and modules
   - Streamlit web interface
   - RESTful API endpoints

## 🔧 Configuration Options

### Memory Management
The system is optimized for 4GB RAM systems:
- Automatic memory cleanup
- Efficient resource usage
- Fallback mechanisms for heavy operations

### Language Support
- **Primary**: English
- **Secondary**: Hindi, Hinglish
- **Extensible**: Additional languages via Whisper

### Processing Options
- **Real-time**: Live meeting recording
- **Batch**: File upload processing
- **Hybrid**: Combined approach

## 🚨 Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce batch size
   - Enable memory cleanup
   - Use smaller audio files

2. **API Key Issues**
   - Verify API keys in `.env` file
   - Check API quota limits
   - Ensure proper permissions

3. **Audio Processing Errors**
   - Check file format compatibility
   - Verify audio file integrity
   - Ensure sufficient disk space

4. **Diarization Issues**
   - Install pyannote.audio (Python 3.11 only)
   - Obtain HuggingFace token
   - Check GPU availability

### Performance Optimization

1. **For 4GB RAM Systems**:
   - Use smaller audio chunks
   - Enable memory management
   - Disable heavy features

2. **For Better Performance**:
   - Use GPU acceleration
   - Increase RAM allocation
   - Optimize batch processing

## 📝 API Documentation

### Core Endpoints

#### Upload and Transcribe
```http
POST /api/upload
Content-Type: multipart/form-data

Parameters:
- file: Audio/video file
- language: Language code (en, hi, etc.)
- enable_diarization: Boolean
```

#### Comprehensive Analysis
```http
POST /api/analysis/comprehensive
Content-Type: application/json

{
  "file_path": "path/to/audio.mp3",
  "language": "en",
  "enable_diarization": true,
  "enable_insights": true,
  "enable_summary": true,
  "enable_evaluation": true,
  "enable_agent_analysis": true
}
```

#### Chat with Data
```http
POST /api/chat
Content-Type: application/json

{
  "question": "What was the main issue discussed?",
  "n_context_results": 5
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for Whisper transcription
- Google for Gemini LLM
- LangChain for RAG framework
- Streamlit for web interface
- MeetingBaas for live recording integration

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the API documentation
3. Create an issue in the repository
4. Contact the development team

---

**Note**: This is a prototype system designed for demonstration purposes. For production use, additional security, scalability, and reliability measures should be implemented. 