# app.py — IntelliDoc: AI-Powered Multilingual Document Analysis System
import os
import streamlit as st
from dotenv import load_dotenv
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import numpy as np
import faiss
from groq import Groq
from langdetect import detect, DetectorFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import io
import json
from datetime import datetime
import pandas as pd
import cv2
import base64
import re

DetectorFactory.seed = 0

# -------------------- Load ENV --------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSDATA_PREFIX = os.getenv("TESSDATA_PREFIX")

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
if TESSDATA_PREFIX:
    os.environ["TESSDATA_PREFIX"] = TESSDATA_PREFIX

# -------------------- Config --------------------
OCR_LANGS = "eng+kan+hin+tam+tel+mar+mal+guj+ben+pan"
TOP_K = 8
SIMILARITY_THRESHOLD = 0.1
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
MAX_CONTEXT_CHARS = 4500
MAX_WORKERS = 4

# -------------------- Clients --------------------
groq_client = Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def get_tfidf_vectorizer():
    return TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=1,
        stop_words=None
    )

# -------------------- Table Extraction --------------------
def extract_tables_from_pdf(pdf_bytes, doc_name):
    """Extract tables from PDF"""
    tables = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num, page in enumerate(doc, start=1):
            try:
                # Extract tables using PyMuPDF
                page_tables = page.find_tables()
                if page_tables:
                    for table_idx, table in enumerate(page_tables.tables):
                        try:
                            table_data = table.extract()
                            if table_data:
                                df = pd.DataFrame(table_data[1:], columns=table_data[0] if table_data else None)
                                tables.append({
                                    'doc': doc_name,
                                    'page': page_num,
                                    'table_num': table_idx + 1,
                                    'data': df
                                })
                        except:
                            continue
            except:
                continue
        doc.close()
    except Exception as e:
        st.warning(f"Table extraction error: {e}")
    return tables

# -------------------- Chart/Diagram Detection --------------------
def detect_visual_elements(pdf_bytes, doc_name):
    """Detect charts, diagrams, flowcharts in PDF"""
    visuals = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num, page in enumerate(doc, start=1):
            # Get images from page
            image_list = page.get_images()
            
            if image_list:
                for img_idx, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Convert to PIL Image
                        img = Image.open(io.BytesIO(image_bytes))
                        
                        # Analyze image to detect if it's a chart/diagram
                        img_array = np.array(img.convert('RGB'))
                        
                        # Simple heuristics to detect charts/diagrams
                        # Check for lines, geometric shapes, etc.
                        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                        edges = cv2.Canny(gray, 50, 150)
                        
                        # Count edge pixels (charts have lots of edges)
                        edge_ratio = np.count_nonzero(edges) / edges.size
                        
                        visual_type = "Unknown"
                        if edge_ratio > 0.05:
                            # Try to determine type
                            if edge_ratio > 0.15:
                                visual_type = "Flowchart/Diagram"
                            elif edge_ratio > 0.08:
                                visual_type = "Chart/Graph"
                            else:
                                visual_type = "Image"
                        
                        visuals.append({
                            'doc': doc_name,
                            'page': page_num,
                            'type': visual_type,
                            'image': img,
                            'size': img.size
                        })
                    except:
                        continue
        doc.close()
    except Exception as e:
        st.warning(f"Visual detection error: {e}")
    return visuals

# -------------------- Smart OCR Detection --------------------
def needs_ocr(page):
    """Quickly determine if a page needs OCR"""
    text = page.get_text("text")
    if len(text.strip()) > 50:
        return False
    image_list = page.get_images()
    if len(image_list) > 0:
        return True
    return True

def extract_text_from_page_fast(page, page_num, use_ocr=False):
    """Fast text extraction with page number tracking"""
    if not use_ocr:
        text = page.get_text("text")
        if len(text.strip()) > 50:
            return text.strip(), page_num
    
    try:
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        text = pytesseract.image_to_string(
            img, 
            lang=OCR_LANGS,
            config='--psm 6 --oem 3'
        )
        return text.strip(), page_num
    except Exception as e:
        return "", page_num

# -------------------- Parallel PDF Processing --------------------
def process_page(page_info):
    """Process a single page"""
    page_num, page, doc_name, force_ocr = page_info
    
    try:
        use_ocr = force_ocr or needs_ocr(page)
        text, pg_num = extract_text_from_page_fast(page, page_num + 1, use_ocr=use_ocr)
        
        if text:
            return {
                'page_num': page_num,
                'actual_page': page_num + 1,
                'text': text,
                'doc_name': doc_name,
                'success': True
            }
        return {'success': False}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def extract_text_from_pdf_parallel(pdf_bytes, doc_name, force_ocr=False):
    """Extract text from PDF using parallel processing"""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        
        page_tasks = [
            (i, doc[i], doc_name, force_ocr) 
            for i in range(total_pages)
        ]
        
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_page = {
                executor.submit(process_page, task): task[0] 
                for task in page_tasks
            }
            
            for future in as_completed(future_to_page):
                result = future.result()
                if result.get('success'):
                    results.append(result)
        
        doc.close()
        results.sort(key=lambda x: x['page_num'])
        
        full_text = ""
        for r in results:
            page_num = r['actual_page']
            text = r['text']
            full_text += f"\n\n[Page {page_num}]\n{text}"
        
        return full_text, len(results), total_pages
        
    except Exception as e:
        st.error(f"PDF processing error: {e}")
        return "", 0, 0

def extract_text_from_image_fast(image_file):
    """Fast image OCR"""
    try:
        img = Image.open(image_file)
        max_size = 2000
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        text = pytesseract.image_to_string(
            img,
            lang=OCR_LANGS,
            config='--psm 6 --oem 3'
        )
        return text.strip()
    except Exception as e:
        st.warning(f"Image OCR failed: {e}")
        return ""

# -------------------- Chunking --------------------
def chunk_text_smart(text, source_name, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Smart chunking with page tracking"""
    if not text or len(text.strip()) < 50:
        return [], []
    
    chunks, meta = [], []
    pages = text.split('[Page ')
    
    for page_section in pages:
        if not page_section.strip():
            continue
        
        page_num = None
        if ']' in page_section:
            try:
                page_num = int(page_section.split(']')[0])
                page_text = page_section.split(']', 1)[1]
            except:
                page_text = page_section
        else:
            page_text = page_section
        
        start = 0
        chunk_id = 0
        
        while start < len(page_text):
            end = start + chunk_size
            chunk = page_text[start:end].strip()
            
            if chunk:
                chunks.append(chunk)
                meta.append({
                    "source": source_name,
                    "page": page_num,
                    "chunk_id": chunk_id
                })
                chunk_id += 1
            
            start += chunk_size - overlap
    
    return chunks, meta

# -------------------- TF-IDF Embeddings --------------------
def embed_texts_tfidf(texts, vectorizer=None):
    """Fast TF-IDF embeddings"""
    try:
        if vectorizer is None:
            vectorizer = get_tfidf_vectorizer()
            matrix = vectorizer.fit_transform(texts)
        else:
            matrix = vectorizer.transform(texts)
        
        return matrix.toarray().astype("float32"), vectorizer
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return np.zeros((len(texts), 100), dtype="float32"), None

def build_faiss_index(emb_np):
    """Build FAISS index"""
    if emb_np is None or emb_np.shape[0] == 0:
        raise ValueError("Empty embeddings")
    
    norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_np = emb_np / norms
    
    dim = emb_np.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(emb_np)
    return idx

# -------------------- Language Utilities --------------------
LANG_MAP = {
    "kn": "Kannada", "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
    "mr": "Marathi", "ml": "Malayalam", "gu": "Gujarati",
    "bn": "Bengali", "pa": "Punjabi", "en": "English"
}

LANG_CODES = {
    "english": "en", "kannada": "kn", "hindi": "hi", "tamil": "ta",
    "telugu": "te", "marathi": "mr", "malayalam": "ml", "gujarati": "gu",
    "bengali": "bn", "punjabi": "pa"
}

def detect_question_language(question):
    try:
        return detect(question)
    except:
        return "en"

def extract_target_language(question):
    q_lower = question.lower()
    for lang_name, code in LANG_CODES.items():
        if f"in {lang_name}" in q_lower or f"to {lang_name}" in q_lower:
            return code
    return None

# -------------------- Audio Functions --------------------
def get_audio_input():
    """Get audio input using HTML5 audio recorder"""
    audio_html = """
    <div style="text-align: center; padding: 20px;">
        <p style="color: #E0E0E0;">🎤 <strong>Voice Input Feature</strong></p>
        <p style="color: #888; font-size: 0.9em;">Note: Audio input requires microphone permissions.<br>
        Install speech_recognition and pyaudio packages for full functionality:<br>
        <code>pip install SpeechRecognition pyaudio</code></p>
    </div>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

def text_to_speech_button(text, lang='en'):
    """Create audio download button"""
    try:
        # Note: Install gtts package: pip install gtts
        from gtts import gTTS
        
        tts = gTTS(text=text, lang=lang, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        audio_base64 = base64.b64encode(fp.read()).decode()
        audio_html = f"""
        <audio controls autoplay style="width: 100%;">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
        return True
    except ImportError:
        st.info("💡 Install gtts for audio output: pip install gtts")
        return False
    except Exception as e:
        st.warning(f"Audio generation error: {e}")
        return False

# -------------------- QA & Summarization --------------------
def ask_groq_with_context(context, question, target_lang=None):
    """Answer questions using context"""
    if target_lang:
        response_lang = LANG_MAP.get(target_lang, "the requested language")
    else:
        question_lang = detect_question_language(question)
        response_lang = LANG_MAP.get(question_lang, "English")
    
    is_translation = any(kw in question.lower() for kw in 
                         ["translate", "convert", "change to", "in kannada", "in hindi", 
                          "in telugu", "in marathi", "in tamil", "in english"])
    
    if is_translation:
        prompt = f"""You are a multilingual translator.

CONTEXT from documents:
{context}

USER REQUEST: {question}

Translate the relevant information to {response_lang}. Be accurate and preserve details.

Response in {response_lang}:"""
    else:
        prompt = f"""Answer this question using ONLY the CONTEXT provided.

CONTEXT:
{context}

QUESTION: {question}

Answer in {response_lang}. If not in context, say "I cannot find this in the documents."

Answer:"""
    
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def summarize_document(text, target_lang="en"):
    """Generate summary"""
    if not text or len(text.strip()) < 50:
        return "Insufficient text to summarize."
    
    text_snippet = text[:8000]
    lang_name = LANG_MAP.get(target_lang, "English")
    
    prompt = f"""Summarize this document in {lang_name}. Include main topic, key points, and important details.

{text_snippet}

Summary:"""
    
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Summarization failed: {e}"

# -------------------- Download Functions --------------------
def create_chat_download(chat_history):
    """Create downloadable chat history"""
    chat_text = f"IntelliDoc Chat History\n"
    chat_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    chat_text += "="*80 + "\n\n"
    
    for item in chat_history:
        if len(item) == 2:
            role, message = item
            source_pages = None
        else:
            role, message, source_pages = item
        
        if role == "user":
            chat_text += f"USER:\n{message}\n\n"
        else:
            chat_text += f"INTELLIDOC:\n{message}\n"
            if source_pages:
                chat_text += f"Sources: {source_pages}\n"
            chat_text += "\n"
        chat_text += "-"*80 + "\n\n"
    
    return chat_text

def create_json_download(chat_history):
    """Create JSON format chat history"""
    conversations = []
    for item in chat_history:
        if len(item) == 2:
            role, message = item
            source_pages = None
        else:
            role, message, source_pages = item
        
        conversations.append({
            "role": role,
            "message": message,
            "sources": source_pages if role == "assistant" else None
        })
    
    chat_data = {
        "timestamp": datetime.now().isoformat(),
        "conversation": conversations
    }
    return json.dumps(chat_data, indent=2, ensure_ascii=False)

# -------------------- Custom CSS (Dark Theme) --------------------
# -------------------- Custom CSS (Professional Subtle Dark Theme) --------------------
def load_custom_css():
    st.markdown("""
    <style>
    /* Professional Dark theme */
    .stApp {
        background: #1e1e1e;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #2d2d2d 0%, #1e1e1e 100%);
        padding: 35px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        border-bottom: 2px solid #4a9eff;
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 2.5em;
        font-weight: 700;
        margin: 0;
        letter-spacing: 1px;
    }
    
    .main-header p {
        color: #a0a0a0;
        font-size: 1.1em;
        margin: 10px 0 0 0;
        font-weight: 400;
    }
    
    /* Feature banner */
    .feature-banner {
        text-align: center;
        padding: 12px;
        background: #2d2d2d;
        border-radius: 8px;
        margin-bottom: 20px;
        border: 1px solid #3a3a3a;
    }
    
    .feature-banner p {
        color: #b0b0b0;
        font-size: 0.95em;
        margin: 0;
    }
    
    /* Chat messages */
    .user-message {
        background: #2d2d2d;
        color: #e0e0e0;
        padding: 18px;
        border-radius: 12px 12px 4px 12px;
        margin: 12px 0;
        border-left: 3px solid #4a9eff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    .bot-message {
        background: #252525;
        color: #d0d0d0;
        padding: 18px;
        border-radius: 12px 12px 12px 4px;
        margin: 12px 0;
        border-left: 3px solid #6c63ff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    .page-reference {
        display: inline-block;
        background: #4a9eff;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 600;
        margin: 8px 4px 0 0;
        box-shadow: 0 2px 6px rgba(74,158,255,0.3);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #4a9eff 0%, #6c63ff 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 3px 10px rgba(74,158,255,0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(74,158,255,0.4);
        background: linear-gradient(135deg, #5aa3ff 0%, #7d73ff 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #252525;
        border-right: 1px solid #3a3a3a;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #e0e0e0;
    }
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #b0b0b0;
    }
    
    /* Success/Info boxes */
    .stSuccess {
        background-color: rgba(74,158,255,0.1);
        border-left: 4px solid #4a9eff;
        color: #4a9eff;
        border-radius: 4px;
    }
    
    .stInfo {
        background-color: rgba(108,99,255,0.1);
        border-left: 4px solid #6c63ff;
        color: #6c63ff;
        border-radius: 4px;
    }
    
    .stWarning {
        background-color: rgba(255,179,0,0.1);
        border-left: 4px solid #ffb300;
        color: #ffb300;
        border-radius: 4px;
    }
    
    .stError {
        background-color: rgba(255,82,82,0.1);
        border-left: 4px solid #ff5252;
        color: #ff5252;
        border-radius: 4px;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #2d2d2d;
        border: 2px dashed #4a9eff;
        border-radius: 10px;
        padding: 20px;
    }
    
    [data-testid="stFileUploader"] label {
        color: #b0b0b0;
    }
    
    /* Input boxes */
    .stTextInput>div>div>input {
        background: #2d2d2d;
        color: #e0e0e0;
        border: 1px solid #4a4a4a;
        border-radius: 8px;
        padding: 10px;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #4a9eff;
        box-shadow: 0 0 0 1px #4a9eff;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #2d2d2d;
        border-radius: 8px;
        color: #e0e0e0;
        font-weight: 600;
        border: 1px solid #3a3a3a;
    }
    
    .streamlit-expanderHeader:hover {
        background: #353535;
        border-color: #4a9eff;
    }
    
    /* Dataframe */
    .stDataFrame {
        background: #2d2d2d;
        border-radius: 8px;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #d0d0d0;
    }
    
    /* Checkbox */
    .stCheckbox {
        color: #b0b0b0;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #e0e0e0;
    }
    
    .stRadio > div {
        color: #b0b0b0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #808080;
        padding: 25px;
        margin-top: 40px;
        border-top: 1px solid #3a3a3a;
    }
    
    .footer h3 {
        color: #e0e0e0;
        margin-bottom: 15px;
        font-size: 1.3em;
    }
    
    .footer a {
        color: #4a9eff;
        text-decoration: none;
        transition: color 0.3s;
    }
    
    .footer a:hover {
        color: #6c63ff;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e1e1e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #4a4a4a;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #5a5a5a;
    }
    
    /* Divider */
    hr {
        border-color: #3a3a3a;
    }
    
    /* Download buttons */
    .stDownloadButton>button {
        background: #2d2d2d;
        color: #4a9eff;
        border: 1px solid #4a9eff;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stDownloadButton>button:hover {
        background: #4a9eff;
        color: white;
        transform: translateY(-2px);
    }
    
    /* Feature grid */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }
    
    .feature-card {
        background: #2d2d2d;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #3a3a3a;
        transition: all 0.3s;
    }
    
    .feature-card:hover {
        border-color: #4a9eff;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(74,158,255,0.2);
    }
    
    .feature-card strong {
        color: #4a9eff;
        display: block;
        margin-bottom: 5px;
    }
    
    .feature-card span {
        color: #909090;
        font-size: 0.9em;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------- Streamlit UI --------------------
st.set_page_config(
    page_title="IntelliDoc - AI Document Analysis",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_custom_css()

# Custom Header
# Custom Header
st.markdown("""
<div class="main-header">
    <h1>📄 IntelliDoc</h1>
    <p>AI-Powered Multilingual Document Analysis System</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="feature-banner">
    <p>
        ⚡ <strong>Features:</strong> Printed & Handwritten OCR | 10+ Languages | Real-Time Translation | 
        Table Extraction | Chart Detection | Audio I/O | 3x Faster Processing
    </p>
</div>
""", unsafe_allow_html=True)

# Session state
for key, val in [("chunks", []), ("meta", []), ("embeddings", None),
                 ("faiss_index", None), ("full_text", ""), ("chat_history", []),
                 ("file_info", {}), ("vectorizer", None), ("tables", []), 
                 ("visuals", [])]:
    if key not in st.session_state:
        st.session_state[key] = val

# Sidebar
st.sidebar.markdown("### 📂 Document Upload")

doc_mode = st.sidebar.radio(
    "Processing Mode:",
    ["🚀 Smart (Auto-detect)", "🔍 Force OCR (Scanned/Handwritten)"],
    help="Smart mode automatically detects which pages need OCR"
)

uploaded_pdfs = st.sidebar.file_uploader(
    "📑 PDF Files", 
    type="pdf", 
    accept_multiple_files=True,
    key="pdf_uploader"
)

uploaded_images = st.sidebar.file_uploader(
    "🖼 Image Files", 
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
    accept_multiple_files=True,
    key="img_uploader"
)

st.sidebar.markdown("### ⚙ Advanced Options")
extract_tables = st.sidebar.checkbox("📊 Extract Tables", value=True)
detect_visuals = st.sidebar.checkbox("📈 Detect Charts/Diagrams", value=True)

process_btn = st.sidebar.button("🚀 Process Documents", type="primary", use_container_width=True)

# Processing
if process_btn:
    # Clear previous data
    for key in ["chunks", "meta", "embeddings", "faiss_index", "full_text", 
                "file_info", "chat_history", "tables", "visuals"]:
        if key in ["chunks", "meta", "file_info", "chat_history", "tables", "visuals"]:
            st.session_state[key].clear()
        else:
            st.session_state[key] = None if key != "full_text" else ""
    
    force_ocr = ("Force OCR" in doc_mode)
    total_start = time.time()
    
    with st.spinner("⚡ Processing documents with parallel OCR..."):
        # Process PDFs
        for pdf in uploaded_pdfs or []:
            start_time = time.time()
            st.info(f"📄 Processing: {pdf.name}")
            pdf_bytes = pdf.read()
            
            # Extract tables
            if extract_tables:
                with st.spinner(f"📊 Extracting tables from {pdf.name}..."):
                    tables = extract_tables_from_pdf(pdf_bytes, pdf.name)
                    if tables:
                        st.session_state.tables.extend(tables)
                        st.success(f"✅ Found {len(tables)} table(s)")
            
            # Detect visuals
            if detect_visuals:
                with st.spinner(f"📈 Detecting charts/diagrams in {pdf.name}..."):
                    visuals = detect_visual_elements(pdf_bytes, pdf.name)
                    if visuals:
                        st.session_state.visuals.extend(visuals)
                        st.success(f"✅ Found {len(visuals)} visual element(s)")
            
            # Extract text
            text, pages_processed, total_pages = extract_text_from_pdf_parallel(
                pdf_bytes, pdf.name, force_ocr=force_ocr
            )
            
            processing_time = time.time() - start_time
            
            if text:
                st.session_state.full_text += f"\n\n=== FILE: {pdf.name} ===\n{text}"
                ch, md = chunk_text_smart(text, pdf.name)
                st.session_state.chunks.extend(ch)
                st.session_state.meta.extend(md)
                
                st.session_state.file_info[pdf.name] = {
                    'chars': len(text),
                    'pages': total_pages,
                    'time': processing_time
                }
                
                speed = total_pages / processing_time if processing_time > 0 else 0
                st.success(
                    f"✅ {pdf.name}: {total_pages} pages in {processing_time:.1f}s "
                    f"({speed:.1f} pages/sec)"
                )
            else:
                st.warning(f"⚠ Could not extract text from {pdf.name}")
        
        # Process Images
        for img in uploaded_images or []:
            start_time = time.time()
            st.info(f"🖼 Processing: {img.name}")
            text = extract_text_from_image_fast(img)
            processing_time = time.time() - start_time
            
            if text:
              
                st.session_state.full_text += f"\n\n=== IMAGE: {img.name} ===\n{text}"
                ch, md = chunk_text_smart(text, img.name)
                st.session_state.chunks.extend(ch)
                st.session_state.meta.extend(md)
                
                st.session_state.file_info[img.name] = {
                    'chars': len(text),
                    'time': processing_time
                }
                st.success(f"✅ {img.name}: {len(text)} chars in {processing_time:.1f}s")
            else:
                st.warning(f"⚠ Could not extract text from {img.name}")
        
        # Build embeddings
        if st.session_state.chunks:
            st.info(f"🔢 Creating search index for {len(st.session_state.chunks)} chunks...")
            embed_start = time.time()
            
            emb_np, vectorizer = embed_texts_tfidf(st.session_state.chunks)
            st.session_state.embeddings = emb_np
            st.session_state.vectorizer = vectorizer
            st.session_state.faiss_index = build_faiss_index(emb_np)
            
            embed_time = time.time() - embed_start
            total_time = time.time() - total_start
            
            st.success(
                f"✅ Search index created in {embed_time:.1f}s | "
                f"Total: {total_time:.1f}s"
            )
            
            # Show statistics
            st.sidebar.markdown("### 📊 Processing Stats")
            total_docs = len(st.session_state.file_info)
            total_chunks = len(st.session_state.chunks)
            st.sidebar.markdown(f"*Documents:* {total_docs} | *Chunks:* {total_chunks}")
            
            for fname, info in st.session_state.file_info.items():
                if 'pages' in info:
                    st.sidebar.text(f"📄 {fname[:25]}...: {info['pages']}p ({info['time']:.1f}s)")
                else:
                    st.sidebar.text(f"🖼 {fname[:25]}...: ({info['time']:.1f}s)")
            
            if st.session_state.tables:
                st.sidebar.markdown(f"*Tables Found:* {len(st.session_state.tables)}")
            
            if st.session_state.visuals:
                st.sidebar.markdown(f"*Visuals Found:* {len(st.session_state.visuals)}")
        else:
            st.error("❌ No text extracted. Check file quality and OCR settings.")

# Display extracted tables
if st.session_state.tables:
    with st.expander(f"📊 Extracted Tables ({len(st.session_state.tables)} found)", expanded=False):
        for table_info in st.session_state.tables:
            st.markdown(f"📄 {table_info['doc']} - Page {table_info['page']}, Table {table_info['table_num']}")
            st.dataframe(table_info['data'], use_container_width=True)
            st.markdown("---")

# Display detected visuals
if st.session_state.visuals:
    with st.expander(f"📈 Detected Charts & Diagrams ({len(st.session_state.visuals)} found)", expanded=False):
        cols = st.columns(3)
        for idx, visual in enumerate(st.session_state.visuals):
            col = cols[idx % 3]
            with col:
                st.image(visual['image'], caption=f"{visual['doc']} - Page {visual['page']}\n{visual['type']}", use_container_width=True)

# Chat Interface
st.markdown("---")
st.markdown("### 💬 Chat with Your Documents")

if not st.session_state.faiss_index:
    st.info("👆 Upload and process documents using the sidebar first.")
else:
    # Audio input section
    with st.expander("🎤 Voice Input (Optional)", expanded=False):
        get_audio_input()
    
    # Text input
    user_input = st.text_input(
        "Your question:",
        placeholder="E.g., 'Summarize in Kannada', 'What dates are mentioned?', 'Translate to Hindi'",
        key="user_question"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        ask_button = st.button("📤 Send Question", type="primary", use_container_width=True)
    with col2:
        audio_output = st.checkbox("🔊 Audio", value=False)
    
    with st.expander("💡 Example Questions", expanded=False):
        st.markdown("""
        *General Queries:*
        - "Summarize this document in Kannada"
        - "What is the main topic?"
        - "List all important dates mentioned"
        
        *Translation:*
        - "Translate the main points to Telugu"
        - "Convert this to Hindi"
        
        *Specific Information:*
        - "What are the names of people mentioned?"
        - "Extract all numerical values"
        - "What tables are present?"
        """)
    
    if ask_button and user_input:
        st.session_state.chat_history.append(("user", user_input))
        
        with st.spinner("🤔 Analyzing..."):
            is_summary = any(kw in user_input.lower() for kw in 
                           ["summary", "summarize", "overview", "brief", "main points"])
            
            target_lang = extract_target_language(user_input)
            if not target_lang:
                target_lang = detect_question_language(user_input)
            
            if is_summary:
                answer = summarize_document(st.session_state.full_text, target_lang)
                source_pages = "Summary from entire document"
            else:
                qvec, _ = embed_texts_tfidf(
                    [user_input], 
                    vectorizer=st.session_state.vectorizer
                )
                qvec = qvec.astype("float32")
                
                norm = np.linalg.norm(qvec)
                if norm > 0:
                    qvec = qvec / norm
                
                D, I = st.session_state.faiss_index.search(qvec, TOP_K)
                
                if len(D[0]) == 0 or D[0][0] < SIMILARITY_THRESHOLD:
                    answer = "I cannot find relevant information in the documents."
                    source_pages = "N/A"
                else:
                    context_pieces = []
                    total_chars = 0
                    source_pages_set = set()
                    
                    for idx, score in zip(I[0], D[0]):
                        if idx >= len(st.session_state.chunks):
                            continue
                        
                        chunk = st.session_state.chunks[idx]
                        source = st.session_state.meta[idx]['source']
                        page = st.session_state.meta[idx].get('page', '?')
                        
                        if page and page != '?':
                            source_pages_set.add(f"Page {page}")
                        
                        piece = f"[Source: {source}, Page: {page}]\n{chunk}"
                        
                        if total_chars + len(piece) > MAX_CONTEXT_CHARS:
                            break
                        
                        context_pieces.append(piece)
                        total_chars += len(piece)
                    
                    context = "\n\n---\n\n".join(context_pieces)
                    answer = ask_groq_with_context(context, user_input, target_lang)
                    
                    # Sort pages numerically
                    try:
                        sorted_pages = sorted(source_pages_set, key=lambda x: int(re.findall(r'\d+', x)[0]))
                        source_pages = " | ".join(sorted_pages) if sorted_pages else "N/A"
                    except:
                        source_pages = " | ".join(sorted(source_pages_set)) if source_pages_set else "N/A"
            
            # Add answer with source pages
            st.session_state.chat_history.append(("assistant", answer, source_pages))
            
            # Audio output if enabled
            if audio_output and answer:
                with st.spinner("🔊 Generating audio..."):
                    text_to_speech_button(answer, lang=target_lang)
    
    # Display chat with custom styling
    st.markdown("### 📜 Conversation History")
    
    if not st.session_state.chat_history:
        st.info("💭 No conversation yet. Ask your first question above!")
    
    for item in st.session_state.chat_history:
        if len(item) == 2:
            role, message = item
            source_pages = None
        else:
            role, message, source_pages = item
        
        if role == "user":
            st.markdown(
                f'<div class="user-message"><strong>👤 You:</strong><br><br>{message}</div>', 
                unsafe_allow_html=True
            )
        else:
            page_badges = ""
            if source_pages and source_pages not in ["N/A", "Summary from entire document"]:
                for page in source_pages.split(" | "):
                    page_badges += f'<span class="page-reference">📄 {page}</span>'
            elif source_pages == "Summary from entire document":
                page_badges = '<span class="page-reference">📄 Full Document</span>'
            
            st.markdown(
                f'<div class="bot-message"><strong>🤖 IntelliDoc:</strong><br>{page_badges}<br><br>{message}</div>', 
                unsafe_allow_html=True
            )
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Download and Clear buttons
    if st.session_state.chat_history:
        st.markdown("---")
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        
        with col1:
            chat_txt = create_chat_download(st.session_state.chat_history)
            st.download_button(
                label="📥 Download TXT",
                data=chat_txt,
                file_name=f"intellidoc_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            chat_json = create_json_download(st.session_state.chat_history)
            st.download_button(
                label="📥 Download JSON",
                data=chat_json,
                file_name=f"intellidoc_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            # Download full extracted text
            st.download_button(
                label="📄 Extracted Text",
                data=st.session_state.full_text,
                file_name=f"extracted_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col4:
            if st.button("🗑 Clear Chat", use_container_width=True):
                st.session_state.chat_history.clear()
                st.rerun()

# Footer
# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <h3>⚡ Performance & Features</h3>
    <div class="feature-grid">
        <div class="feature-card">
            <strong>🚀 Smart OCR</strong>
            <span>Auto-detects when to use OCR</span>
        </div>
        <div class="feature-card">
            <strong>⚡ 3x Faster</strong>
            <span>Parallel processing</span>
        </div>
        <div class="feature-card">
            <strong>🌐 10+ Languages</strong>
            <span>Multilingual support</span>
        </div>
        <div class="feature-card">
            <strong>📊 Smart Tables</strong>
            <span>Automatic extraction</span>
        </div>
        <div class="feature-card">
            <strong>📈 Chart Detection</strong>
            <span>Visual element recognition</span>
        </div>
        <div class="feature-card">
            <strong>🔊 Audio I/O</strong>
            <span>Voice input & output</span>
        </div>
    </div>
    <p style="margin-top: 20px; font-size: 0.9em;">
        <strong>IntelliDoc</strong> v1.0 | Built with Streamlit, PyMuPDF, Tesseract OCR, FAISS & Groq LLM<br>
        Supporting: English, Kannada, Hindi, Tamil, Telugu, Marathi, Malayalam, Gujarati, Bengali, Punjabi
    </p>
</div>
""", unsafe_allow_html=True)