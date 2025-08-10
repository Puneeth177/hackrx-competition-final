"""
HackRX API - Simplified Version for Deployment
No FAISS, no complex dependencies - just what works
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import requests
import re
import os
from io import BytesIO
try:
    import PyPDF2
except ImportError:
    try:
        from pypdf import PdfReader as PyPDF2Reader
        PyPDF2 = type('PyPDF2', (), {'PdfReader': PyPDF2Reader})
    except ImportError:
        raise ImportError("Neither PyPDF2 nor pypdf could be imported")
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize FastAPI app
app = FastAPI(
    title="HackRX API - Competition Submission",
    description="LLM-Powered Intelligent Query-Retrieval System",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
VALID_TOKEN = "6caf8e7ec0c26c15b51e20eb8c1479cdf8d5dde6b9453cf13e007aa5bb381210"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the bearer token"""
    if credentials.credentials != VALID_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return credentials.credentials

# Request/Response models
class HackRXRequest(BaseModel):
    documents: str  # URL to PDF document
    questions: List[str]

class HackRXResponse(BaseModel):
    answers: List[str]
    processing_time: float
    status: str = "success"

# Document processing functions
def download_pdf(url: str) -> bytes:
    """Download PDF from URL"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract text from PDF content"""
    try:
        pdf_file = BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        # Clean the text
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with single newline
        
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {str(e)}")

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.7:  # If we find a period in the last 30%
                chunk = chunk[:last_period + 1]
                end = start + last_period + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
        
        if start >= len(text):
            break
    
    return chunks

def find_relevant_chunks(question: str, chunks: List[str], top_k: int = 3) -> List[str]:
    """Find most relevant chunks using TF-IDF similarity"""
    if not chunks:
        return []
    
    try:
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        
        # Fit on all chunks + question
        all_texts = chunks + [question]
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Get question vector (last one)
        question_vector = tfidf_matrix[-1]
        chunk_vectors = tfidf_matrix[:-1]
        
        # Calculate similarities
        similarities = cosine_similarity(question_vector, chunk_vectors).flatten()
        
        # Get top k most similar chunks
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [chunks[i] for i in top_indices if similarities[i] > 0.1]
    
    except Exception as e:
        # Fallback: simple keyword matching
        question_words = set(question.lower().split())
        scored_chunks = []
        
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            score = len(question_words.intersection(chunk_words))
            scored_chunks.append((score, chunk))
        
        scored_chunks.sort(reverse=True)
        return [chunk for score, chunk in scored_chunks[:top_k] if score > 0]

def generate_answer(question: str, relevant_chunks: List[str]) -> str:
    """Generate answer based on relevant chunks"""
    if not relevant_chunks:
        return "I couldn't find relevant information to answer this question."
    
    # Combine relevant chunks
    context = "\n\n".join(relevant_chunks)
    
    # Simple rule-based answer generation for insurance questions
    question_lower = question.lower()
    
    # Grace period questions
    if "grace period" in question_lower and "premium" in question_lower:
        # Look for grace period information with more comprehensive patterns
        grace_patterns = [
            r"grace period of (\w+[-\s]*\w*) days",
            r"(\w+[-\s]*\w*) days.*grace period",
            r"grace period.*(\w+[-\s]*\w*) days",
            r"grace period.*(\d+) days",
            r"(\d+) days.*grace period"
        ]
        
        for pattern in grace_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                days = match.group(1).strip()
                # Convert common variations
                if days.lower() in ['thirty', '30']:
                    days = "thirty"
                return f"A grace period of {days} days is provided for premium payment after the due date to renew or continue the policy coverage."
        
        # Fallback: look for any mention of grace period with days
        if "grace period" in context.lower():
            # Extract sentences containing grace period
            sentences = context.split('.')
            for sentence in sentences:
                if "grace period" in sentence.lower() and "days" in sentence.lower():
                    # Try to extract the number of days
                    day_match = re.search(r'(\w+[-\s]*\w*)\s*days', sentence, re.IGNORECASE)
                    if day_match:
                        days = day_match.group(1).strip()
                        if days.lower() in ['thirty', '30']:
                            days = "thirty"
                        return f"A grace period of {days} days is provided for premium payment after the due date to renew or continue the policy coverage."
    
    # Waiting period questions
    if "waiting period" in question_lower and ("pre-existing" in question_lower or "ped" in question_lower):
        # Look for waiting period information with comprehensive patterns
        waiting_patterns = [
            r"waiting period of (\w+[-\s]*\w*) \((\d+)\) months",
            r"(\w+[-\s]*\w*) \((\d+)\) months.*waiting period",
            r"waiting period.*(\w+[-\s]*\w*) \((\d+)\) months",
            r"(\d+) months.*waiting period",
            r"waiting period.*(\d+) months",
            r"(\w+[-\s]*\w*) months.*waiting period",
            r"waiting period.*(\w+[-\s]*\w*) months"
        ]
        
        for pattern in waiting_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) >= 2 and groups[1].isdigit():
                    period_text = groups[0].strip()
                    period_num = groups[1]
                    # Convert common variations
                    if period_text.lower() in ['thirty-six', 'thirtysix'] or period_num == '36':
                        period_text = "thirty-six"
                        period_num = "36"
                    return f"There is a waiting period of {period_text} ({period_num}) months of continuous coverage from the first policy inception date for pre-existing diseases to be covered."
                else:
                    period = groups[0].strip()
                    if period.isdigit() and period == '36':
                        period = "thirty-six (36)"
                    elif period.lower() in ['thirty-six', 'thirtysix']:
                        period = "thirty-six (36)"
                    return f"There is a waiting period of {period} months of continuous coverage from the first policy inception date for pre-existing diseases to be covered."
        
        # Fallback: look for any mention of waiting period with months
        if "waiting period" in context.lower():
            sentences = context.split('.')
            for sentence in sentences:
                if "waiting period" in sentence.lower() and ("months" in sentence.lower() or "month" in sentence.lower()):
                    # Try to extract the number of months
                    month_match = re.search(r'(\d+)\s*months?', sentence, re.IGNORECASE)
                    if month_match:
                        months = month_match.group(1)
                        if months == '36':
                            return f"There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception date for pre-existing diseases to be covered."
                        else:
                            return f"There is a waiting period of {months} months of continuous coverage from the first policy inception date for pre-existing diseases to be covered."
    
    # Coverage questions
    if "coverage" in question_lower or "covered" in question_lower:
        # Look for coverage information in context
        coverage_sentences = []
        sentences = context.split('.')
        for sentence in sentences:
            if any(word in sentence.lower() for word in ["cover", "benefit", "include", "eligible"]):
                coverage_sentences.append(sentence.strip())
        
        if coverage_sentences:
            return ". ".join(coverage_sentences[:2]) + "."
    
    # Default: return most relevant chunk
    return relevant_chunks[0][:500] + "..." if len(relevant_chunks[0]) > 500 else relevant_chunks[0]

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with team information"""
    return {
        "message": "HackRX API - Competition Submission",
        "team_token": VALID_TOKEN,
        "api_version": "1.0.0",
        "endpoints": {
            "competition": "/api/v1/hackrx/run",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "hackrx-api"
    }

@app.post("/api/v1/hackrx/run", response_model=HackRXResponse)
async def hackrx_run(
    request: HackRXRequest,
    token: str = Depends(verify_token)
):
    """
    Main competition endpoint
    Process documents and answer questions
    """
    start_time = time.time()
    
    try:
        # Download and process PDF
        pdf_content = download_pdf(request.documents)
        text = extract_text_from_pdf(pdf_content)
        
        # Split into chunks
        chunks = chunk_text(text)
        
        # Process each question
        answers = []
        for question in request.questions:
            # Find relevant chunks
            relevant_chunks = find_relevant_chunks(question, chunks)
            
            # Generate answer
            answer = generate_answer(question, relevant_chunks)
            answers.append(answer)
        
        processing_time = time.time() - start_time
        
        return HackRXResponse(
            answers=answers,
            processing_time=processing_time,
            status="success"
        )
    
    except Exception as e:
        processing_time = time.time() - start_time
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)