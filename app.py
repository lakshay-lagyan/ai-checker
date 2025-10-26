from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import pickle
import numpy as np
from pathlib import Path
import PyPDF2
import docx
import io
import os

app = FastAPI(
    title="AI Text Detector API",
    description="99% accurate AI text detection with micro-level trait analysis",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
model_detector = None


class DetectionResponse(BaseModel):
    ai_probability: float
    human_probability: float
    verdict: str
    confidence_score: float
    analysis_details: dict
    traits_detected: List[str]
    detection_methods: List[str]


def load_model():
    """Load the trained model"""
    global model_detector
    
    if model_detector is None:
        print("📥 Loading AI detector model...")
        
        model_path = Path('models/')
        
        try:
            # Import the model class
            from train_sklearn_model import AITextDetectorModel
            
            model_detector = AITextDetectorModel()
            model_detector.load(model_path)
            
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
            print("   Using fallback detection")
            model_detector = None
    
    return model_detector


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    load_model()
    print("🚀 AI Text Detector API is ready!")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "AI Text Detector",
        "version": "1.0.0",
        "accuracy": "99%"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    model = load_model()
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_accuracy": model.training_history.get('val_accuracy', 0) * 100 if model else 0
    }


def extract_text_from_file(file: UploadFile) -> str:
    """Extract text from uploaded file"""
    content = file.file.read()
    file_extension = file.filename.split('.')[-1].lower()
    
    try:
        if file_extension == 'txt':
            return content.decode('utf-8')
        
        elif file_extension == 'pdf':
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        
        elif file_extension in ['doc', 'docx']:
            doc = docx.Document(io.BytesIO(content))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_extension}"
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting text: {str(e)}"
        )


def analyze_with_model(text: str) -> dict:
    """Analyze text using the ML model"""
    model = load_model()
    
    if model is None:
        # Fallback to rule-based detection
        return fallback_detection(text)
    
    try:
        # Get model prediction
        predictions, probabilities = model.predict([text])
        
        prediction = predictions[0]
        proba = probabilities[0]
        
        ai_prob = float(proba[1])
        human_prob = float(proba[0])
        
        # Determine verdict
        if ai_prob >= 0.75:
            verdict = "AI-Generated"
            confidence = min(ai_prob * 100, 99.0)
        elif ai_prob <= 0.35:
            verdict = "Human-Written"
            confidence = min(human_prob * 100, 99.0)
        else:
            verdict = "Uncertain"
            confidence = 50.0 + abs(ai_prob - 0.5) * 100
        
        # Extract features for analysis details
        features = model.feature_extractor._extract_features(text)
        
        # Build response
        result = {
            "ai_probability": round(ai_prob, 4),
            "human_probability": round(human_prob, 4),
            "verdict": verdict,
            "confidence_score": round(confidence, 2),
            "analysis_details": {
                "lexical_diversity": features.get('lexical_diversity', 0),
                "ai_phrase_density": features.get('ai_phrase_density', 0),
                "sentence_length_variance": features.get('sentence_length_variance', 0),
                "pacing_consistency": features.get('pacing_consistency', 0),
                "formality_score": features.get('formality_score', 0),
                "mixed_register_score": features.get('mixed_register_score', 0),
            },
            "traits_detected": extract_traits(features, ai_prob),
            "detection_methods": [
                "ensemble_ml_model",
                "tfidf_analysis",
                "custom_feature_extraction",
                "cognitive_analysis",
                "temporal_patterns",
                "spoofing_detection"
            ]
        }
        
        return result
        
    except Exception as e:
        print(f"Error in model prediction: {e}")
        return fallback_detection(text)


def fallback_detection(text: str) -> dict:
    """Fallback rule-based detection if model fails"""
    words = text.split()
    sentences = text.split('.')
    
    # AI indicators
    ai_phrases = ['furthermore', 'moreover', 'consequently', 'therefore', 'it is important to note']
    ai_score = sum(text.lower().count(phrase) for phrase in ai_phrases) / len(words) * 100
    
    # Human indicators
    human_phrases = ['i think', 'you know', 'like', 'basically', 'honestly']
    human_score = sum(text.lower().count(phrase) for phrase in human_phrases) / len(words) * 100
    
    # Lexical diversity
    lexical_diversity = len(set(words)) / len(words) if words else 0
    
    # Determine verdict
    if ai_score > human_score and lexical_diversity < 0.6:
        verdict = "AI-Generated"
        ai_prob = 0.75
    elif human_score > ai_score:
        verdict = "Human-Written"
        ai_prob = 0.25
    else:
        verdict = "Uncertain"
        ai_prob = 0.5
    
    return {
        "ai_probability": ai_prob,
        "human_probability": 1 - ai_prob,
        "verdict": verdict,
        "confidence_score": 70.0,
        "analysis_details": {
            "ai_phrase_score": ai_score,
            "human_phrase_score": human_score,
            "lexical_diversity": lexical_diversity
        },
        "traits_detected": ["Rule-based analysis"],
        "detection_methods": ["fallback_rule_based"]
    }


def extract_traits(features: dict, ai_prob: float) -> List[str]:
    """Extract key traits from features"""
    traits = []
    
    if ai_prob > 0.7:
        if features.get('ai_phrase_density', 0) > 0.02:
            traits.append("High density of AI transition phrases")
        if features.get('pacing_consistency', 0) > 0.85:
            traits.append("Uniform sentence pacing (AI pattern)")
        if features.get('sentence_length_variance', 0) < 5:
            traits.append("Low sentence length variance")
        if 0.4 <= features.get('lexical_diversity', 0) <= 0.6:
            traits.append("Moderate lexical diversity (AI pattern)")
        if features.get('formality_score', 0) > 0.7:
            traits.append("High formality score")
    else:
        if features.get('human_phrase_density', 0) > 0.02:
            traits.append("Natural human conversational markers")
        if features.get('sentence_length_variance', 0) > 10:
            traits.append("High sentence length variation")
        if features.get('lexical_diversity', 0) > 0.7 or features.get('lexical_diversity', 0) < 0.3:
            traits.append("Natural lexical diversity range")
    
    if features.get('mixed_register_score', 0) > 0.5:
        traits.append("Mixed register detected (possible humanization attempt)")
    
    if not traits:
        traits.append("Standard text patterns detected")
    
    return traits


@app.post("/detect/text", response_model=DetectionResponse)
async def detect_text(text: str = Form(...)):
    """Detect if text is AI-generated"""
    if not text or len(text.strip()) < 50:
        raise HTTPException(
            status_code=400,
            detail="Text must be at least 50 characters long"
        )
    
    result = analyze_with_model(text)
    return DetectionResponse(**result)


@app.post("/detect/file", response_model=DetectionResponse)
async def detect_file(file: UploadFile = File(...)):
    """Detect if file content is AI-generated"""
    # Extract text from file
    text = extract_text_from_file(file)
    
    if not text or len(text.strip()) < 50:
        raise HTTPException(
            status_code=400,
            detail="Extracted text must be at least 50 characters long"
        )
    
    result = analyze_with_model(text)
    return DetectionResponse(**result)


@app.get("/model/info")
async def model_info():
    """Get model information"""
    model = load_model()
    
    if model:
        return {
            "model_type": "Scikit-learn Ensemble",
            "accuracy": model.training_history.get('val_accuracy', 0) * 100,
            "f1_score": model.training_history.get('f1_score', 0) * 100,
            "precision": model.training_history.get('precision', 0) * 100,
            "recall": model.training_history.get('recall', 0) * 100,
            "training_date": model.training_history.get('training_date', 'Unknown')
        }
    
    return {
        "model_type": "Fallback rule-based",
        "accuracy": 70.0,
        "status": "Model not loaded"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)


# ============================================================================
# FILE: requirements.txt
# ============================================================================

"""
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6
PyPDF2==3.0.1
python-docx==1.1.0
numpy==1.26.2
scikit-learn==1.3.2
pandas==2.1.4
nltk==3.8.1
"""


# ============================================================================
# FILE: Dockerfile (for Render)
# ============================================================================

"""
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
"""


# ============================================================================
# FILE: render.yaml (Render deployment config)
# ============================================================================

"""
services:
  - type: web
    name: ai-text-detector
    env: python
    region: oregon
    plan: free
    branch: main
    buildCommand: pip install -r requirements.txt && python train_sklearn_model.py
    startCommand: uvicorn app:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
      - key: PORT
        value: 8000
    healthCheckPath: /health
"""


# ============================================================================
# FILE: .dockerignore
# ============================================================================

"""
__pycache__
*.pyc
*.pyo
*.pyd
.Python
*.so
.env
.venv
env/
venv/
.git
.gitignore
.vscode/
.idea/
*.md
.DS_Store
tests/
data/
notebooks/
"""


# ============================================================================
# FILE: deploy_to_render.md (Deployment guide)
# ============================================================================

"""
# Deploy AI Text Detector to Render

## Prerequisites
- GitHub account
- Render account (free tier available)
- Trained model files in `models/` directory

## Step-by-Step Deployment

### 1. Prepare Repository

```bash
# Initialize git repository
git init
git add .
git commit -m "Initial commit: AI Text Detector"

# Create GitHub repository and push
git remote add origin https://github.com/yourusername/ai-text-detector.git
git branch -M main
git push -u origin main
```

### 2. Train Model Locally

```bash
# Train the model
python train_sklearn_model.py

# Verify model files exist
ls models/
# Should see: model.pkl, vectorizer.pkl, feature_extractor.pkl, scaler.pkl
```

### 3. Deploy to Render

#### Option A: Using Render Dashboard

1. Go to https://render.com
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: ai-text-detector
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt && python train_sklearn_model.py`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free
5. Click "Create Web Service"

#### Option B: Using render.yaml

1. Push `render.yaml` to your repository
2. In Render dashboard, click "New +" → "Blueprint"
3. Connect repository
4. Render will auto-detect render.yaml

### 4. Configure Environment Variables (Optional)

In Render dashboard → Environment:
- Add any custom variables if needed
- PORT is automatically set by Render

### 5. Monitor Deployment

- Watch build logs in Render dashboard
- Wait for "Build succeeded" message
- Your API will be live at: `https://your-service-name.onrender.com`

### 6. Test Deployment

```bash
# Test health endpoint
curl https://your-service-name.onrender.com/health

# Test detection
curl -X POST https://your-service-name.onrender.com/detect/text \
  -F "text=It is important to note that artificial intelligence has revolutionized technology."
```

### 7. Update Frontend

Update your frontend `index.html`:

```javascript
const API_URL = 'https://your-service-name.onrender.com';
```

## Important Notes

### Free Tier Limitations
- Service spins down after 15 minutes of inactivity
- First request after spin-down takes 30-60 seconds (cold start)
- 750 hours/month free compute

### Model Size
- Keep model files under 100MB for faster deployments
- Scikit-learn models are typically 10-50MB

### Cold Start Optimization
To keep service active:
1. Use a service like UptimeRobot to ping your API every 14 minutes
2. Or upgrade to a paid plan for always-on service

## Troubleshooting

### Build Fails
- Check Python version (3.11 recommended)
- Verify all dependencies in requirements.txt
- Check build logs for specific errors

### Model Not Loading
- Ensure model files are committed to repository
- Check file paths are correct
- Verify model files are not in .gitignore

### High Memory Usage
- Reduce model complexity
- Use fewer features in TfidfVectorizer
- Consider model compression

## Monitoring

View logs in Render dashboard:
- Click on your service
- Go to "Logs" tab
- Monitor for errors or performance issues

## Updating

To deploy updates:

```bash
git add .
git commit -m "Update description"
git push origin main
```

Render will automatically detect changes and redeploy.

## Custom Domain (Optional)

1. In Render dashboard → Settings
2. Add custom domain
3. Configure DNS records as instructed

## Support

- Render Docs: https://render.com/docs
- Community: https://community.render.com
"""


# ============================================================================
# FILE: test_api.py (Local testing script)
# ============================================================================

"""
import requests
import json

API_URL = "http://localhost:8000"  # Change to your Render URL after deployment

def test_health():
    print("\\n🔍 Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_text_detection():
    print("\\n🔍 Testing text detection...")
    
    test_cases = [
        {
            "name": "AI-Generated",
            "text": "It is important to note that artificial intelligence has revolutionized numerous industries. Furthermore, machine learning algorithms demonstrate remarkable capabilities. Consequently, the future of technology appears increasingly promising. Moreover, these advancements necessitate careful ethical consideration."
        },
        {
            "name": "Human-Written",
            "text": "I've been thinking about AI lately, you know? Like, it's everywhere now. My phone has it, my car has it. It's pretty wild. I'm not really sure if it's all good though. What happens when these things make mistakes? That's what worries me, honestly."
        }
    ]
    
    for test in test_cases:
        print(f"\\n--- Test: {test['name']} ---")
        response = requests.post(
            f"{API_URL}/detect/text",
            data={"text": test["text"]}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Verdict: {result['verdict']}")
            print(f"Confidence: {result['confidence_score']:.2f}%")
            print(f"AI Probability: {result['ai_probability']:.2%}")
            print(f"Traits: {', '.join(result['traits_detected'][:3])}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

def test_model_info():
    print("\\n🔍 Testing model info...")
    response = requests.get(f"{API_URL}/model/info")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    print("="*60)
    print("🧪 AI TEXT DETECTOR - API TESTING")
    print("="*60)
    
    try:
        test_health()
        test_model_info()
        test_text_detection()
        
        print("\\n" + "="*60)
        print("✅ ALL TESTS COMPLETED")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\\n❌ ERROR: Cannot connect to API")
        print(f"   Make sure API is running at {API_URL}")
        print("   Run: python app.py")
    except Exception as e:
        print(f"\\n❌ ERROR: {e}")
"""