"""
FastAPI Application for Credit Risk Model
Task 6: Model Deployment
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import os
import sys
from typing import List

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Pydantic models
from api.pydantic_models import (
    PredictionRequest, PredictionResponse, HealthResponse, CustomerFeatures
)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit risk using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded model and preprocessor
model = None
preprocessor = None

def load_model():
    """Load the trained model and preprocessor"""
    global model, preprocessor
    
    try:
        model_path = os.path.join(os.path.dirname(__file__), '../../models/best_credit_risk_model.pkl')
        preprocessor_path = os.path.join(os.path.dirname(__file__), '../../models/preprocessor.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        print(f"✅ Model loaded: {type(model).__name__}")
        print(f"✅ Preprocessor loaded")
        
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    print("="*60)
    print("STARTING CREDIT RISK PREDICTION API")
    print("="*60)
    load_model()

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Credit Risk Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health_check": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    if model is not None and preprocessor is not None:
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_type=type(model).__name__ if model else None,
            message="API is running and model is loaded"
        )
    else:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            message="Model not loaded. Check model files."
        )

def calculate_credit_score(risk_probability: float) -> int:
    """
    Convert risk probability to credit score (300-850)
    Lower probability = higher score
    """
    # Invert probability: low risk (0.1) -> high score (800), high risk (0.9) -> low score (400)
    base_score = 300
    max_score = 850
    score_range = max_score - base_score
    
    # Invert and scale: (1 - risk) * range + base
    credit_score = int((1 - risk_probability) * score_range + base_score)
    
    # Ensure within bounds
    credit_score = max(base_score, min(max_score, credit_score))
    
    return credit_score

def get_risk_category(risk_probability: float) -> str:
    """Categorize risk based on probability"""
    if risk_probability < 0.3:
        return "LOW"
    elif risk_probability < 0.7:
        return "MEDIUM"
    else:
        return "HIGH"

def get_recommendation(risk_category: str, credit_score: int) -> str:
    """Generate loan recommendation based on risk and score"""
    if risk_category == "LOW" and credit_score >= 700:
        return "APPROVE: Low risk customer with excellent credit score"
    elif risk_category == "LOW" and credit_score >= 600:
        return "APPROVE: Low risk customer with good credit score"
    elif risk_category == "MEDIUM" and credit_score >= 650:
        return "CONSIDER: Medium risk with conditions (lower limit, higher interest)"
    elif risk_category == "MEDIUM":
        return "REVIEW: Medium risk requires manual review"
    else:
        return "DECLINE: High risk customer"

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict credit risk for a customer
    
    - **customer_id**: Unique customer identifier
    - **features**: Customer transaction features
    
    Returns risk probability, credit score, and recommendation
    """
    try:
        # Check if model is loaded
        if model is None or preprocessor is None:
            load_model()
            if model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Convert features to DataFrame
        features_dict = request.features.dict()
        features_df = pd.DataFrame([features_dict])
        
        # Make prediction
        risk_probability = model.predict_proba(features_df)[0, 1]
        
        # Calculate credit score (300-850)
        credit_score = calculate_credit_score(risk_probability)
        
        # Get risk category
        risk_category = get_risk_category(risk_probability)
        
        # Get recommendation
        recommendation = get_recommendation(risk_category, credit_score)
        
        # Create response
        response = PredictionResponse(
            customer_id=request.customer_id,
            risk_probability=round(risk_probability, 4),
            risk_score=credit_score,
            risk_category=risk_category,
            recommendation=recommendation
        )
        
        print(f"📊 Prediction for {request.customer_id}: "
              f"Risk={risk_probability:.2%}, Score={credit_score}, Category={risk_category}")
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-batch", tags=["Prediction"])
async def predict_batch(requests: List[PredictionRequest]):
    """
    Predict credit risk for multiple customers in batch
    """
    try:
        if model is None or preprocessor is None:
            load_model()
            if model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
        
        responses = []
        features_list = []
        customer_ids = []
        
        # Collect all features and customer IDs
        for request in requests:
            features_dict = request.features.dict()
            features_list.append(features_dict)
            customer_ids.append(request.customer_id)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Make batch predictions
        probabilities = model.predict_proba(features_df)[:, 1]
        
        # Create responses
        for i, (customer_id, prob) in enumerate(zip(customer_ids, probabilities)):
            credit_score = calculate_credit_score(prob)
            risk_category = get_risk_category(prob)
            recommendation = get_recommendation(risk_category, credit_score)
            
            response = PredictionResponse(
                customer_id=customer_id,
                risk_probability=round(prob, 4),
                risk_score=credit_score,
                risk_category=risk_category,
                recommendation=recommendation
            )
            responses.append(response.dict())
        
        return {"predictions": responses, "count": len(responses)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)