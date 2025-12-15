"""
Pydantic models for API request/response validation
Task 6: Model Deployment
"""

from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np

class CustomerFeatures(BaseModel):
    """Input features for a customer"""
    transaction_count: float = Field(..., description="Number of transactions")
    total_amount: float = Field(..., description="Total transaction amount")
    avg_amount: float = Field(..., description="Average transaction amount")
    std_amount: float = Field(..., description="Standard deviation of transaction amounts")
    min_amount: float = Field(..., description="Minimum transaction amount")
    max_amount: float = Field(..., description="Maximum transaction amount")
    unique_transactions: float = Field(..., description="Number of unique transactions")
    recency: float = Field(..., description="Days since last transaction")
    frequency: float = Field(..., description="Transaction frequency")
    monetary: float = Field(..., description="Total monetary value")
    avg_transaction_value: float = Field(..., description="Average transaction value")
    transaction_std: float = Field(..., description="Transaction standard deviation")
    provider_diversity: float = Field(..., description="Number of unique providers")
    product_diversity: float = Field(..., description="Number of product categories")
    channel_diversity: float = Field(..., description="Number of transaction channels")
    amount_range: float = Field(..., description="Range of transaction amounts")
    monetary_per_day: float = Field(..., description="Monetary value per day")
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_count": 25.56,
                "total_amount": 171737.7,
                "avg_amount": 15715.62,
                "std_amount": 13605.17,
                "min_amount": 3863.51,
                "max_amount": 50838.73,
                "unique_transactions": 25.56,
                "recency": 31.46,
                "frequency": 25.56,
                "monetary": 171737.7,
                "avg_transaction_value": 15715.62,
                "transaction_std": 13605.17,
                "provider_diversity": 2.56,
                "product_diversity": 2.11,
                "channel_diversity": 1.76,
                "amount_range": 46975.22,
                "monetary_per_day": 34085.12
            }
        }

class PredictionRequest(BaseModel):
    """Request model for prediction endpoint"""
    customer_id: str = Field(..., description="Customer identifier")
    features: CustomerFeatures
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST_12345",
                "features": {
                    "transaction_count": 25.56,
                    "total_amount": 171737.7,
                    "avg_amount": 15715.62,
                    "std_amount": 13605.17,
                    "min_amount": 3863.51,
                    "max_amount": 50838.73,
                    "unique_transactions": 25.56,
                    "recency": 31.46,
                    "frequency": 25.56,
                    "monetary": 171737.7,
                    "avg_transaction_value": 15715.62,
                    "transaction_std": 13605.17,
                    "provider_diversity": 2.56,
                    "product_diversity": 2.11,
                    "channel_diversity": 1.76,
                    "amount_range": 46975.22,
                    "monetary_per_day": 34085.12
                }
            }
        }

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    customer_id: str
    risk_probability: float = Field(..., ge=0, le=1, description="Probability of being high risk (0-1)")
    risk_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    risk_category: str = Field(..., description="Risk category: LOW, MEDIUM, or HIGH")
    recommendation: str = Field(..., description="Loan recommendation")
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST_12345",
                "risk_probability": 0.15,
                "risk_score": 750,
                "risk_category": "LOW",
                "recommendation": "APPROVE: Low risk customer with good credit history"
            }
        }

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    message: str