# Credit Risk Model for Alternative Data
## Buy Now Pay Later (BNPL) Credit Scoring System

### 📋 Project Overview
A machine learning system for Bati Bank that predicts customer creditworthiness using e-commerce transaction data. The model enables risk-based decisions for the new BNPL service by transforming behavioral patterns into predictive risk signals.

### 🎯 Business Context & Regulatory Compliance

#### 1. Basel II Requirements & Model Transparency
The Basel II Capital Accord mandates that financial institutions maintain capital reserves proportional to quantified credit risk. This requires:
- **Transparent risk quantification** with explainable probability estimates
- **Full model documentation** for regulatory validation and audit trails
- **Stable performance** across economic conditions for capital calculation

#### 2. Proxy Variable Justification
**Challenge**: No direct default labels in e-commerce transaction data.

**Solution**: RFM (Recency, Frequency, Monetary) behavioral clustering creates a proxy for credit risk based on customer engagement patterns.

**Business Risk**: Transactional disengagement may not perfectly correlate with credit default, potentially causing:
- **Type I Errors**: Rejecting creditworthy customers (lost revenue)
- **Type II Errors**: Approving high-risk customers (credit losses)
- **Proxy Misalignment**: Behavioral patterns ≠ financial reliability

**Mitigation**: Conservative risk thresholds, phased deployment with real-default tracking, and continuous model recalibration.

#### 3. Model Selection: Regulatory vs Performance Trade-offs
| Consideration | Logistic Regression | Gradient Boosting |
|--------------|-------------------|------------------|
| **Interpretability** | High - Clear feature coefficients | Low - Black-box predictions |
| **Regulatory Fit** | Strong - Easily validated | Challenging - Extensive documentation needed |
| **Predictive Power** | Moderate - Linear assumptions | High - Captures complex patterns |
| **Implementation** | Fast deployment, minimal compute | Slower, resource-intensive |

**Decision**: Deploy Logistic Regression initially for regulatory compliance, with Gradient Boosting as secondary validation.

### 📊 Data Analysis & Methodology

#### Dataset Characteristics
- **Scope**: 95,662 transactions from 3,633 customers (Nov 2018 - Feb 2019)
- **Quality**: Complete data - zero missing values across all fields
- **Geography**: 100% Uganda-based transactions (Currency: UGX)

#### Key Behavioral Insights
1. **Highly Skewed Engagement**: 50% of customers have ≤4 transactions; top 5% account for 48% of total activity
2. **Fraud Patterns**: Rare (0.20%) but severe - average fraud transaction ($1.53M) vs normal ($3,627)
3. **Temporal Trends**: Peak activity at 7-9 AM; December highest volume (35,635 transactions)

#### Feature Engineering Pipeline
**RFM Metrics**:
- **Recency**: Days since last transaction (snapshot: 2019-02-28)
- **Frequency**: Transaction count per customer
- **Monetary**: Spending patterns (total, average, variability)

**Additional Signals**:
- Transaction timing (hour, day, seasonality)
- Customer tenure and engagement consistency
- Product category preferences

#### Risk Segmentation Strategy
**Method**: K-Means clustering (k=3) on normalized RFM features

**Risk Logic**:
- **High-Risk**: High recency + low frequency + low monetary value
- **Medium-Risk**: Moderate engagement across all dimensions
- **Low-Risk**: Recent activity + high frequency + substantial spending

### 🏗️ Technical Implementation

#### Model Development
**Algorithms**:
1. Logistic Regression (primary - regulatory compliance)
2. Random Forest (comparison - predictive power)

**Evaluation Framework**:
- **Primary Metric**: ROC-AUC (discrimination ability)
- **Secondary Metrics**: Precision, Recall, F1-Score
- **Business Metrics**: False positive rate, risk coverage

#### MLOps Architecture
Data → Feature Engineering → Model Training → API Service
↓ ↓ ↓ ↓
Validation → MLflow Tracking → Registry → CI/CD Pipeline

**Components**:
- **FastAPI**: REST API for real-time risk scoring
- **MLflow**: Experiment tracking and model versioning
- **Docker**: Containerized deployment
- **GitHub Actions**: Automated testing and validation

### 📁 Project Structure
credit-risk-model/
├── data/ # Data storage
├── notebooks/ # Exploratory analysis
├── src/ # Production code
│ ├── data_processing.py # Feature engineering
│ ├── target_engineering.py # Risk labeling
│ ├── train.py # Model training
│ └── api/ # Deployment API
├── tests/ # Unit tests
├── models/ # Trained models
└── .github/workflows/ # CI/CD automation

### 🚀 Getting Started

#### Prerequisites
- Python 3.8+
- Docker (for containerized deployment)

#### Installation
```bash
# Clone repository
git clone <repository-url>
cd credit-risk-model

# Install dependencies
pip install -r requirements.txt

# Execute pipeline
python src/data_processing.py      # Feature engineering
python src/target_engineering.py   # Risk labeling
python src/train.py                # Model training
API Usage
# Start service
uvicorn src.api.main:app --reload

# Request risk prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"recency": 30, "frequency": 5, "monetary_total": 250000}'
  ⚠️ Limitations & Future Considerations
Current Constraints
Proxy Validation: RFM-based risk labels require validation against actual default data

Temporal Scope: Limited to 3 months of transaction history

Geographic Specificity: Uganda-only data may limit generalizability

Enhancement Roadmap
Data Enrichment: Integrate browsing behavior, device signals, demographic indicators

Model Sophistication: Sequence-based models for temporal pattern recognition

Fairness Assurance: Systematic bias detection across customer segments
Production Monitoring: Real-time performance tracking and drift detection

📚 References
Basel II Capital Accord - Risk-weighted capital requirements

HKMA Alternative Credit Scoring Guidelines

World Bank Credit Scoring Approaches for Emerging Markets