# Credit Risk Model for Alternative Data
## Buy Now Pay Later (BNPL) Credit Scoring System

### 📋 Project Overview
A machine learning system for Bati Bank that predicts customer creditworthiness using e-commerce transaction data. The model enables risk-based decisions for the new BNPL service by transforming behavioral patterns into predictive risk signals.

## Credit Scoring Business Understanding

### 1. How Basel II Accord's Emphasis on Risk Measurement Influences Our Model
The Basel II Capital Accord requires banks to maintain capital reserves proportional to their credit risk exposure. This mandates our model to be:
- **Interpretable**: Regulators must understand prediction logic to validate capital calculations
- **Well-documented**: Every feature transformation needs clear documentation for audit trails  
- **Statistically validated**: Probability of Default (PD) estimates must withstand regulatory scrutiny
- **Stable**: Consistent performance across economic scenarios for accurate capital allocation

### 2. Proxy Variable Necessity and Business Risks
**Why necessary**: No direct "default" labels exist in e-commerce transaction data. We create a proxy using RFM (Recency, Frequency, Monetary) behavioral patterns as the closest indicator of creditworthiness.

**Business risks of proxy-based predictions**:
- **Misalignment Risk**: Transactional disengagement ≠ actual credit default (25.55% outlier rate in Amount shows extreme value dispersion)
- **False Classification**: Correlation analysis shows Amount-Value near-perfect correlation (0.99), but this may not indicate credit risk
- **Validation Gap**: Cannot measure true accuracy without actual default data
- **Regulatory Scrutiny**: Using unvalidated proxies may violate compliance requirements

**Mitigation**: Conservative credit limits, phased deployment, and continuous monitoring of actual loan performance.

### 3. Simple vs Complex Model Trade-offs in Regulated Finance
| Consideration | Logistic Regression (Simple) | Gradient Boosting (Complex) |
|--------------|-------------------|------------------|
| **Interpretability** | High - Clear feature coefficients | Low - Black-box predictions |
| **Regulatory Fit** | Strong - Easily validated & documented | Challenging - Requires extensive justification |
| **Accuracy Potential** | Moderate - May miss non-linear patterns | High - Captures complex relationships |
| **Outlier Handling** | Sensitive to extreme values (25.55% outliers) | Robust to outliers through ensemble methods |
| **Implementation Speed** | Fast deployment, minimal compute | Resource-intensive training |

**Our Decision**: Start with **Logistic Regression** for regulatory compliance and interpretability, despite potential accuracy trade-off. The 0.557 correlation between FraudResult and Amount suggests fraud patterns are detectable with simpler models.

### 📊 Data Analysis & Methodology

#### Dataset Characteristics
- **Scope**: 95,662 transactions from 3,633 customers (Nov 2018 - Feb 2019)
- **Quality**: Complete data - zero missing values across all fields
- **Geography**: 100% Uganda-based transactions (Currency: UGX)

#### Key Behavioral Insights
1. **Highly Skewed Engagement**: 50% of customers have ≤4 transactions; top 5% account for 48% of total activity
2. **Fraud Patterns**: Rare (0.20%) but severe - average fraud transaction ($1.53M) vs normal ($3,627)
3. **Temporal Trends**: Peak activity at 7-9 AM; December highest volume (35,635 transactions)

#### Correlation Analysis Findings
- **Amount-Value Correlation**: 0.99 (near-perfect, as Value is absolute Amount)
- **Fraud-Amount Correlation**: 0.557 (fraud transactions tend to be larger)
- **CountryCode**: No correlations (all values = 256 for Uganda)
- **Insight**: Monetary features strongly indicate fraudulent activity

#### Outlier Detection Results
- **Amount**: 25.55% outliers by IQR method, range up to $9.88M
- **Value**: 9.43% outliers, extreme values indicate fraud cases
- **PricingStrategy**: 16.53% outliers, values clustered (0, 2, 4)
- **Business Implication**: Extreme transactions require separate risk assessment

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
├── data/ # Data storage (.gitignored)
│ ├── raw/ # Original dataset
│ └── processed/ # Processed data for training
├── notebooks/ # Exploratory analysis
│ └── eda.ipynb # Task 2: Complete EDA with visualizations
├── src/ # Production code
│ ├── data_processing.py # Task 3: Feature engineering
│ ├── target_engineering.py # Task 4: Risk labeling
│ ├── train.py # Task 5: Model training
│ └── api/ # Task 6: Deployment
├── tests/ # Unit tests
├── models/ # Trained models
├── .github/workflows/ # CI/CD pipeline (ci.yml)
├── Dockerfile # Container configuration
├── docker-compose.yml # Service orchestration
├── requirements.txt # Python dependencies
├── .gitignore # Excludes data/, caches, envs
└── README.md # Project documentation

### 🚀 Getting Started

#### Prerequisites
- Python 3.8+
- Docker (for containerized deployment)

#### Installation
```bash
# Clone repository
git clone https://github.com/beza1619/credit-risk-model.git
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