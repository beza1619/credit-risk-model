# Credit Risk Probability Model for Alternative Data

## Project Overview
This project develops a credit risk scoring model for Bati Bank's buy-now-pay-later service using e-commerce transaction data. The model transforms behavioral data into predictive risk signals to assess customer creditworthiness.

## Credit Scoring Business Understanding

### 1. Basel II Accord and Model Interpretability
The Basel II Capital Accord emphasizes rigorous risk measurement and adequate capital allocation based on credit risk exposure. This influences our model development in three key ways:

- **Regulatory Compliance**: Banks must demonstrate that their risk models are robust, validated, and conceptually sound. Our model needs clear documentation of methodology and assumptions.

- **Capital Requirements**: Accurate risk probability estimates directly impact the amount of capital Bati Bank must hold. Underestimation could lead to insufficient capital buffers.

- **Explainability**: Regulators require models to be interpretable for validation purposes. We must be able to explain why a customer receives a particular risk score, which is crucial for compliance and customer communication.

### 2. Proxy Variable Necessity and Risks
Since we lack direct loan performance data, creating a proxy variable is essential:

**Why necessary:**
- No historical loan default data exists for e-commerce customers
- RFM (Recency, Frequency, Monetary) patterns serve as behavioral proxies for creditworthiness
- Disengaged customers (low frequency, low spending) may correlate with higher default risk

**Potential business risks:**
- **Misclassification Risk**: Customers labeled as high-risk based on shopping behavior may actually be creditworthy
- **Opportunity Cost**: Overly conservative models could reject profitable customers
- **Discrimination Risk**: Behavioral proxies might unintentionally exclude certain customer segments
- **Validation Challenges**: Proxy-based models are harder to validate against actual loan performance

### 3. Model Selection Trade-offs in Financial Context

**Simple, Interpretable Models (Logistic Regression with WoE):**
- **Advantages**: 
  - Easily explainable to regulators and business stakeholders
  - Linear relationships make risk factor contributions transparent
  - Well-established in credit scoring with proven regulatory acceptance
  - Lower computational requirements

- **Disadvantages**:
  - May capture fewer complex, non-linear patterns
  - Lower predictive power if relationships are non-linear
  - Requires careful feature engineering and transformation

**Complex, High-Performance Models (Gradient Boosting):**
- **Advantages**:
  - Higher predictive accuracy through complex pattern recognition
  - Automatic feature interaction detection
  - Better handling of non-linear relationships

- **Disadvantages**:
  - "Black box" nature challenges regulatory compliance
  - Harder to explain individual predictions
  - Potential overfitting without careful regularization
  - Higher computational costs and maintenance complexity

**Recommended Approach**: Start with interpretable models for regulatory acceptance, then explore ensemble methods if interpretability requirements allow and performance gains justify the complexity. Consider using SHAP values for model explainability with complex models.

## Project Structure
[Project structure as defined in requirements]

## Setup Instructions
1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Download data from Kaggle to `data/raw/`
4. Run EDA: `jupyter notebook notebooks/eda.ipynb`
5. Train model: `python src/train.py`
