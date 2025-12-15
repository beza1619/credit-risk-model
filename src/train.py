"""
Model Training Script for Credit Risk Model
Task 5: Model Training and Tracking
STEP 1: Basic Imports and Setup
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

print("=" * 70)
print("CREDIT RISK MODEL TRAINING")
print("=" * 70)
print("✅ Basic imports completed")
# STEP 2: Add ML and sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Model imports
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# MLflow imports
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

print("✅ ML and sklearn imports added")


# STEP 3: Add data loading function
def load_and_prepare_data(data_path):
    """
    Load and prepare data for modeling
    """
    print(f"\n📊 Loading data from {data_path}...")

    # Load data
    df = pd.read_csv(data_path)
    print(f"   Data shape: {df.shape}")

    # Separate features and target
    # Drop columns that shouldn't be used as features
    columns_to_drop = ["customer_id", "cluster"]  # ID and cluster label
    features = df.drop(columns=columns_to_drop + ["is_high_risk"], errors="ignore")
    target = df["is_high_risk"]

    print(f"   Features: {features.shape[1]} columns")
    print(f"   Target distribution:")
    print(
        f"      High Risk (1): {(target == 1).sum()} ({(target == 1).mean()*100:.1f}%)"
    )
    print(
        f"      Low Risk (0): {(target == 0).sum()} ({(target == 0).mean()*100:.1f}%)"
    )

    return features, target


print("✅ Data loading function added")


# STEP 4: Add data splitting function
def split_data(features, target, test_size=0.2, random_state=42):
    """
    Split data into train and test sets
    """
    print(f"\n🔀 Splitting data (test_size={test_size})...")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,  # Maintain class distribution
    )

    print(f"   Train set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    print(f"   Train target distribution: {y_train.mean():.3f} high-risk")
    print(f"   Test target distribution: {y_test.mean():.3f} high-risk")

    return X_train, X_test, y_train, y_test


print("✅ Data splitting function added")


# STEP 5: Add preprocessor function
def create_preprocessor():
    """
    Create preprocessing pipeline
    """
    print("\n🔧 Creating preprocessing pipeline...")

    # All our features are numeric
    numeric_features = [
        "transaction_count",
        "total_amount",
        "avg_amount",
        "std_amount",
        "min_amount",
        "max_amount",
        "unique_transactions",
        "recency",
        "frequency",
        "monetary",
        "avg_transaction_value",
        "transaction_std",
        "provider_diversity",
        "product_diversity",
        "channel_diversity",
        "amount_range",
        "monetary_per_day",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)]
    )

    print(f"✅ Preprocessor created with {len(numeric_features)} numeric features")
    return preprocessor


print("✅ Preprocessor function added")


# STEP 6: Add Logistic Regression training function
def train_logistic_regression(X_train, y_train, preprocessor):
    """
    Train Logistic Regression model
    """
    print("\n" + "=" * 50)
    print("TRAINING LOGISTIC REGRESSION")
    print("=" * 50)

    # Create pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
        ]
    )

    # Define hyperparameters for tuning
    param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__penalty": ["l2"],
        "classifier__solver": ["liblinear", "lbfgs"],
    }

    # Perform grid search
    print("🔍 Performing hyperparameter tuning with GridSearchCV...")
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\n✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best CV score (ROC-AUC): {grid_search.best_score_:.4f}")

    # Get the best model
    best_model = grid_search.best_estimator_

    return best_model, grid_search


print("✅ Logistic Regression function added")


# STEP 7: Add evaluation functions
def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance
    """
    print(f"\n" + "=" * 50)
    print(f"EVALUATING {model_name.upper()}")
    print("=" * 50)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"📊 Performance Metrics:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   ROC-AUC:   {roc_auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📈 Confusion Matrix:")
    print(f"   True Negatives:  {cm[0, 0]}")
    print(f"   False Positives: {cm[0, 1]}")
    print(f"   False Negatives: {cm[1, 0]}")
    print(f"   True Positives:  {cm[1, 1]}")

    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))

    # Plot ROC curve
    plot_roc_curve(y_test, y_pred_proba, model_name)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
    }


def plot_roc_curve(y_true, y_pred_proba, model_name):
    """
    Plot ROC curve
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"ROC curve (AUC = {roc_auc_score(y_true, y_pred_proba):.4f})",
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()


print("✅ Evaluation functions added")


# STEP 8: Add Random Forest training function (FAST VERSION FOR TESTING)
def train_random_forest(X_train, y_train, preprocessor):
    """
    Train Random Forest model
    """
    print("\n" + "=" * 50)
    print("TRAINING RANDOM FOREST")
    print("=" * 50)

    # Create pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    # Define SIMPLIFIED hyperparameters for faster testing
    param_grid = {
        "classifier__n_estimators": [50, 100],  # Reduced
        "classifier__max_depth": [10, 20],  # Reduced
        "classifier__min_samples_split": [2, 5],  # Reduced
    }

    # Perform grid search with fewer folds
    print("🔍 Performing hyperparameter tuning (simplified for testing)...")
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\n✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best CV score (ROC-AUC): {grid_search.best_score_:.4f}")

    # Get the best model
    best_model = grid_search.best_estimator_

    return best_model, grid_search


print("✅ Random Forest function added (fast version)")


# STEP 9: Add MLflow tracking functions
def setup_mlflow():
    """
    Setup MLflow tracking
    """
    print("\n" + "=" * 50)
    print("SETTING UP MLFLOW TRACKING")
    print("=" * 50)

    # Set tracking URI (local by default)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Create experiment
    experiment_name = "Credit_Risk_Modeling"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    mlflow.set_experiment(experiment_name)

    print(f"✅ MLflow tracking setup complete")
    print(f"   Experiment: {experiment_name}")
    print(f"   Tracking URI: {mlflow.get_tracking_uri()}")

    return experiment_id


def log_model_to_mlflow(model, model_name, params, metrics, X_train, y_train):
    """
    Log model and metrics to MLflow
    """
    print(f"\n📊 Logging {model_name} to MLflow...")

    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(model, model_name)

        # Log signature
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model, artifact_path=model_name, signature=signature
        )

        print(f"✅ {model_name} logged to MLflow")
        print(f"   Run ID: {mlflow.active_run().info.run_id}")


print("✅ MLflow tracking functions added")


# STEP 10: Add main training function (FAST VERSION)
def main(fast_mode=True):
    """
    Main training function
    """
    print("=" * 70)
    print("CREDIT RISK MODEL TRAINING PIPELINE")
    if fast_mode:
        print("(FAST MODE FOR TESTING)")
    print("=" * 70)

    # 1. Load data
    data_path = "../data/processed/training_data.csv"
    features, target = load_and_prepare_data(data_path)

    # 2. Split data
    X_train, X_test, y_train, y_test = split_data(features, target)

    # 3. Create preprocessor
    preprocessor = create_preprocessor()

    models_trained = {}

    if fast_mode:
        # FAST MODE: Train simple models without grid search
        print("\n⚠️ FAST MODE: Training simplified models...")

        # Simple Logistic Regression
        print("\n" + "=" * 50)
        print("TRAINING SIMPLE LOGISTIC REGRESSION")
        print("=" * 50)
        from sklearn.linear_model import LogisticRegression

        lr_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(C=1, random_state=42, max_iter=1000)),
            ]
        )
        lr_pipeline.fit(X_train, y_train)
        lr_metrics = evaluate_model(
            lr_pipeline, X_test, y_test, "Logistic Regression (Simple)"
        )
        models_trained["Logistic Regression"] = (lr_pipeline, lr_metrics)

        # Simple Random Forest
        print("\n" + "=" * 50)
        print("TRAINING SIMPLE RANDOM FOREST")
        print("=" * 50)
        from sklearn.ensemble import RandomForestClassifier

        rf_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(n_estimators=50, random_state=42),
                ),
            ]
        )
        rf_pipeline.fit(X_train, y_train)
        rf_metrics = evaluate_model(
            rf_pipeline, X_test, y_test, "Random Forest (Simple)"
        )
        models_trained["Random Forest"] = (rf_pipeline, rf_metrics)

    else:
        # FULL MODE: Train with grid search
        print("\n" + "=" * 70)
        print("PHASE 1: LOGISTIC REGRESSION (WITH GRID SEARCH)")
        print("=" * 70)
        lr_model, lr_grid = train_logistic_regression(X_train, y_train, preprocessor)
        lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
        models_trained["Logistic Regression"] = (lr_model, lr_metrics)

        print("\n" + "=" * 70)
        print("PHASE 2: RANDOM FOREST (WITH GRID SEARCH)")
        print("=" * 70)
        rf_model, rf_grid = train_random_forest(X_train, y_train, preprocessor)
        rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
        models_trained["Random Forest"] = (rf_model, rf_metrics)

    # Compare models
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    import pandas as pd

    comparison_data = {}
    for model_name, (model, metrics) in models_trained.items():
        comparison_data[model_name] = [
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["roc_auc"],
        ]

    comparison = pd.DataFrame(
        comparison_data,
        index=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    )

    print("\n📊 Performance Comparison:")
    display(comparison.round(4))

    # Determine best model
    best_model_name = max(
        models_trained.keys(), key=lambda x: models_trained[x][1]["roc_auc"]
    )
    best_model, best_metrics = models_trained[best_model_name]

    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print(f"   Accuracy: {best_metrics['accuracy']:.4f}")

    # Save best model
    print("\n" + "=" * 70)
    print("SAVING BEST MODEL")
    print("=" * 70)

    import joblib
    import os

    # Create models directory if it doesn't exist
    os.makedirs("../models", exist_ok=True)

    model_path = "../models/best_credit_risk_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"✅ Best model saved to: {model_path}")

    # Also save the preprocessor separately
    preprocessor_path = "../models/preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    print(f"✅ Preprocessor saved to: {preprocessor_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    return best_model, best_model_name, comparison


print("✅ Main training function added (fast version)")

# Run if executed directly
if __name__ == "__main__":
    main(fast_mode=True)  # Set to False for full training
