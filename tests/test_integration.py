"""
Integration test for the complete pipeline
"""

import pandas as pd
import numpy as np
import sys
import os
import joblib

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))


def test_model_loading():
    """Test that trained model can be loaded"""
    print("Testing model loading...")
    try:
        model_path = "../models/best_credit_risk_model.pkl"
        preprocessor_path = "../models/preprocessor.pkl"

        if os.path.exists(model_path) and os.path.exists(preprocessor_path):
            model = joblib.load(model_path)
            preprocessor = joblib.load(preprocessor_path)

            print(f"✅ Model loaded: {type(model).__name__}")
            print(f"✅ Preprocessor loaded: {type(preprocessor).__name__}")
            return True
        else:
            print("⚠️ Model files not found")
            return False
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def test_prediction():
    """Test making a prediction"""
    print("\nTesting prediction...")
    try:
        from data_processing import DataProcessor
        import joblib

        # Load model
        model = joblib.load("../models/best_credit_risk_model.pkl")

        # Create test data
        test_data = pd.DataFrame(
            {
                "transaction_count": [25.56],
                "total_amount": [171737.7],
                "avg_amount": [15715.62],
                "std_amount": [13605.17],
                "min_amount": [3863.51],
                "max_amount": [50838.73],
                "unique_transactions": [25.56],
                "recency": [31.46],
                "frequency": [25.56],
                "monetary": [171737.7],
                "avg_transaction_value": [15715.62],
                "transaction_std": [13605.17],
                "provider_diversity": [2.56],
                "product_diversity": [2.11],
                "channel_diversity": [1.76],
                "amount_range": [46975.22],
                "monetary_per_day": [34085.12],
            }
        )

        # Make prediction
        prediction = model.predict(test_data)
        probability = model.predict_proba(test_data)

        print(f"✅ Prediction made successfully")
        print(f"   Class prediction: {prediction[0]}")
        print(f"   Probability shape: {probability.shape}")
        return True
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False


def run_integration_tests():
    """Run integration tests"""
    print("=" * 60)
    print("RUNNING INTEGRATION TESTS")
    print("=" * 60)

    tests = [test_model_loading, test_prediction]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"INTEGRATION TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
