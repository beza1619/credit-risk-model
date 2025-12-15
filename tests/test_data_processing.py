"""
Simple unit tests for data processing module
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

def test_imports():
    """Test that we can import the module"""
    print("Testing imports...")
    try:
        from data_processing import DataProcessor
        print("✅ Import test passed")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_data_processor_creation():
    """Test DataProcessor class creation"""
    print("\nTesting DataProcessor creation...")
    try:
        from data_processing import DataProcessor
        processor = DataProcessor()
        assert processor is not None
        print("✅ DataProcessor created successfully")
        return True
    except Exception as e:
        print(f"❌ DataProcessor creation failed: {e}")
        return False

def test_extract_time_features():
    """Test time feature extraction"""
    print("\nTesting time feature extraction...")
    try:
        from data_processing import DataProcessor
        
        # Create test data
        test_data = pd.DataFrame({
            'CustomerId': ['C1', 'C2'],
            'TransactionStartTime': ['2023-01-01 10:30:00', '2023-01-02 14:45:00'],
            'Amount': [100, 200]
        })
        
        processor = DataProcessor()
        result = processor.extract_time_features(test_data)
        
        # Check that new columns were added
        assert 'transaction_hour' in result.columns
        assert 'transaction_day' in result.columns
        print("✅ Time feature extraction passed")
        return True
    except Exception as e:
        print(f"❌ Time feature extraction failed: {e}")
        return False

def run_all_tests():
    """Run all unit tests"""
    print("="*60)
    print("RUNNING UNIT TESTS")
    print("="*60)
    
    tests = [
        test_imports,
        test_data_processor_creation,
        test_extract_time_features
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)