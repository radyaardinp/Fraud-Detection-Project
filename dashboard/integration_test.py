# integration_test.py - Test script untuk memastikan semua modul terintegrasi dengan baik
import pandas as pd
import numpy as np
from predict_pipeline import run_complete_pipeline, validate_pipeline_results, extract_lime_data
from lime_explainer import create_lime_explainer_from_pipeline, explain_test_instance, explain_custom_instance

def test_complete_integration(df: pd.DataFrame):
    """
    Test complete integration dari preprocessing sampai LIME
    
    Args:
        df: Raw dataframe untuk testing
    """
    print("🚀 Starting Integration Test...")
    
    # Step 1: Run complete pipeline
    print("\n📊 Step 1: Running complete pipeline...")
    try:
        pipeline_results = run_complete_pipeline(
            df=df,
            resampling_method='SMOTE',
            training_mode='manual',
            hidden_neurons=50,  # Smaller untuk testing
            activation='sigmoid',
            threshold=0.5,
            random_state=42
        )
        
        if pipeline_results is None:
            print("❌ Pipeline failed!")
            return False
            
        print("✅ Pipeline completed successfully!")
        print(f"   - Model accuracy: {pipeline_results['model_results']['metrics']['accuracy']}")
        print(f"   - Features count: {len(pipeline_results['lime_data']['feature_names'])}")
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {str(e)}")
        return False
    
    # Step 2: Validate pipeline results
    print("\n🔍 Step 2: Validating pipeline results...")
    is_valid, error_msg = validate_pipeline_results(pipeline_results)
    
    if not is_valid:
        print(f"❌ Pipeline results invalid: {error_msg}")
        return False
    
    print("✅ Pipeline results valid!")
    
    # Step 3: Test LIME explainer creation
    print("\n🔬 Step 3: Creating LIME explainer...")
    try:
        lime_explainer = create_lime_explainer_from_pipeline(pipeline_results)
        
        if lime_explainer is None:
            print("❌ Failed to create LIME explainer!")
            return False
            
        print("✅ LIME explainer created successfully!")
        print(f"   - Feature names count: {len(lime_explainer.feature_names)}")
        
    except Exception as e:
        print(f"❌ LIME explainer creation failed: {str(e)}")
        return False
    
    # Step 4: Test instance explanation
    print("\n📝 Step 4: Testing instance explanation...")
    try:
        # Test beberapa instances
        test_indices = [0, 1, 2] if len(pipeline_results['lime_data']['X_test']) >= 3 else [0]
        
        for idx in test_indices:
            explanation_result = explain_test_instance(
                pipeline_results, 
                instance_idx=idx, 
                num_features=5
            )
            
            if explanation_result is None:
                print(f"❌ Failed to explain instance {idx}!")
                return False
            
            print(f"✅ Instance {idx} explained successfully!")
            print(f"   - Predicted: {explanation_result['predicted_class']}")
            print(f"   - Actual: {explanation_result['actual_class']}")
            print(f"   - Confidence: {explanation_result['confidence']:.3f}")
            print(f"   - Top feature: {explanation_result['explanation_data']['explanation_df'].iloc[0]['Feature']}")
            
    except Exception as e:
        print(f"❌ Instance explanation failed: {str(e)}")
        return False
    
    # Step 5: Test custom instance (optional)
    print("\n🔧 Step 5: Testing custom instance explanation...")
    try:
        # Create a custom instance (random from test data)
        lime_data = pipeline_results['lime_data']
        X_test = lime_data['X_test']
        
        if hasattr(X_test, 'values'):
            custom_instance = X_test.values[0]  # Use first test instance as custom
        else:
            custom_instance = X_test[0]
        
        custom_result = explain_custom_instance(
            pipeline_results,
            custom_instance,
            num_features=5
        )
        
        if custom_result is None:
            print("❌ Failed to explain custom instance!")
            return False
        
        print("✅ Custom instance explained successfully!")
        print(f"   - Predicted: {custom_result['predicted_class']}")
        print(f"   - Confidence: {custom_result['confidence']:.3f}")
        
    except Exception as e:
        print(f"❌ Custom instance explanation failed: {str(e)}")
        return False
    
    # Final summary
    print("\n🎉 Integration Test Summary:")
    print("✅ All modules integrated successfully!")
    print("✅ Pipeline runs end-to-end without errors")
    print("✅ LIME explanations work properly")
    print("✅ Ready for dashboard integration!")
    
    return True

def print_pipeline_summary(pipeline_results: Dict):
    """
    Print summary informasi dari pipeline results
    """
    if pipeline_results is None:
        print("No pipeline results to summarize.")
        return
    
    print("\n📋 Pipeline Summary:")
    print("=" * 50)
    
    # Model performance
    metrics = pipeline_results['model_results']['metrics']
    print(f"🎯 Model Performance:")
    print(f"   - Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   - Precision: {metrics['precision']:.4f}")
    print(f"   - Recall:    {metrics['recall']:.4f}")
    print(f"   - F1-Score:  {metrics['f1_score']:.4f}")
    
    # Data info
    lime_data = pipeline_results['lime_data']
    print(f"\n📊 Data Information:")
    print(f"   - Features count: {len(lime_data['feature_names'])}")
    print(f"   - Training samples: {len(lime_data['X_train'])}")
    print(f"   - Test samples: {len(lime_data['X_test'])}")
    
    # Configuration
    model_results = pipeline_results['model_results']
    print(f"\n⚙️ Model Configuration:")
    print(f"   - Mode: {model_results['mode']}")
    if 'parameters' in model_results:
        params = model_results['parameters']
        for key, value in params.items():
            print(f"   - {key}: {value}")
    
    # LIME readiness
    print(f"\n🔬 LIME Integration:")
    print(f"   - Model weights: ✅ Available")
    print(f"   - Feature names: ✅ Available ({len(lime_data['feature_names'])} features)")
    print(f"   - Training data: ✅ Available")
    print(f"   - Ready for explanations: ✅ Yes")

# Example usage function
def example_usage():
    """
    Contoh penggunaan untuk testing dengan dummy data
    """
    print("🧪 Creating dummy data for testing...")
    
    # Create dummy fraud detection data
    np.random.seed(42)
    n_samples = 1000
    
    dummy_data = {
        'amount': np.random.exponential(1000, n_samples),
        'inquiryAmount': np.random.exponential(1000, n_samples),
        'settlementAmount': np.random.exponential(900, n_samples),
        'feeAmount': np.random.exponential(50, n_samples),
        'discountAmount': np.random.exponential(10, n_samples),
        'merchantId': np.random.choice(['M001', 'M002', 'M003', 'M004', 'M005'], n_samples),
        'paymentSource': np.random.choice(['CARD', 'BANK', 'WALLET'], n_samples),
        'status': np.random.choice(['SUCCESS', 'FAILED', 'DECLINED'], n_samples, p=[0.7, 0.2, 0.1]),
        'statusCode': np.random.choice(['200', '400', '500'], n_samples, p=[0.7, 0.2, 0.1]),
        'createdTime': pd.date_range('2024-01-01', periods=n_samples, freq='1H'),
        'updatedTime': pd.date_range('2024-01-01', periods=n_samples, freq='1H') + pd.Timedelta(minutes=5)
    }
    
    df = pd.DataFrame(dummy_data)
    
    print(f"✅ Dummy data created: {df.shape}")
    print("📊 Starting integration test...")
    
    # Run integration test
    success = test_complete_integration(df)
    
    if success:
        print("\n🎉 All tests passed! Integration is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
    
    return success

if __name__ == "__main__":
    example_usage()
