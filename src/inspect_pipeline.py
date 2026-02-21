import joblib
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "car_price_pipeline.pkl")

# Import required classes for deserialization
from train_inference_pipeline import CategoricalGrouper, feature_engineering

try:
    pipeline = joblib.load(MODEL_PATH)
    print(f"Pipeline steps: {pipeline.named_steps.keys()}")
    
    preprocessor = pipeline.named_steps['preprocessor']
    print(f"Preprocessor type: {type(preprocessor)}")
    
    if hasattr(preprocessor, 'transformers_'):
        print("Transformers in preprocessor:")
        for name, transformer, columns in preprocessor.transformers_:
            print(f" - Name: {name}, Columns: {columns}")
            
    if hasattr(preprocessor, 'named_transformers_'):
        print(f"Named transformers keys: {preprocessor.named_transformers_.keys()}")
        
except Exception as e:
    print(f"Error inspecting pipeline: {e}")
