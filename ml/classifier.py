"""
ECG Classifier Module
Provides prediction functionality for trained ECG arrhythmia classifier
"""
import joblib
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from ml.ecg_processing import ECGProcessor


class ECGClassifier:
    """ECG Arrhythmia Classifier for real-time predictions"""
    
    def __init__(self, model_path: str = "ml/models"):
        """
        Initialize classifier with pre-trained model
        
        Args:
            model_path: Path to model files directory
        """
        self.model_path = Path(model_path)
        self.processor = ECGProcessor(sampling_rate=200)
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.label_map = None
        self.reverse_label_map = None
        
        self._load_model()
    
    def _load_model(self):
        """Load trained model, scaler, and metadata"""
        try:
            # Load model
            model_file = self.model_path / "ecg_classifier.pkl"
            self.model = joblib.load(model_file)
            
            # Load scaler
            scaler_file = self.model_path / "scaler.pkl"
            self.scaler = joblib.load(scaler_file)
            
            # Load metadata
            metadata_file = self.model_path / "metadata.json"
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            self.feature_names = metadata['feature_names']
            self.label_map = metadata['label_map']
            self.reverse_label_map = metadata['reverse_label_map']
            
            print(f"✓ ECG Classifier loaded successfully from {self.model_path}")
            
        except Exception as e:
            print(f"⚠ Warning: Could not load ECG classifier: {e}")
            print("  Model needs to be trained first. Run: python ml/train_classifier.py")
    
    def predict(self, ecg_data: List[List[str]]) -> Dict:
        """
        Predict arrhythmia type from ECG data
        
        Args:
            ecg_data: List of [value1, value2] string pairs
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            return {
                'success': False,
                'error': 'Model not loaded. Please train the model first.',
                'predicted_class': None,
                'confidence': 0.0,
                'probabilities': {}
            }
        
        try:
            # Extract features
            features = self.processor.extract_all_features(ecg_data)
            
            # Convert to array in correct order
            feature_array = np.array([features[name] for name in self.feature_names]).reshape(1, -1)
            
            # Scale features
            feature_scaled = self.scaler.transform(feature_array)
            
            # Predict
            prediction = self.model.predict(feature_scaled)[0]
            probabilities = self.model.predict_proba(feature_scaled)[0]
            
            # Get class probabilities
            classes = self.model.classes_
            prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
            
            # Get confidence (max probability)
            confidence = float(max(probabilities))
            
            return {
                'success': True,
                'predicted_class': prediction,
                'confidence': confidence,
                'probabilities': prob_dict,
                'features_extracted': len(features),
                'num_data_points': len(ecg_data)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'predicted_class': None,
                'confidence': 0.0,
                'probabilities': {}
            }
    
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self.model is not None


# Global classifier instance
_classifier_instance = None


def get_classifier() -> ECGClassifier:
    """
    Get or create global classifier instance (singleton pattern)
    
    Returns:
        ECGClassifier instance
    """
    global _classifier_instance
    
    if _classifier_instance is None:
        _classifier_instance = ECGClassifier()
    
    return _classifier_instance
