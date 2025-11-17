"""
ECG Classification Model Training Script
Trains a Random Forest classifier to detect AFIB, VTACH, or PAUSE from ECG signals
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import sys

# Add parent directory to path to import ecg_processing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.ecg_processing import ECGProcessor


class ECGClassifierTrainer:
    """Train and evaluate ECG arrhythmia classifier"""
    
    def __init__(self, data_path: str = "test-trifetch"):
        """
        Initialize trainer
        
        Args:
            data_path: Path to test-trifetch folder
        """
        self.data_path = Path(data_path)
        self.processor = ECGProcessor(sampling_rate=200)
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        # Label mapping
        self.label_map = {
            'AF_Approved': 'AFIB',
            'PAUSE_Approved': 'PAUSE',
            'VTACH_Approved': 'VTACH'
        }
        
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
    
    def read_ecg_files(self, patient_folder: Path) -> List[List[str]]:
        """
        Read all ECG data files for a patient
        
        Args:
            patient_folder: Path to patient folder
            
        Returns:
            Combined ECG data from all files
        """
        ecg_data = []
        
        # Find all ECGData files
        ecg_files = sorted(patient_folder.glob("ECGData_*.txt"))
        
        for ecg_file in ecg_files:
            try:
                with open(ecg_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and ',' in line:
                            values = line.split(',')
                            if len(values) == 2:
                                ecg_data.append(values)
            except Exception as e:
                print(f"Error reading {ecg_file}: {e}")
        
        return ecg_data
    
    def load_training_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load and extract features from all training samples
        
        Returns:
            Tuple of (features_df, labels_array)
        """
        all_features = []
        all_labels = []
        
        categories = ['AF_Approved', 'PAUSE_Approved', 'VTACH_Approved']
        
        for category in categories:
            category_path = self.data_path / category
            
            if not category_path.exists():
                print(f"Warning: {category_path} does not exist")
                continue
            
            label = self.label_map[category]
            print(f"\nProcessing {category} ({label})...")
            
            # Get all patient folders
            patient_folders = [f for f in category_path.iterdir() if f.is_dir()]
            
            for idx, patient_folder in enumerate(patient_folders, 1):
                try:
                    # Read ECG data
                    ecg_data = self.read_ecg_files(patient_folder)
                    
                    if len(ecg_data) == 0:
                        print(f"  Warning: No ECG data found in {patient_folder.name}")
                        continue
                    
                    # Extract features
                    features = self.processor.extract_all_features(ecg_data)
                    
                    all_features.append(features)
                    all_labels.append(label)
                    
                    print(f"  [{idx}/{len(patient_folders)}] Processed {patient_folder.name} - {len(ecg_data)} data points")
                    
                except Exception as e:
                    print(f"  Error processing {patient_folder.name}: {e}")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        labels_array = np.array(all_labels)
        
        print(f"\n✓ Total samples loaded: {len(features_df)}")
        print(f"✓ Feature count: {len(features_df.columns)}")
        print(f"✓ Label distribution: {dict(pd.Series(labels_array).value_counts())}")
        
        return features_df, labels_array
    
    def train_model(self, X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2) -> Dict:
        """
        Train Random Forest classifier
        
        Args:
            X: Features dataframe
            y: Labels array
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with training results
        """
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST CLASSIFIER")
        print("="*60)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        print("\nTraining Random Forest...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        # Training accuracy
        train_pred = self.model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, train_pred)
        print(f"\nTraining Accuracy: {train_accuracy:.4f}")
        
        # Test accuracy
        test_pred = self.model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, test_pred)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Classification report
        print("\n" + "-"*60)
        print("CLASSIFICATION REPORT (Test Set)")
        print("-"*60)
        print(classification_report(y_test, test_pred))
        
        # Confusion matrix
        print("-"*60)
        print("CONFUSION MATRIX (Test Set)")
        print("-"*60)
        cm = confusion_matrix(y_test, test_pred)
        classes = sorted(np.unique(y))
        
        # Print confusion matrix
        print("\n" + " "*15 + "  ".join(f"{c:>8}" for c in classes))
        for i, actual_class in enumerate(classes):
            print(f"{actual_class:>12}  " + "  ".join(f"{cm[i][j]:>8}" for j in range(len(classes))))
        
        # Feature importance
        print("\n" + "-"*60)
        print("TOP 10 MOST IMPORTANT FEATURES")
        print("-"*60)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(feature_importance.head(10).to_string(index=False))
        
        results = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance.to_dict('records')
        }
        
        return results
    
    def save_model(self, model_path: str = "ml/models"):
        """
        Save trained model and scaler
        
        Args:
            model_path: Path to save model files
        """
        model_dir = Path(model_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = model_dir / "ecg_classifier.pkl"
        joblib.dump(self.model, model_file)
        print(f"\n✓ Model saved to {model_file}")
        
        # Save scaler
        scaler_file = model_dir / "scaler.pkl"
        joblib.dump(self.scaler, scaler_file)
        print(f"✓ Scaler saved to {scaler_file}")
        
        # Save feature names and label map
        metadata = {
            'feature_names': self.feature_names,
            'label_map': self.label_map,
            'reverse_label_map': self.reverse_label_map
        }
        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved to {metadata_file}")
    
    def load_model(self, model_path: str = "ml/models"):
        """
        Load trained model and scaler
        
        Args:
            model_path: Path to model files
        """
        model_dir = Path(model_path)
        
        # Load model
        model_file = model_dir / "ecg_classifier.pkl"
        self.model = joblib.load(model_file)
        
        # Load scaler
        scaler_file = model_dir / "scaler.pkl"
        self.scaler = joblib.load(scaler_file)
        
        # Load metadata
        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['feature_names']
        self.label_map = metadata['label_map']
        self.reverse_label_map = metadata['reverse_label_map']
        
        print(f"✓ Model loaded from {model_path}")


def main():
    """Main training script"""
    print("="*60)
    print("ECG ARRHYTHMIA CLASSIFIER TRAINING")
    print("="*60)
    
    # Initialize trainer
    trainer = ECGClassifierTrainer(data_path="test-trifetch")
    
    # Load data
    print("\n" + "="*60)
    print("LOADING TRAINING DATA")
    print("="*60)
    X, y = trainer.load_training_data()
    
    # Train model
    results = trainer.train_model(X, y)
    
    # Save model
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    trainer.save_model()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\n✓ Final Test Accuracy: {results['test_accuracy']:.4f}")
    print("✓ Model ready for deployment")


if __name__ == "__main__":
    main()
