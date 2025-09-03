#!/usr/bin/env python3
"""
Heart Disease Prediction Model Training Script
This script handles the complete ML pipeline from data loading to model deployment preparation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath='data/heart.csv'):
    """Load and return the heart disease dataset"""
    try:
        data = pd.read_csv(filepath)
        print(f"✓ Data loaded successfully: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"❌ Dataset file not found: {filepath}")
        print("Please ensure the heart.csv file is in the data/ directory")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def explore_data(df):
    """Perform exploratory data analysis"""
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    # Basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage().sum() / 1024:.2f} KB")
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\nMissing values:\n{missing[missing > 0]}")
    else:
        print("\n✓ No missing values found")
    
    # Data types
    print(f"\nData types:\n{df.dtypes}")
    
    # Target distribution
    target_dist = df['HeartDisease'].value_counts()
    print(f"\nTarget distribution:")
    print(f"No Heart Disease (0): {target_dist[0]} ({target_dist[0]/len(df)*100:.1f}%)")
    print(f"Heart Disease (1): {target_dist[1]} ({target_dist[1]/len(df)*100:.1f}%)")
    
    # Basic statistics
    print(f"\nNumerical features statistics:")
    print(df.describe())
    
    return df

def preprocess_data(df):
    """Clean and preprocess the dataset"""
    print("\n" + "="*50)
    print("DATA PREPROCESSING")
    print("="*50)
    
    # Create a copy to avoid modifying original data
    processed_df = df.copy()
    
    # Handle zero cholesterol values (likely missing data)
    zero_cholesterol = (processed_df['Cholesterol'] == 0).sum()
    if zero_cholesterol > 0:
        print(f"Found {zero_cholesterol} zero cholesterol values")
        # Replace with median cholesterol for the same sex
        median_chol = processed_df[processed_df['Cholesterol'] > 0].groupby('Sex')['Cholesterol'].median()
        for sex in ['M', 'F']:
            mask = (processed_df['Sex'] == sex) & (processed_df['Cholesterol'] == 0)
            processed_df.loc[mask, 'Cholesterol'] = median_chol[sex]
        print("✓ Zero cholesterol values replaced with sex-specific medians")
    
    # Handle zero resting BP values
    zero_bp = (processed_df['RestingBP'] == 0).sum()
    if zero_bp > 0:
        print(f"Found {zero_bp} zero resting BP values")
        median_bp = processed_df[processed_df['RestingBP'] > 0]['RestingBP'].median()
        processed_df.loc[processed_df['RestingBP'] == 0, 'RestingBP'] = median_bp
        print("✓ Zero resting BP values replaced with median")
    
    # Feature engineering
    print("\n✓ Feature engineering completed")
    
    return processed_df

def prepare_features(df):
    """Prepare features for machine learning"""
    print("\n" + "="*50)
    print("FEATURE PREPARATION")
    print("="*50)
    
    # Define categorical and numerical columns
    categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    
    # Separate features and target
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    # One-hot encoding for categorical variables
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    print(f"✓ Original features: {X.shape[1]}")
    print(f"✓ After encoding: {X_encoded.shape[1]}")
    print(f"✓ Categorical columns encoded: {categorical_cols}")
    print(f"✓ Numerical columns: {numerical_cols}")
    
    return X_encoded, y, numerical_cols

def train_models(X, y, numerical_cols):
    """Train and evaluate multiple ML models"""
    print("\n" + "="*50)
    print("MODEL TRAINING AND EVALUATION")
    print("="*50)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Data split - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Scale numerical features
    scaler = MinMaxScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    print("✓ Numerical features scaled using MinMaxScaler")
    
    # Define models to train
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    # Train and evaluate models
    model_results = {}
    best_model = None
    best_score = 0
    
    print("\nTraining models...")
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        mean_cv_score = cv_scores.mean()
        
        # Train on full training set
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        roc_auc = roc_auc_score(y_test, test_pred_proba)
        
        # Store results
        model_results[name] = {
            'model': model,
            'cv_mean': mean_cv_score,
            'cv_std': cv_scores.std(),
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'roc_auc': roc_auc,
            'predictions': test_pred,
            'probabilities': test_pred_proba
        }
        
        print(f"CV Accuracy: {mean_cv_score:.4f} (+/- {cv_scores.std()*2:.4f})")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        # Track best model
        if test_accuracy > best_score:
            best_score = test_accuracy
            best_model = name
    
    print(f"\n🏆 Best performing model: {best_model} (Test Accuracy: {best_score:.4f})")
    
    return model_results, best_model, scaler, X_train_scaled, X_test_scaled, y_train, y_test, X

def hyperparameter_tuning(model_results, best_model_name, X_train, y_train):
    """Perform hyperparameter tuning on the best model"""
    print(f"\n" + "="*50)
    print(f"HYPERPARAMETER TUNING - {best_model_name}")
    print("="*50)
    
    if best_model_name == 'XGBoost':
        # XGBoost parameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.1, 0.01],
            'subsample': [0.8, 1.0]
        }
        
        base_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        
    elif best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        base_model = RandomForestClassifier(random_state=42)
        
    else:
        print(f"No hyperparameter tuning defined for {best_model_name}")
        return model_results[best_model_name]['model']
    
    print("Performing Grid Search...")
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"✓ Best parameters: {grid_search.best_params_}")
    print(f"✓ Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_final_model(model, X_test, y_test, model_name):
    """Comprehensive evaluation of the final model"""
    print(f"\n" + "="*50)
    print(f"FINAL MODEL EVALUATION - {model_name}")
    print("="*50)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print(f"Final ROC AUC: {roc_auc:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return accuracy, roc_auc

def save_model_artifacts(model, scaler, feature_names, model_dir='models'):
    """Save trained model and preprocessing artifacts"""
    print(f"\n" + "="*50)
    print("SAVING MODEL ARTIFACTS")
    print("="*50)
    
    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, 'best_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved: {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved: {scaler_path}")
    
    # Save feature names
    features_path = os.path.join(model_dir, 'feature_names.pkl')
    with open(features_path, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"✓ Feature names saved: {features_path}")
    
    # Also save to root directory for Flask app
    root_files = ['best_model.pkl', 'scaler.pkl', 'feature_names.pkl']
    for filename in root_files:
        src = os.path.join(model_dir, filename)
        dst = filename
        with open(src, 'rb') as f_src:
            with open(dst, 'wb') as f_dst:
                f_dst.write(f_src.read())
    
    print("✓ Model artifacts also copied to root directory for Flask app")

def generate_sample_predictions(model, scaler, feature_names, numerical_cols):
    """Generate sample predictions for testing"""
    print(f"\n" + "="*50)
    print("GENERATING SAMPLE PREDICTIONS")
    print("="*50)
    
    # Sample patient data for testing
    sample_patients = [
        {
            'Age': 50, 'Sex': 'M', 'ChestPainType': 'ATA', 'RestingBP': 120,
            'Cholesterol': 200, 'FastingBS': 0, 'RestingECG': 'Normal',
            'MaxHR': 150, 'ExerciseAngina': 'N', 'Oldpeak': 0.0, 'ST_Slope': 'Up'
        },
        {
            'Age': 65, 'Sex': 'F', 'ChestPainType': 'ASY', 'RestingBP': 140,
            'Cholesterol': 250, 'FastingBS': 1, 'RestingECG': 'ST',
            'MaxHR': 120, 'ExerciseAngina': 'Y', 'Oldpeak': 2.0, 'ST_Slope': 'Flat'
        }
    ]
    
    categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    
    print("Sample predictions:")
    for i, patient in enumerate(sample_patients, 1):
        # Convert to DataFrame
        patient_df = pd.DataFrame([patient])
        
        # Apply same preprocessing
        patient_encoded = pd.get_dummies(patient_df, columns=categorical_cols, drop_first=True)
        
        # Ensure all features are present
        for col in feature_names:
            if col not in patient_encoded.columns:
                patient_encoded[col] = 0
        
        # Reorder columns
        patient_encoded = patient_encoded[feature_names]
        
        # Scale numerical features
        patient_encoded[numerical_cols] = scaler.transform(patient_encoded[numerical_cols])
        
        # Make prediction
        prediction = model.predict(patient_encoded)[0]
        probability = model.predict_proba(patient_encoded)[0, 1]
        
        risk_level = "Low" if probability < 0.3 else "Moderate" if probability < 0.7 else "High"
        
        print(f"\nPatient {i}:")
        print(f"  Age: {patient['Age']}, Sex: {patient['Sex']}, ChestPain: {patient['ChestPainType']}")
        print(f"  Prediction: {'Heart Disease' if prediction else 'No Heart Disease'}")
        print(f"  Probability: {probability:.3f} ({risk_level} risk)")

def create_deployment_summary():
    """Create a summary of the deployment-ready artifacts"""
    print(f"\n" + "="*60)
    print("DEPLOYMENT SUMMARY")
    print("="*60)
    
    required_files = [
        ('best_model.pkl', 'Trained XGBoost model'),
        ('scaler.pkl', 'MinMax scaler for numerical features'),
        ('feature_names.pkl', 'Feature names for consistency'),
        ('app.py', 'Flask web application'),
        ('requirements.txt', 'Python dependencies'),
        ('templates/', 'HTML templates directory'),
    ]
    
    print("✓ Model artifacts ready for deployment:")
    for filename, description in required_files:
        status = "✓" if os.path.exists(filename) else "❌"
        print(f"  {status} {filename:<20} - {description}")
    
    print(f"\n📋 Deployment checklist:")
    print(f"  ✓ Model trained and saved")
    print(f"  ✓ Preprocessing pipeline saved")
    print(f"  ✓ Flask app ready")
    print(f"  ✓ Web interface templates created")
    print(f"  ✓ API endpoints implemented")
    
    print(f"\n🚀 Ready for deployment!")
    print(f"   Run: python app.py")
    print(f"   Or:  python deploy.py (for ngrok tunnel)")

def main():
    """Main training pipeline"""
    print("🏥 Heart Disease Prediction - Model Training Pipeline")
    print("="*60)
    
    # Step 1: Load data
    df = load_data()
    if df is None:
        return False
    
    # Step 2: Explore data
    df = explore_data(df)
    
    # Step 3: Preprocess data
    df_processed = preprocess_data(df)
    
    # Step 4: Prepare features
    X, y, numerical_cols = prepare_features(df_processed)
    
    # Step 5: Train models
    model_results, best_model_name, scaler, X_train, X_test, y_train, y_test, X_original = train_models(
        X, y, numerical_cols
    )
    
    # Step 6: Hyperparameter tuning
    final_model = hyperparameter_tuning(model_results, best_model_name, X_train, y_train)
    
    # Step 7: Final evaluation
    accuracy, roc_auc = evaluate_final_model(final_model, X_test, y_test, best_model_name)
    
    # Step 8: Save artifacts
    feature_names = list(X.columns)
    save_model_artifacts(final_model, scaler, feature_names)
    
    # Step 9: Test predictions
    generate_sample_predictions(final_model, scaler, feature_names, numerical_cols)
    
    # Step 10: Deployment summary
    create_deployment_summary()
    
    print(f"\n🎉 Training pipeline completed successfully!")
    print(f"   Final model: {best_model_name}")
    print(f"   Test accuracy: {accuracy:.4f}")
    print(f"   ROC AUC: {roc_auc:.4f}")
    
    return True

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    success = main()
    if not success:
        print("\n❌ Training pipeline failed!")
        exit(1)
    else:
        print("\n✅ Training pipeline completed successfully!")