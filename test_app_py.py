#!/usr/bin/env python3
"""
Test suite for the Flask application
"""

import unittest
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, load_model_artifacts, predict_heart_disease, validate_input

class TestFlaskApp(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.app = app.test_client()
        self.app.testing = True
        
        # Load model artifacts for testing
        load_model_artifacts()
    
    def test_home_page(self):
        """Test the home page loads successfully"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Heart Disease Risk Assessment', response.data)
    
    def test_health_check(self):
        """Test the health check endpoint"""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('model_status', data)
    
    def test_model_info(self):
        """Test the model info endpoint"""
        response = self.app.get('/model_info')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('model_type', data)
        self.assertIn('features_count', data)
        self.assertIn('feature_names', data)
    
    def test_valid_form_prediction(self):
        """Test form prediction with valid data"""
        form_data = {
            'age': '50',
            'sex': 'M',
            'chest_pain_type': 'ATA',
            'resting_bp': '120',
            'cholesterol': '200',
            'fasting_bs': '0',
            'resting_ecg': 'Normal',
            'max_hr': '150',
            'exercise_angina': 'N',
            'oldpeak': '0.0',
            'st_slope': 'Up'
        }
        
        response = self.app.post('/predict', data=form_data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        # Should redirect to results page
        self.assertIn(b'Prediction Results', response.data)
    
    def test_invalid_form_data(self):
        """Test form prediction with invalid data"""
        form_data = {
            'age': '200',  # Invalid age
            'sex': 'M',
            'chest_pain_type': 'ATA',
            'resting_bp': '120',
            'cholesterol': '200',
            'fasting_bs': '0',
            'resting_ecg': 'Normal',
            'max_hr': '150',
            'exercise_angina': 'N',
            'oldpeak': '0.0',
            'st_slope': 'Up'
        }
        
        response = self.app.post('/predict', data=form_data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        # Should redirect back to home with error message
        self.assertIn(b'Age must be between 1 and 120', response.data)
    
    def test_api_predict_valid(self):
        """Test API prediction with valid JSON data"""
        valid_data = {
            "Age": 50,
            "Sex": "M",
            "ChestPainType": "ATA",
            "RestingBP": 120,
            "Cholesterol": 200,
            "FastingBS": 0,
            "RestingECG": "Normal",
            "MaxHR": 150,
            "ExerciseAngina": "N",
            "Oldpeak": 0.0,
            "ST_Slope": "Up"
        }
        
        response = self.app.post('/api/predict',
                               data=json.dumps(valid_data),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('prediction', data)
        self.assertIn('probability', data['prediction'])
        self.assertIn('prediction', data['prediction'])
        self.assertIn('risk_level', data['prediction'])
    
    def test_api_predict_missing_field(self):
        """Test API prediction with missing required field"""
        invalid_data = {
            "Age": 50,
            "Sex": "M",
            # Missing ChestPainType
            "RestingBP": 120,
            "Cholesterol": 200,
            "FastingBS": 0,
            "RestingECG": "Normal",
            "MaxHR": 150,
            "ExerciseAngina": "N",
            "Oldpeak": 0.0,
            "ST_Slope": "Up"
        }
        
        response = self.app.post('/api/predict',
                               data=json.dumps(invalid_data),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('Missing field: ChestPainType', data['error'])
    
    def test_api_predict_invalid_values(self):
        """Test API prediction with invalid values"""
        invalid_data = {
            "Age": 50,
            "Sex": "Invalid",  # Invalid sex
            "ChestPainType": "ATA",
            "RestingBP": 120,
            "Cholesterol": 200,
            "FastingBS": 0,
            "RestingECG": "Normal",
            "MaxHR": 150,
            "ExerciseAngina": "N",
            "Oldpeak": 0.0,
            "ST_Slope": "Up"
        }
        
        response = self.app.post('/api/predict',
                               data=json.dumps(invalid_data),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)

class TestValidationFunctions(unittest.TestCase):
    
    def test_validate_input_valid_data(self):
        """Test input validation with valid data"""
        valid_data = {
            'Age': 50,
            'Sex': 'M',
            'ChestPainType': 'ATA',
            'RestingBP': 120,
            'Cholesterol': 200,
            'FastingBS': 0,
            'RestingECG': 'Normal',
            'MaxHR': 150,
            'ExerciseAngina': 'N',
            'Oldpeak': 0.0,
            'ST_Slope': 'Up'
        }
        
        result = validate_input(valid_data)
        self.assertIsNone(result)  # Should return None for valid data
    
    def test_validate_input_age_range(self):
        """Test age validation"""
        # Test minimum age
        data = {'Age': 0}
        result = validate_input(data)
        self.assertIn('Age must be between 1 and 120', result)
        
        # Test maximum age
        data = {'Age': 121}
        result = validate_input(data)
        self.assertIn('Age must be between 1 and 120', result)
        
        # Test valid age
        data = {'Age': 50}
        result = validate_input(data)
        self.assertIsNone(result)
    
    def test_validate_input_categorical_values(self):
        """Test categorical field validation"""
        # Test invalid sex
        data = {'Sex': 'X'}
        result = validate_input(data)
        self.assertIn('Sex must be one of', result)
        
        # Test invalid chest pain type
        data = {'ChestPainType': 'INVALID'}
        result = validate_input(data)
        self.assertIn('ChestPainType must be one of', result)
    
    def test_validate_input_numerical_ranges(self):
        """Test numerical field range validation"""
        # Test RestingBP
        data = {'RestingBP': 300}
        result = validate_input(data)
        self.assertIn('Resting BP must be between', result)
        
        # Test Cholesterol
        data = {'Cholesterol': 700}
        result = validate_input(data)
        self.assertIn('Cholesterol must be between', result)
        
        # Test MaxHR
        data = {'MaxHR': 30}
        result = validate_input(data)
        self.assertIn('Max HR must be between', result)

class TestPredictionFunction(unittest.TestCase):
    
    def setUp(self):
        """Load model artifacts before testing"""
        load_model_artifacts()
    
    def test_predict_heart_disease_valid(self):
        """Test heart disease prediction with valid data"""
        patient_data = {
            'Age': 50,
            'Sex': 'M',
            'ChestPainType': 'ATA',
            'RestingBP': 120,
            'Cholesterol': 200,
            'FastingBS': 0,
            'RestingECG': 'Normal',
            'MaxHR': 150,
            'ExerciseAngina': 'N',
            'Oldpeak': 0.0,
            'ST_Slope': 'Up'
        }
        
        result, error = predict_heart_disease(patient_data)
        
        self.assertIsNone(error)
        self.assertIsNotNone(result)
        self.assertIn('probability', result)
        self.assertIn('prediction', result)
        self.assertIn('risk_level', result)
        
        # Check data types and ranges
        self.assertIsInstance(result['probability'], float)
        self.assertTrue(0 <= result['probability'] <= 1)
        self.assertIn(result['prediction'], [0, 1])
        self.assertIn(result['risk_level'], ['Low', 'Moderate', 'High'])

if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [TestFlaskApp, TestValidationFunctions, TestPredictionFunction]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)