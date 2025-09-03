# 🫀 Heart Disease Prediction System

A machine learning-powered web application that predicts heart disease risk using patient medical data. Built with Flask, scikit-learn, and XGBoost.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v2.3.3-green.svg)
![XGBoost](https://img.shields.io/badge/xgboost-v1.7.6-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Features

- **Interactive Web Interface**: User-friendly form for entering patient data
- **Real-time Predictions**: Instant heart disease risk assessment
- **REST API**: Programmatic access for integration with other systems
- **Risk Stratification**: Categorizes risk as Low, Moderate, or High
- **Model Transparency**: Displays prediction confidence and patient data summary
- **Responsive Design**: Works on desktop and mobile devices

## 🚀 Live Demo

The application can be deployed locally or on cloud platforms. See [Deployment](#deployment) section for instructions.

## 📊 Model Performance

Our XGBoost classifier achieves:
- **Accuracy**: ~85-90% on test data
- **Precision**: High precision for positive cases
- **Recall**: Balanced sensitivity and specificity
- **Features**: Uses 11 key medical indicators

## 🏥 Medical Features Used

The model uses the following patient data:

| Feature | Description | Range/Values |
|---------|-------------|--------------|
| Age | Patient age in years | 1-120 |
| Sex | Biological sex | Male/Female |
| ChestPainType | Type of chest pain experienced | TA, ATA, NAP, ASY |
| RestingBP | Resting blood pressure (mmHg) | 50-250 |
| Cholesterol | Serum cholesterol (mg/dl) | 0-600 |
| FastingBS | Fasting blood sugar > 120 mg/dl | Yes/No |
| RestingECG | Resting electrocardiogram results | Normal, ST, LVH |
| MaxHR | Maximum heart rate achieved | 60-220 |
| ExerciseAngina | Exercise-induced angina | Yes/No |
| Oldpeak | ST depression induced by exercise | -3 to 7 |
| ST_Slope | Slope of peak exercise ST segment | Up, Flat, Down |

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model (if artifacts don't exist)**
   ```bash
   python train_model.py
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   - Local: http://localhost:5000
   - For public access, see [ngrok setup](#ngrok-tunnel)

## 📁 Project Structure

```
heart-disease-prediction/
├── app.py                    # Main Flask application
├── train_model.py           # Model training script
├── deploy.py                # Deployment automation script
├── requirements.txt         # Python dependencies
├── runtime.txt             # Python version for Heroku
├── Procfile                # Heroku deployment configuration
├── .gitignore              # Git ignore rules
├── README.md               # This file
├── LICENSE                 # MIT license
├── data/
│   └── heart.csv           # Training dataset
├── models/
│   ├── best_model.pkl      # Trained XGBoost model
│   ├── scaler.pkl          # Feature scaler
│   └── feature_names.pkl   # Feature names for consistency
├── templates/
│   ├── base.html           # Base HTML template
│   ├── index.html          # Home page with form
│   └── result.html         # Prediction results page
├── static/
│   ├── css/
│   │   └── style.css       # Custom styles
│   ├── js/
│   │   └── main.js         # JavaScript functionality
│   └── images/
│       └── logo.png        # Application logo
├── tests/
│   ├── test_app.py         # Application tests
│   ├── test_model.py       # Model tests
│   └── test_api.py         # API endpoint tests
└── docs/
    ├── api.md              # API documentation
    ├── deployment.md       # Deployment guide
    └── model_details.md    # Model architecture details
```

## 🔧 Usage

### Web Interface

1. Navigate to the home page
2. Fill in the patient information form
3. Click "Predict Risk" to get results
4. View the risk assessment and patient summary

### API Usage

The application provides a REST API for programmatic access:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "probability": 0.23,
    "prediction": 0,
    "risk_level": "Low"
  },
  "input_data": { ... }
}
```

### Available Endpoints

- `GET /` - Home page with prediction form
- `POST /predict` - Web form prediction submission
- `POST /api/predict` - API endpoint for predictions
- `GET /health` - Health check endpoint
- `GET /model_info` - Model information and metadata

## 🚀 Deployment

### Local Development with ngrok Tunnel

For testing with external access:

```bash
# Install ngrok
pip install pyngrok

# Get your auth token from https://dashboard.ngrok.com/get-started/your-authtoken
# Set it in deploy.py or as environment variable

# Run deployment script
python deploy.py
```

### Heroku Deployment

1. **Install Heroku CLI**
2. **Login to Heroku**
   ```bash
   heroku login
   ```

3. **Create Heroku app**
   ```bash
   heroku create your-app-name
   ```

4. **Deploy**
   ```bash
   git push heroku main
   ```

### Docker Deployment

```bash
# Build image
docker build -t heart-disease-prediction .

# Run container
docker run -p 5000:5000 heart-disease-prediction
```

### Cloud Platforms

The application is compatible with:
- **Heroku** (configuration included)
- **Google Cloud Platform**
- **AWS Elastic Beanstalk**
- **Azure App Service**
- **DigitalOcean App Platform**

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=app

# Run specific test file
python -m pytest tests/test_app.py -v
```

## 📈 Model Details

### Algorithm
- **Primary Model**: XGBoost Classifier
- **Preprocessing**: MinMax scaling for numerical features
- **Encoding**: One-hot encoding for categorical features
- **Cross-validation**: 5-fold stratified cross-validation

### Training Dataset
- **Source**: Heart Disease Dataset (commonly used in ML research)
- **Size**: ~900+ samples
- **Features**: 11 medical indicators
- **Target**: Binary classification (Heart Disease: Yes/No)

### Performance Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC for model discrimination
- Confusion Matrix for detailed analysis

## ⚠️ Important Disclaimers

- **This is for educational and research purposes only**
- **NOT a substitute for professional medical diagnosis**
- **Always consult healthcare professionals for medical decisions**
- **Predictions are probabilistic estimates, not definitive diagnoses**

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Ensure all tests pass

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Documentation**: [docs/](docs/)
- **API Reference**: [docs/api.md](docs/api.md)
- **Deployment Guide**: [docs/deployment.md](docs/deployment.md)
- **Issue Tracker**: GitHub Issues
- **Discussions**: GitHub Discussions

## 👥 Authors

- **Loyanganba Ngathem** - *Initial work* - https://github.com/LoyaNg-rgb/Heart-Disease-Prediction-using-Machine-Learning
  
## 🙏 Acknowledgments

- Heart Disease Dataset contributors
- Flask and scikit-learn communities
- Medical professionals who provided domain expertise
- Open source contributors

## 📞 Support

If you have questions or need help:

- 📧 Email: loyanganba.ngathem@gmail.com
- 💬 GitHub Discussions
- 🐛 GitHub Issues for bugs
- 📖 Check the [documentation](docs/)

---

**Made with ❤️ for better healthcare predictions**
