# 📊 Churn Intelligence Dashboard

AI-powered customer churn prediction system built with Streamlit and XGBoost following MLOps best practices.

## 🚀 Features

- **Real-time Churn Prediction**: Predict customer churn probability based on usage patterns
- **Interactive Dashboard**: Beautiful Streamlit interface with Material Design aesthetics
- **ML-Powered**: XGBoost classifier trained with GridSearch optimization
- **MLOps Ready**: Structured following industry best practices
- **CI/CD Pipeline**: Automated testing and Docker deployment
- **Production Ready**: Containerized with comprehensive testing

## 📁 MLOps Project Structure

```
churn/
├── .github/
│   └── workflows/
│       ├── docker-build-push.yml    # Docker CI/CD with testing
│       └── render-cd.yml            # Render deployment
├── .venv/                           # Virtual environment (gitignored)
├── config/
│   ├── config.yaml                  # Application configuration
│   └── logging_config.yaml          # Logging setup
├── data/
│   └── customer_subscription_churn_usage_patterns.csv
├── logs/                            # Application logs (gitignored)
├── models/
│   └── best_classifier.pkl          # Trained XGBoost model
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── src/
│   ├── __init__.py                  # Package initialization
│   ├── app.py                       # Streamlit application
│   ├── data_processing.py           # Data preprocessing
│   ├── helper_functions.py          # Utility functions
│   ├── main.py                      # Entry point
│   └── ml_functions.py              # ML operations
├── tests/
│   ├── __init__.py
│   ├── test_app.py                  # App integration tests
│   ├── test_data_processing.py      # Data processing tests
│   └── test_ml_functions.py         # ML function tests
├── .env                             # Environment variables
├── .gitignore                       # Git ignore rules
├── Dockerfile                       # Container configuration
├── LICENSE                          # MIT License
├── README.md                        # This file
└── requirements.txt                 # Python dependencies
```

## 🛠️ Installation

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd churn
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run src/app.py
   ```

   Or use the main entry point:
   ```bash
   python src/main.py
   ```

### Docker Setup

1. **Build the image**
   ```bash
   docker build -t churn-prediction .
   ```

2. **Run the container**
   ```bash
   docker run -p 8501:8501 churn-prediction
   ```

3. **Access the app**
   Open your browser to `http://localhost:8501`

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src

# Run specific test file
pytest tests/test_ml_functions.py -v
```

## 📊 Model Training

To retrain the model with new data:

```python
from src.ml_functions import train_churn_model

# Train model
model, metrics = train_churn_model(
    data_path="data/customer_subscription_churn_usage_patterns.csv",
    output_path="models/best_classifier.pkl"
)

print(f"Model Accuracy: {metrics['accuracy']}")
```

## 📓 Notebooks

Jupyter notebooks for exploration and analysis:

```bash
# Start Jupyter
jupyter notebook

# Navigate to notebooks/
# - 01_exploratory_data_analysis.ipynb
```

## 🔧 Configuration

### Environment Variables (`.env`)
- Model paths
- Data paths
- Streamlit server settings
- Docker credentials
- Deployment hooks

### Application Config (`config/config.yaml`)
- App settings
- Model hyperparameters
- Training configuration
- Logging settings

## 🚢 Deployment

### CI/CD Pipeline

The project includes automated workflows:

1. **Testing**: Runs pytest on every push/PR
2. **Docker Build & Push**: Builds and pushes Docker images after tests pass
3. **Render Deployment**: Triggers deployment to Render platform

Configure GitHub secrets:
- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`
- `RENDER_DEPLOY_HOOK`

### Manual Deployment

Deploy to any platform supporting Docker or Python:
- Render
- Heroku
- AWS ECS
- Google Cloud Run
- Azure Container Instances

## 📈 Usage

1. **Input Customer Data**: Use the sidebar to input customer features
   - Plan Type (Basic/Premium)
   - Monthly Fee
   - Weekly Usage Hours
   - Support Tickets
   - Payment Failures
   - Tenure (Months)
   - Last Login Gap (Days)

2. **Predict**: Click "Predict Churn Risk" button

3. **View Results**: See churn probability and risk assessment

## 🧪 Model Performance

The XGBoost model is optimized using GridSearch with:
- Cross-validation (3-fold)
- Hyperparameter tuning
- Multiple evaluation metrics

Typical performance:
- Accuracy: ~85-90%
- Precision: High for churn detection
- Recall: Balanced for both classes

## 🏗️ Development

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/
```

### Project Structure Philosophy

This project follows MLOps best practices:
- **src/**: Source code as a Python package
- **models/**: ML artifacts and trained models
- **tests/**: Comprehensive test coverage
- **notebooks/**: Exploratory analysis
- **config/**: Centralized configuration
- **logs/**: Application logging

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

Built with ❤️ using Streamlit, XGBoost, and MLOps best practices

## 🙏 Acknowledgments

- Streamlit for the amazing web framework
- XGBoost for powerful ML capabilities
- scikit-learn for preprocessing utilities
- pytest for testing framework

---

© 2026 Churn Intelligence • Built with Streamlit & XGBoost • MLOps Ready
