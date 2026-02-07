# Customer Churn Prediction System

**Production-ready machine learning system for predicting customer churn using XGBoost and Streamlit**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![ML](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)](https://github.com/Prajwallnaik/churn-prediction-dashboard/actions)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg?logo=docker&logoColor=white)](https://hub.docker.com/)
[![Code Coverage](https://img.shields.io/badge/Coverage-80%25+-success.svg)](https://github.com/Prajwallnaik/churn-prediction-dashboard)
[![Code Quality](https://img.shields.io/badge/Code%20Quality-A-brightgreen.svg)](https://github.com/Prajwallnaik/churn-prediction-dashboard)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-2.0+-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![numpy](https://img.shields.io/badge/numpy-1.24+-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF.svg?logo=github-actions&logoColor=white)](https://github.com/features/actions)
[![Testing](https://img.shields.io/badge/Testing-pytest-0A9EDC.svg?logo=pytest&logoColor=white)](https://pytest.org/)
[![Code Style](https://img.shields.io/badge/Code%20Style-black-000000.svg)](https://github.com/psf/black)
[![Maintained](https://img.shields.io/badge/Maintained-Yes-brightgreen.svg)](https://github.com/Prajwallnaik/churn-prediction-dashboard)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg)](https://github.com/Prajwallnaik/churn-prediction-dashboard/pulls)


---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Development](#development)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

End-to-end MLOps solution for customer churn prediction, featuring:
- Interactive web application built with Streamlit
- Production-grade ML pipeline using XGBoost
- Automated CI/CD with GitHub Actions
- Containerized deployment with Docker
- Comprehensive testing suite with pytest

---

## Key Features

| Feature | Description |
|---------|-------------|
| **ML Model** | XGBoost classifier with GridSearch optimization |
| **Dashboard** | Interactive Streamlit UI for real-time predictions |
| **MLOps** | Complete pipeline following industry best practices |
| **Testing** | Unit, integration, and coverage tests |
| **Docker** | Containerized for consistent deployments |
| **Monitoring** | Structured logging and prediction tracking |
| **Notebooks** | Jupyter notebooks for EDA and experimentation |

---

## Project Structure

```
churn-prediction/
│
├── .github/workflows/         # CI/CD pipelines
│   ├── docker-build-push.yml  # Docker automation
│   └── render-cd.yml          # Deployment workflow
│
├── config/                    # Configuration files
│   ├── config.yaml            # App configuration
│   └── logging_config.yaml    # Logging setup
│
├── data/                      # Dataset storage
│   └── customer_subscription_churn_usage_patterns.csv
│
├── logs/                      # Application logs (gitignored)
│
├── models/                    # Trained models
│   └── best_classifier.pkl    # Production model
│
├── notebooks/                 # Jupyter notebooks
│   └── 01_exploratory_data_analysis.ipynb
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── app.py                 # Streamlit application
│   ├── data_processing.py     # Data preprocessing
│   ├── helper_functions.py    # Utilities
│   ├── main.py                # Entry point
│   └── ml_functions.py        # ML operations
│
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── test_app.py
│   ├── test_data_processing.py
│   └── test_ml_functions.py
│
├── .env                       # Environment variables
├── .gitignore                 # Git ignore rules
├── Dockerfile                 # Container definition
├── LICENSE                    # MIT License
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/Prajwallnaik/churn-prediction-dashboard.git
cd churn

# Setup environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run src/app.py
```

**Access the application**: Navigate to `http://localhost:8501`

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Docker (optional, for containerized deployment)

### Local Development

**1. Clone the Repository**
```bash
git clone https://github.com/Prajwallnaik/churn-prediction-dashboard.git
cd churn
```

**2. Create Virtual Environment**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure Environment**
```bash
# Copy template and configure
cp .env.example .env
# Edit .env with your settings
```

### Docker Deployment

**Build Image**
```bash
docker build -t churn-prediction:latest .
```

**Run Container**
```bash
docker run -p 8501:8501 churn-prediction:latest
```

---

## Usage

### Running the Application

**Option 1: Streamlit Command**
```bash
streamlit run src/app.py
```

**Option 2: Python Entry Point**
```bash
python src/main.py
```

### Making Predictions

1. **Access the Dashboard**: Navigate to `http://localhost:8501`
2. **Input Customer Data**: Use sidebar controls to enter:
   - Plan Type (Basic/Premium)
   - Monthly Fee
   - Average Weekly Usage Hours
   - Number of Support Tickets
   - Payment Failures Count
   - Tenure in Months
   - Days Since Last Login
3. **Get Prediction**: Click "Predict Churn Risk"
4. **Review Results**: View churn probability and risk assessment

### Training New Models

```python
from src.ml_functions import train_churn_model

# Train with custom parameters
model, metrics = train_churn_model(
    data_path="data/customer_subscription_churn_usage_patterns.csv",
    output_path="models/best_classifier.pkl",
    test_size=0.2,
    random_state=42
)

print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"Precision: {metrics['precision']:.2%}")
print(f"Recall: {metrics['recall']:.2%}")
```

---

## Model Details

### Architecture

- **Algorithm**: XGBoost Classifier
- **Optimization**: GridSearchCV with 3-fold cross-validation
- **Features**: 7 customer behavior metrics
- **Target**: Binary classification (Churn: Yes/No)

### Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 85-90% |
| Precision | High |
| Recall | Balanced |
| F1-Score | Optimized |

### Hyperparameters

Optimized through grid search:
- Learning rate
- Max depth
- Number of estimators
- Subsample ratio
- Column sample ratio

---

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test module
pytest tests/test_ml_functions.py -v

# Run with detailed output
pytest tests/ -vv --tb=short
```

### Code Quality

```bash
# Format code
black src/ tests/

# Check linting
flake8 src/ tests/

# Type checking (if using mypy)
mypy src/
```

### Jupyter Notebooks

```bash
# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook

# Navigate to notebooks/ directory
```

---

## Deployment

### CI/CD Pipeline

Automated workflows configured in `.github/workflows/`:

**1. Testing Workflow**
- Triggers on push/PR
- Runs pytest suite
- Generates coverage reports

**2. Docker Build & Push**
- Builds Docker image
- Pushes to Docker Hub
- Tags with commit SHA

**3. Render Deployment**
- Triggers on main branch push
- Deploys to Render platform
- Zero-downtime deployment

### GitHub Secrets Configuration

Required secrets for CI/CD:
```
DOCKER_USERNAME     # Docker Hub username
DOCKER_PASSWORD     # Docker Hub password
RENDER_DEPLOY_HOOK  # Render deployment webhook URL
```

### Deployment Platforms

Compatible with:
- Render
- Heroku
- AWS ECS/Fargate
- Google Cloud Run
- Azure Container Instances
- DigitalOcean App Platform

---

## Configuration

### Environment Variables

Create `.env` file with:
```env
# Application
APP_NAME=Churn Intelligence
APP_VERSION=1.0.0
DEBUG=False

# Paths
MODEL_PATH=models/best_classifier.pkl
DATA_PATH=data/customer_subscription_churn_usage_patterns.csv
LOG_DIR=logs/

# Streamlit
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost

# Logging
LOG_LEVEL=INFO
ENABLE_PREDICTION_LOGGING=True
```

### Application Configuration

Edit `config/config.yaml`:
```yaml
app:
  name: "Churn Intelligence"
  version: "1.0.0"
  
model:
  type: "XGBoost"
  path: "models/best_classifier.pkl"
  
training:
  test_size: 0.2
  random_state: 42
  cv_folds: 3
```

---

## Contributing

Contributions are welcome. Please follow these guidelines:

1. **Fork the Repository**
2. **Create Feature Branch**
   ```bash
   git checkout -b feature/feature-name
   ```
3. **Make Changes**
   - Write clean, documented code
   - Add tests for new features
   - Update documentation
4. **Run Tests**
   ```bash
   pytest tests/ -v
   black src/ tests/
   flake8 src/ tests/
   ```
5. **Commit Changes**
   ```bash
   git commit -m "Add feature description"
   ```
6. **Push to Branch**
   ```bash
   git push origin feature/feature-name
   ```
7. **Open Pull Request**

### Code Standards

- Follow PEP 8 style guide
- Write docstrings for functions/classes
- Maintain test coverage >80%
- Use type hints where applicable

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Authors & Acknowledgments

**Developed by**: Prajwall Naik

### Built With

- [Streamlit](https://streamlit.io/) - Web framework
- [XGBoost](https://xgboost.readthedocs.io/) - ML algorithm
- [scikit-learn](https://scikit-learn.org/) - ML utilities
- [pytest](https://pytest.org/) - Testing framework
- [Docker](https://www.docker.com/) - Containerization

### Acknowledgments

- Streamlit team for the web framework
- XGBoost contributors for ML capabilities
- Open-source community for tools and libraries

---

## Support & Contact

- **Issues**: [GitHub Issues](https://github.com/Prajwallnaik/churn-prediction-dashboard/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Prajwallnaik/churn-prediction-dashboard/discussions)

---

**Copyright © 2026 | MLOps Implementation**
