# 🌍 Earthquake Prediction & Severity Assessment System

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com)
[![CatBoost](https://img.shields.io/badge/CatBoost-ML-orange.svg)](https://catboost.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A real-time earthquake prediction system that leverages machine learning and geospatial data to predict earthquake occurrence and severity. The system uses live USGS earthquake data, tectonic plate boundaries, and geological features to provide actionable insights for disaster preparedness.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset & Feature Engineering](#dataset--feature-engineering)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a two-stage machine learning pipeline:

1. **Occurrence Model**: Predicts whether an earthquake will occur (magnitude ≥ 5.0) in a specific location
2. **Severity Model**: For predicted earthquakes, classifies severity as Medium (5.0-6.0) or High (≥6.0)

The system processes 30+ years of historical earthquake data (1964-1994) and integrates real-time seismic activity from the USGS API to make predictions.

### Key Highlights

- **200,000+ earthquake records** processed and analyzed
- **Real-time predictions** using live USGS data feeds
- **Geospatial analysis** with tectonic plate boundary integration
- **RESTful API** for easy integration
- **Docker support** for seamless deployment

## ✨ Features

### Data Processing
- Automated data cleaning and quality filtering
- Unit conversion across different magnitude scales (mb, ms, ml, mw)
- Rolling window feature extraction (1, 7, 30, 90 days)
- Tectonic boundary distance calculations using PB2002 dataset

### Machine Learning
- **CatBoost** ensemble models for occurrence and severity prediction
- Advanced feature engineering with 16 selected features
- Class imbalance handling using SMOTETomek
- Hyperparameter tuning with cross-validation

### API Capabilities
- Real-time earthquake occurrence prediction
- Severity classification for high-risk events
- Risk assessment with actionable recommendations
- Data quality metrics and transparency

### Geospatial Integration
- Distance to nearest tectonic plate boundary
- Boundary type classification (convergent, divergent, transform)
- Elevation-based crust type determination
- Global coverage with coordinate validation

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Application                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Server                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Endpoint: /predict                                   │  │
│  │  - Input validation                                   │  │
│  │  - Feature computation                                │  │
│  │  - Model inference                                    │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────┬────────────────────────┬───────────────────────┘
             │                        │
             ▼                        ▼
┌────────────────────────┐  ┌────────────────────────┐
│   External APIs        │  │   ML Models            │
│  ┌──────────────────┐  │  │  ┌──────────────────┐  │
│  │ USGS API         │  │  │  │ Occurrence Model │  │
│  │ - Historical data│  │  │  │ (CatBoost)       │  │
│  │ - Real-time feed │  │  │  └──────────────────┘  │
│  └──────────────────┘  │  │  ┌──────────────────┐  │
│  ┌──────────────────┐  │  │  │ Severity Model   │  │
│  │ Elevation API    │  │  │  │ (CatBoost)       │  │
│  └──────────────────┘  │  │  └──────────────────┘  │
└────────────────────────┘  └────────────────────────┘
             │                        │
             ▼                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   Feature Engineering                        │
│  - Rolling statistics (1d, 7d, 30d, 90d)                   │
│  - Seismic energy calculations                              │
│  - Tectonic boundary analysis                               │
│  - Temporal encoding (cyclic month)                         │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Dataset & Feature Engineering

### Data Sources

1. **Historical Earthquakes**: Global earthquake dataset (1964-1994, magnitude ≥ 4.5)
2. **USGS Real-time API**: Live earthquake data for feature computation
3. **PB2002 Tectonic Plates**: Plate boundary locations and classifications
4. **Open-Elevation API**: Terrain elevation data

### Feature Categories

#### Rolling Features (Dynamic)
- `count_prev_1d/7d/30d/90d`: Earthquake counts in time windows
- `meanmag_prev_*`: Average magnitude in time windows
- `maxmag_prev_*`: Maximum magnitude in time windows
- `log_energy_prev_*`: Logarithmic seismic energy release
- `days_since_last_event`: Time since most recent earthquake
- `rate_change_7d_vs_30d`: Acceleration of seismic activity

#### Spatial Features (Static)
- `dist_to_boundary_km`: Distance to nearest tectonic boundary
- `boundary_type`: Convergent (0), Divergent (1), Transform (2), Other (3)
- `elevation_m`: Terrain elevation
- `crust_type`: Oceanic (0) or Continental (1)

#### Temporal Features
- `month_sin`, `month_cos`: Cyclic encoding of seasonality

### Data Processing Pipeline

```python
# Stage 1: Data Cleaning
- Filter earthquake events only
- Remove high-error measurements (RMS ≥ 20)
- Filter out sparse observations (gap ≥ 350)
- Remove invalid stations (nst = 0)

# Stage 2: Unit Conversion
- Convert all magnitude scales to Moment Magnitude (Mw)
- mb → mw: mag * 0.85 + 1.03
- ms → mw: mag * 0.67 + 2.07
- ml → mw: mag * 0.69 + 1.08

# Stage 3: Feature Extraction
- Compute rolling statistics using custom Python script
- Calculate distances to tectonic boundaries
- Extract elevation and crust type

# Stage 4: Model Preparation
- Create binary labels (magnitude ≥ 5.0)
- Handle missing values with median/mean imputation
- Apply PowerTransformer for skewness reduction
- Balance classes using SMOTETomek
```

**Model Hyperparameters:**
```python
CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    class_weights=balanced
)
```

### Feature Importance (Top 10)

1. `maxmag_prev_7d` (0.183)
2. `meanmag_prev_30d` (0.156)
3. `log_energy_prev_90d` (0.142)
4. `dist_to_boundary_km` (0.128)
5. `days_since_last_event` (0.095)
6. `rate_change_7d_vs_30d` (0.087)
7. `boundary_type` (0.071)
8. `elevation_m` (0.058)
9. `meanmag_prev_1d` (0.046)
10. `crust_type` (0.034)

## 🚀 Installation

### Prerequisites

- Python 3.10+
- Docker (optional, for containerized deployment)
- 4GB+ RAM recommended

### Local Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/earthquake-prediction.git
cd earthquake-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download tectonic plate data
# Place PB2002_steps.shp in tectonicplates-master/ directory

# Run the API server
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Docker Deployment

```bash
# Build the Docker image
docker build -t earthquake-prediction .

# Run the container
docker run -p 7860:7860 earthquake-prediction
```

### Hugging Face Spaces

Deploy directly to Hugging Face Spaces:

1. Create a new Space with Docker SDK
2. Upload all files including Dockerfile
3. Add model files: `occurence_model.cbm`, `severity_model.cbm`
4. Add transformers: `occurence_transformer.joblib`, `severity_transformer.joblib`
5. Deploy automatically

## 💻 Usage

### API Request

```bash
curl -X POST "http://localhost:7860/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 35.0,
    "longitude": -118.0,
    "time": "2025-10-22T14:00:00"
  }'
```

### Python Client

```python
import requests

response = requests.post(
    "http://localhost:7860/predict",
    json={
        "latitude": 35.0,
        "longitude": -118.0,
        "time": "2025-10-22T14:00:00"
    }
)

result = response.json()
print(f"Occurrence: {result['occurrence_prediction']['will_occur']}")
print(f"Confidence: {result['occurrence_prediction']['confidence']:.2%}")
print(f"Risk Level: {result['risk_assessment']['risk_level']}")
```

### Response Example

```json
{
  "location": {
    "latitude": 35.0,
    "longitude": -118.0,
    "time": "2025-10-22T14:00:00"
  },
  "occurrence_prediction": {
    "will_occur": 1,
    "confidence": 0.8723
  },
  "severity_prediction": {
    "severity_class": 0,
    "confidence": 0.7156
  },
  "risk_assessment": {
    "risk_level": "MODERATE",
    "recommendation": "Stay alert and prepare emergency supplies"
  },
  "data_quality": {
    "earthquakes_analyzed": {
      "last_1_day": 3,
      "last_7_days": 18,
      "last_30_days": 67,
      "last_90_days": 203
    },
    "boundary_type": 2,
    "crust_type": 1,
    "elevation_m": 245.3
  }
}
```

## 📚 API Documentation

### Endpoints

#### `POST /predict`

Predict earthquake occurrence and severity.

**Request Body:**
```json
{
  "latitude": float,    // -90 to 90
  "longitude": float,   // -180 to 180
  "time": string        // ISO 8601 format
}
```

**Response:** See example above

#### `GET /health`

Check API health and model status.

**Response:**
```json
{
  "status": "healthy",
  "models": {
    "occurrence_transformer": true,
    "occurrence_model": true,
    "severity_transformer": true,
    "severity_model": true
  },
  "timestamp": "2025-10-22T14:30:00"
}
```

#### `GET /`

Root endpoint for basic status check.

### Interactive Documentation

Access interactive API docs at:
- Swagger UI: `https://vikctor-earthquake-managment-model.hf.space/docs`
- ReDoc: `https://vikctor-earthquake-managment-model.hf.space/redoc`

## 📁 Project Structure

```
earthquake-prediction/
├── app.py                          # FastAPI application
├── earthquake_dataset.py           # Data processing pipeline
├── earthquake_model.py             # Model training scripts
├── Dockerfile                      # Container configuration
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── models/
│   ├── occurence_model.cbm        # Occurrence prediction model
│   ├── severity_model.cbm         # Severity classification model
│   ├── occurence_transformer.joblib
│   └── severity_transformer.joblib
│
├── tectonicplates-master/
│   └── PB2002_steps.shp           # Tectonic boundary data
│      
└── data/
    ├── stage_1_earthquake_dataset.csv
    ├── stage_2_earthquake_dataset.csv
    ├── stage_3_earthquake_dataset.csv
    └── stage_4_earthquake_dataset.csv
```

## 🔧 Technical Details

### Machine Learning Pipeline

1. **Data Preprocessing**
   - Skewness reduction using PowerTransformer (Yeo-Johnson)
   - Cyclic encoding for temporal features
   - Label encoding for categorical variables
   - StandardScaler for feature normalization

2. **Class Imbalance Handling**
   - SMOTETomek for combined over/under-sampling
   - Class weights for tree-based models
   - Stratified train-test split

3. **Model Selection**
   - Compared: Random Forest, XGBoost, CatBoost, Isolation Forest
   - Selected: CatBoost for superior performance on imbalanced data
   - Tuned using RandomizedSearchCV with F1-macro scoring

4. **Experiment Tracking**
   - MLflow integration for model versioning
   - Comprehensive metric logging (accuracy, precision, recall, F1, ROC-AUC)
   - Artifact storage for reproducibility

### Geospatial Processing

```python
# Distance calculation using Haversine formula
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c
```

### Energy Calculation

Seismic energy computed using Gutenberg-Richter relation:

```python
E = 10^(1.5 * M + 4.8)  # Joules
```

Where M is the earthquake magnitude.

## 🔮 Future Enhancements

### Short-term
- [ ] Add support for custom time ranges (beyond 90 days)
- [ ] Implement caching for USGS API responses
- [ ] Create web-based visualization dashboard
- [ ] Add batch prediction endpoint

### Medium-term
- [ ] Integrate additional data sources (EMSC, ISC)
- [ ] Implement real-time alert system
- [ ] Add confidence intervals for predictions
- [ ] Support for aftershock sequence modeling

### Long-term
- [ ] Deep learning models (LSTM, Transformers) for temporal patterns
- [ ] Multi-region simultaneous predictions
- [ ] Mobile application for end-users
- [ ] Integration with early warning systems

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **USGS** for providing real-time earthquake data
- **PB2002** for tectonic plate boundary dataset
- **Open-Elevation API** for terrain elevation data
- **CatBoost** team for the excellent ML library
- **FastAPI** for the modern web framework

## 📧 Contact

For questions or collaboration opportunities:

- **Email**: wearevr2005@gmail.com
- **GitHub**: [VIJAYARAGUL362](https://github.com/VIJAYARAGUL362)
- **Project Link**: [https://github.com/yourusername/Earthquake-Disaster-Management](https://github.com/VIJAYARAGUL362/Earthquake-Disaster-Management)

## 📖 Citations

If you use this work in your research, please cite:

```bibtex
@software{earthquake_prediction_2025,
  author = {S.vijayaragul},
  title = {Earthquake Prediction and Severity Assessment System},
  year = {2025},
  url = {https://github.com/VIJAYARAGUL362/Earthquake-Disaster-Management}
}
```

---

**⚠️ Disclaimer**: This system is designed for research and educational purposes. Earthquake prediction remains a challenging scientific problem. Always follow official guidance from geological and emergency management authorities for real-world safety decisions.

**Built with ❤️ using Python, FastAPI, and CatBoost**
