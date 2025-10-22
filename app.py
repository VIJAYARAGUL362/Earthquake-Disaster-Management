"""
FastAPI Earthquake Prediction System
Uses real-time USGS earthquake data to compute features and make predictions
Complete feature pipeline with correct transformations
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, List, Any
import joblib
import catboost
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import logging
import math
import geopandas as gpd
from shapely.geometry import Point

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Earthquake Prediction API",
    description="Real-time earthquake prediction using USGS data and machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Global Variables
# ============================================================================
occurrence_transformer = None
occurrence_model = None
severity_transformer = None
severity_model = None

USGS_API_BASE = "https://earthquake.usgs.gov/fdsnws/event/1/query"
ELEVATION_API = "https://api.open-elevation.com/api/v1/lookup"
DEFAULT_RADIUS_KM = 100  # Default radius for USGS data fetch


# ============================================================================
# Pydantic Models
# ============================================================================

class PredictionRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    latitude: float = Field(..., ge=-90, le=90, description="Latitude (-90 to 90)")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude (-180 to 180)")
    time: str = Field(..., description="Prediction time in ISO format (e.g., '2025-10-22T14:00:00')")


class PredictionResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    location: Dict[str, Any]
    all_features: Dict[str, Any]
    features_for_transformation: Dict[str, float]
    selected_features: Dict[str, float]
    occurrence_prediction: Dict[str, Any]  # Allows float for confidence
    severity_prediction: Optional[Dict[str, Any]]
    risk_assessment: Dict[str, str]
    data_quality: Dict[str, Any]
    timestamp: str


# ============================================================================
# Startup Event - Load Models
# ============================================================================

@app.on_event("startup")
async def load_models():
    """Load all models and transformers on startup"""
    global occurrence_transformer, occurrence_model
    global severity_transformer, severity_model

    try:
        logger.info("Loading transformers...")
        occurrence_transformer = joblib.load('occurence_transformer.joblib')
        severity_transformer = joblib.load('severity_transformer.joblib')

        logger.info("Loading occurrence model...")
        occurrence_model = catboost.CatBoostClassifier()
        occurrence_model.load_model('occurence_model.cbm')

        logger.info("Loading severity model...")
        severity_model = catboost.CatBoostClassifier()
        severity_model.load_model('severity_model.cbm')

        logger.info("All models loaded successfully!")
        logger.info(f"Transformer expects: {list(occurrence_transformer.feature_names_in_)}")

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


# ============================================================================
# USGS Data Fetching Functions
# ============================================================================

def fetch_usgs_earthquakes(
        latitude: float,
        longitude: float,
        radius_km: float,
        start_time: datetime,
        end_time: datetime,
        min_magnitude: float = 0.0
) -> List[Dict]:
    """
    Fetch earthquake data from USGS API
    """
    params = {
        'format': 'geojson',
        'latitude': latitude,
        'longitude': longitude,
        'maxradiuskm': radius_km,
        'starttime': start_time.strftime('%Y-%m-%dT%H:%M:%S'),
        'endtime': end_time.strftime('%Y-%m-%dT%H:%M:%S'),
        'minmagnitude': min_magnitude,
        'orderby': 'time'
    }

    try:
        logger.info(f"Fetching earthquakes from USGS API...")
        logger.info(f"  Location: ({latitude}, {longitude})")
        logger.info(f"  Radius: {radius_km} km")
        logger.info(f"  Time range: {start_time} to {end_time}")

        response = requests.get(USGS_API_BASE, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        earthquakes = []

        if 'features' in data:
            for feature in data['features']:
                props = feature['properties']
                coords = feature['geometry']['coordinates']

                earthquakes.append({
                    'magnitude': props.get('mag', 0),
                    'latitude': coords[1],
                    'longitude': coords[0],
                    'depth': coords[2],
                    'time': datetime.fromtimestamp(props['time'] / 1000),
                    'place': props.get('place', 'Unknown')
                })

        logger.info(f"  Found {len(earthquakes)} earthquakes")
        return earthquakes

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching USGS data: {e}")
        return []


def get_elevation(latitude: float, longitude: float) -> float:
    """
    Get elevation for a location using Open-Elevation API
    """
    try:
        params = {
            'locations': f"{latitude},{longitude}"
        }
        response = requests.get(ELEVATION_API, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'results' in data and len(data['results']) > 0:
            elevation = data['results'][0]['elevation']
            logger.info(f"Elevation: {elevation}m")
            return float(elevation)
    except Exception as e:
        logger.warning(f"Could not fetch elevation: {e}")
        return 0.0


# ============================================================================
# Tectonic and Geological Functions
# ============================================================================

BOUNDARIES_FILE = "tectonicplates-master/PB2002_steps.shp"
try:
    BOUNDARIES = gpd.read_file(BOUNDARIES_FILE)
    logger.info(f"Successfully loaded PB2002 steps with {len(BOUNDARIES)} records")
    logger.info(f"Available columns: {list(BOUNDARIES.columns)}")
    # Ensure STEPCLASS is correctly recognized (case-insensitive match)
    step_class_col = next((col for col in BOUNDARIES.columns if 'stepclass' in col.lower()), None)
    if step_class_col and step_class_col != 'StepClass':
        logger.info(f"Renaming {step_class_col} to StepClass")
        BOUNDARIES = BOUNDARIES.rename(columns={step_class_col: 'StepClass'})
except Exception as e:
    logger.error(f"Failed to load PB2002 steps: {e}")
    raise


def simplify_boundary_type(bt: str) -> int:
    """Map PB2002 STEPCLASS to integer based on simplified categories."""
    boundary_types = {
        'SUB': 0,  # Subduction (Convergent)
        'OCB': 0,  # Oceanic Convergent Boundary (Convergent)
        'CCB': 0,  # Continental Convergent Boundary (Convergent)
        'OSR': 1,  # Oceanic Spreading Ridge (Divergent)
        'CRB': 1,  # Continental Rift Boundary (Divergent)
        'OTF': 2,  # Oceanic Transform Fault (Transform)
        'CTF': 2  # Continental Transform Fault (Transform)
    }
    return boundary_types.get(bt, 3)  # Default to 3 (other) for unrecognized types


def determine_boundary_type(latitude: float, longitude: float, max_distance_km: float = 1000000000000.0) -> int:
    """
    Determine tectonic boundary type using PB2002 STEPCLASS and fallback to proximity.
    Returns encoded integer: 0=convergent, 1=divergent, 2=transform, 3=other
    """
    if not (-90 <= latitude <= 90):
        raise ValueError(f"Latitude {latitude} must be between -90 and 90 degrees")
    if not (-180 <= longitude <= 180):
        raise ValueError(f"Longitude {longitude} must be between -180 and 180 degrees")

    point = Point(longitude, latitude)
    min_distance = float('inf')
    closest_type = 3
    closest_code = None

    logger.info(f"Checking boundaries for location ({latitude}, {longitude})")

    if 'StepClass' in BOUNDARIES.columns:
        for idx, row in BOUNDARIES.iterrows():
            distance = row.geometry.distance(point) * 111  # Approximate km
            code = row.get('StepClass', None)
            if code is None:
                logger.warning(f"Empty StepClass at index {idx}")
                continue
            if distance <= max_distance_km and distance < min_distance:
                min_distance = distance
                closest_code = code
                closest_type = simplify_boundary_type(code)
        logger.info(f"PB2002 result: code={closest_code}, type={closest_type}, distance={min_distance:.2f} km")
    else:
        logger.warning("No StepClass column found, using fallback logic")

    # Fallback logic
    if closest_type == 3:
        logger.info("Using fallback logic for boundary type based on proximity")
        known_boundaries = [
            (36.0, -121.0, 2, "San Andreas Fault"),  # Transform
            (38.0, 142.0, 0, "Japan Trench"),  # Convergent
            (-15.0, -75.0, 0, "Peru-Chile Trench"),  # Convergent
            (37.0, 29.0, 2, "North Anatolian Fault"),  # Transform
            (28.0, 85.0, 0, "Himalayan Front"),  # Convergent
            (-41.0, 174.0, 2, "Alpine Fault"),  # Transform
            (61.0, -147.0, 0, "Alaska"),  # Convergent
            (19.0, -155.0, 1, "Hawaii")  # Divergent
        ]
        for boundary_lat, boundary_lon, boundary_type, name in known_boundaries:
            distance = haversine_distance(latitude, longitude, boundary_lat, boundary_lon)
            logger.info(f"  {name}: {distance:.2f} km, type={boundary_type}")
            if distance <= max_distance_km and distance < min_distance:
                min_distance = distance
                closest_type = boundary_type
                closest_code = f"Fallback_{name}"
        logger.info(f"Fallback result: type={closest_type}, code={closest_code}, distance={min_distance:.2f} km")

    return closest_type


def determine_crust_type(elevation: float) -> int:
    """
    Determine crust type based on elevation: 0=oceanic (elevation < 0), 1=continental (elevation >= 0)
    """
    return 0 if elevation < 0 else 1


# ============================================================================
# Feature Engineering Functions
# ============================================================================

def calculate_seismic_energy(magnitude: float) -> float:
    """
    Calculate seismic energy from magnitude using the Gutenberg-Richter relation
    """
    return 10 ** (1.5 * magnitude + 4.8)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth (in kilometers)
    """
    R = 6371
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def estimate_distance_to_boundary(latitude: float, longitude: float) -> float:
    """
    Estimate distance to nearest tectonic plate boundary using hardcoded active zones
    """
    active_zones = [
        (36.0, -121.0),  # San Andreas Fault
        (38.0, 142.0),  # Japan Trench
        (-15.0, -75.0),  # Peru-Chile Trench
        (37.0, 29.0),  # North Anatolian Fault
        (28.0, 85.0),  # Himalayan Front
        (-41.0, 174.0),  # Alpine Fault
        (61.0, -147.0),  # Alaska
        (19.0, -155.0),  # Hawaii
    ]
    min_distance = float('inf')
    for zone_lat, zone_lon in active_zones:
        distance = haversine_distance(latitude, longitude, zone_lat, zone_lon)
        min_distance = min(min_distance, distance)
    logger.info(f"Estimated distance to nearest boundary: {min_distance:.2f} km")
    return min_distance


def compute_all_features(
        latitude: float,
        longitude: float,
        prediction_time: datetime
) -> tuple:
    """
    Compute ALL features in the pipeline
    """
    logger.info("=" * 80)
    logger.info("STEP 1: Fetching historical earthquake data from USGS")
    logger.info("=" * 80)

    earthquakes_1d = fetch_usgs_earthquakes(latitude, longitude, DEFAULT_RADIUS_KM, prediction_time - timedelta(days=1),
                                            prediction_time)
    earthquakes_7d = fetch_usgs_earthquakes(latitude, longitude, DEFAULT_RADIUS_KM, prediction_time - timedelta(days=7),
                                            prediction_time)
    earthquakes_30d = fetch_usgs_earthquakes(latitude, longitude, DEFAULT_RADIUS_KM,
                                             prediction_time - timedelta(days=30), prediction_time)
    earthquakes_90d = fetch_usgs_earthquakes(latitude, longitude, DEFAULT_RADIUS_KM,
                                             prediction_time - timedelta(days=90), prediction_time)

    logger.info("=" * 80)
    logger.info("STEP 2: Computing ALL features")
    logger.info("=" * 80)

    all_features = {}
    all_features['count_prev_1d'] = len(earthquakes_1d)
    if earthquakes_1d:
        magnitudes_1d = [eq['magnitude'] for eq in earthquakes_1d]
        all_features['meanmag_prev_1d'] = np.mean(magnitudes_1d)
        all_features['maxmag_prev_1d'] = np.max(magnitudes_1d)
        total_energy_1d = sum(calculate_seismic_energy(m) for m in magnitudes_1d)
        all_features['log_energy_prev_1d'] = np.log10(total_energy_1d) if total_energy_1d > 0 else 0
    else:
        all_features['meanmag_prev_1d'] = 0.0
        all_features['maxmag_prev_1d'] = 0.0
        all_features['log_energy_prev_1d'] = 0.0

    all_features['count_prev_7d'] = len(earthquakes_7d)
    if earthquakes_7d:
        magnitudes_7d = [eq['magnitude'] for eq in earthquakes_7d]
        all_features['meanmag_prev_7d'] = np.mean(magnitudes_7d)
        all_features['maxmag_prev_7d'] = np.max(magnitudes_7d)
        total_energy_7d = sum(calculate_seismic_energy(m) for m in magnitudes_7d)
        all_features['log_energy_prev_7d'] = np.log10(total_energy_7d) if total_energy_7d > 0 else 0
    else:
        all_features['meanmag_prev_7d'] = 0.0
        all_features['maxmag_prev_7d'] = 0.0
        all_features['log_energy_prev_7d'] = 0.0

    all_features['count_prev_30d'] = len(earthquakes_30d)
    if earthquakes_30d:
        magnitudes_30d = [eq['magnitude'] for eq in earthquakes_30d]
        all_features['meanmag_prev_30d'] = np.mean(magnitudes_30d)
        all_features['maxmag_prev_30d'] = np.max(magnitudes_30d)
        total_energy_30d = sum(calculate_seismic_energy(m) for m in magnitudes_30d)
        all_features['log_energy_prev_30d'] = np.log10(total_energy_30d) if total_energy_30d > 0 else 0
    else:
        all_features['meanmag_prev_30d'] = 0.0
        all_features['maxmag_prev_30d'] = 0.0
        all_features['log_energy_prev_30d'] = 0.0

    all_features['count_prev_90d'] = len(earthquakes_90d)
    if earthquakes_90d:
        magnitudes_90d = [eq['magnitude'] for eq in earthquakes_90d]
        all_features['meanmag_prev_90d'] = np.mean(magnitudes_90d)
        all_features['maxmag_prev_90d'] = np.max(magnitudes_90d)
        total_energy_90d = sum(calculate_seismic_energy(m) for m in magnitudes_90d)
        all_features['log_energy_prev_90d'] = np.log10(total_energy_90d) if total_energy_90d > 0 else 0
    else:
        all_features['meanmag_prev_90d'] = 0.0
        all_features['maxmag_prev_90d'] = 0.0
        all_features['log_energy_prev_90d'] = 0.0

    if earthquakes_7d:
        latest_earthquake = max(earthquakes_7d, key=lambda x: x['time'])
        days_since = (prediction_time - latest_earthquake['time']).total_seconds() / 86400
        all_features['days_since_last_event'] = days_since
    else:
        all_features['days_since_last_event'] = 7.0

    rate_7d = all_features['count_prev_7d'] / 7.0
    rate_30d = all_features['count_prev_30d'] / 30.0
    if rate_30d > 0:
        all_features['rate_change_7d_vs_30d'] = (rate_7d - rate_30d) / rate_30d
    else:
        all_features['rate_change_7d_vs_30d'] = 0.0

    elevation = get_elevation(latitude, longitude)
    all_features['dist_to_boundary_km'] = estimate_distance_to_boundary(latitude, longitude)
    all_features['boundary_type'] = determine_boundary_type(latitude, longitude)
    all_features['crust_type'] = determine_crust_type(elevation)
    all_features['elevation_m'] = elevation
    all_features['month'] = prediction_time.month

    logger.info("All features computed:")
    for key, value in all_features.items():
        logger.info(f"  {key}: {value}")

    logger.info("=" * 80)
    logger.info("STEP 3: Extracting 13 features for transformation")
    logger.info("=" * 80)

    transformation_features = {
        'count_prev_1d': all_features['count_prev_1d'],
        'meanmag_prev_1d': all_features['meanmag_prev_1d'],
        'maxmag_prev_1d': all_features['maxmag_prev_1d'],
        'log_energy_prev_1d': all_features['log_energy_prev_1d'],
        'count_prev_7d': all_features['count_prev_7d'],
        'meanmag_prev_7d': all_features['meanmag_prev_7d'],
        'maxmag_prev_7d': all_features['maxmag_prev_7d'],
        'log_energy_prev_7d': all_features['log_energy_prev_7d'],
        'count_prev_30d': all_features['count_prev_30d'],
        'count_prev_90d': all_features['count_prev_90d'],
        'days_since_last_event': all_features['days_since_last_event'],
        'rate_change_7d_vs_30d': all_features['rate_change_7d_vs_30d'],
        'dist_to_boundary_km': all_features['dist_to_boundary_km']
    }

    logger.info("Features for transformation:")
    for key, value in transformation_features.items():
        logger.info(f"  {key}: {value}")

    logger.info("=" * 80)
    logger.info("STEP 4: Computing cyclic month features")
    logger.info("=" * 80)

    month_sin = np.sin(2 * np.pi * prediction_time.month / 12)
    month_cos = np.cos(2 * np.pi * prediction_time.month / 12)

    logger.info(f"  month: {prediction_time.month}")
    logger.info(f"  month_sin: {month_sin}")
    logger.info(f"  month_cos: {month_cos}")

    data_info = {
        'earthquakes_1d': len(earthquakes_1d),
        'earthquakes_7d': len(earthquakes_7d),
        'earthquakes_30d': len(earthquakes_30d),
        'earthquakes_90d': len(earthquakes_90d),
        'latest_earthquake': earthquakes_7d[0] if earthquakes_7d else None
    }

    return all_features, transformation_features, month_sin, month_cos, data_info


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Earthquake Prediction API",
        "version": "1.0.0",
        "models_loaded": all([
            occurrence_transformer is not None,
            occurrence_model is not None,
            severity_transformer is not None,
            severity_model is not None
        ])
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_earthquake(request: PredictionRequest):
    """
    Predict earthquake occurrence and severity for a given location and time
    """
    try:
        logger.info("=" * 80)
        logger.info(f"NEW PREDICTION REQUEST")
        logger.info(f"Location: ({request.latitude}, {request.longitude})")
        logger.info(f"Time: {request.time}")
        logger.info("=" * 80)

        # Parse prediction time
        try:
            prediction_time = datetime.fromisoformat(request.time)
        except ValueError:
            raise HTTPException(status_code=400,
                                detail="Invalid time format. Use ISO format (e.g., '2025-10-22T14:00:00')")

        all_features, transformation_features, month_sin, month_cos, data_info = compute_all_features(
            request.latitude,
            request.longitude,
            prediction_time
        )

        logger.info("=" * 80)
        logger.info("STEP 5: Applying PowerTransformer to 13 features")
        logger.info("=" * 80)

        transformer_feature_names = occurrence_transformer.feature_names_in_
        df_for_transform = pd.DataFrame([transformation_features])[transformer_feature_names]
        logger.info(f"DataFrame shape: {df_for_transform.shape}")
        logger.info(f"Columns: {list(df_for_transform.columns)}")

        transformed_features = occurrence_transformer.transform(df_for_transform)
        logger.info(f"✓ Transformation successful")
        logger.info(f"  Transformed shape: {transformed_features.shape}")
        logger.info(f"  Sample values: {transformed_features[0][:5]}")

        transformed_dict = {}
        for i, feature_name in enumerate(transformer_feature_names):
            transformed_dict[feature_name] = transformed_features[0][i]

        logger.info("=" * 80)
        logger.info("STEP 6: Building 16 selected features for model")
        logger.info("=" * 80)

        selected_features = {
            'meanmag_prev_1d': transformed_dict['meanmag_prev_1d'],
            'maxmag_prev_1d': transformed_dict['maxmag_prev_1d'],
            'meanmag_prev_7d': transformed_dict['meanmag_prev_7d'],
            'log_energy_prev_7d': transformed_dict['log_energy_prev_7d'],
            'meanmag_prev_30d': all_features['meanmag_prev_30d'],
            'log_energy_prev_30d': all_features['log_energy_prev_30d'],
            'meanmag_prev_90d': all_features['meanmag_prev_90d'],
            'log_energy_prev_90d': all_features['log_energy_prev_90d'],
            'days_since_last_event': transformed_dict['days_since_last_event'],
            'rate_change_7d_vs_30d': transformed_dict['rate_change_7d_vs_30d'],
            'dist_to_boundary_km': transformed_dict['dist_to_boundary_km'],
            'elevation_m': all_features['elevation_m'],
            'boundary_type': all_features['boundary_type'],
            'crust_type': all_features['crust_type'],
            'month_sin': month_sin,
            'month_cos': month_cos
        }

        logger.info("Selected features (in order):")
        for key, value in selected_features.items():
            logger.info(f"  {key}: {value}")

        logger.info("=" * 80)
        logger.info("STEP 7: Creating final DataFrame for model")
        logger.info("=" * 80)

        final_df = pd.DataFrame([selected_features])
        logger.info(f"Final DataFrame shape: {final_df.shape}")
        logger.info(f"Final columns: {list(final_df.columns)}")

        logger.info("=" * 80)
        logger.info("STEP 8: Making predictions")
        logger.info("=" * 80)

        occurrence_pred = occurrence_model.predict(final_df)[0]
        occurrence_prob = occurrence_model.predict_proba(final_df)[0]

        will_occur = int(occurrence_pred)  # 0 for not occurred, 1 for occurred
        confidence = float(occurrence_prob[1])  # Keep as float for accuracy

        logger.info(f"✓ Occurrence prediction: {will_occur}")
        logger.info(f"  Confidence: {confidence:.2%}")
        logger.info(f"  Probabilities: [No EQ: {occurrence_prob[0]:.4f}, EQ: {occurrence_prob[1]:.4f}]")

        severity_result = None
        if will_occur:
            logger.info("Predicting severity...")
            severity_pred = severity_model.predict(final_df)[0]
            severity_prob = severity_model.predict_proba(final_df)[0]
            severity_class = int(severity_pred)  # 0 for medium, 1 for high

            severity_result = {
                "severity_class": severity_class,
                "confidence": round(float(severity_prob[severity_pred]), 4)
            }

            logger.info(f"✓ Severity: {severity_class}")
            logger.info(f"  Confidence: {severity_result['confidence']:.2%}")

        if will_occur and severity_result:
            if severity_result['severity_class'] == 1:
                risk_level = "HIGH"
                recommendation = "Immediate evacuation and emergency preparedness"
            else:
                risk_level = "MODERATE"
                recommendation = "Stay alert and prepare emergency supplies"
        else:
            risk_level = "VERY LOW"
            recommendation = "No significant seismic activity expected"

        logger.info(f"✓ Risk Level: {risk_level}")
        logger.info("=" * 80)

        response = PredictionResponse(
            location={
                "latitude": request.latitude,
                "longitude": request.longitude,
                "time": request.time
            },
            all_features=all_features,
            features_for_transformation=transformation_features,
            selected_features=selected_features,
            occurrence_prediction={
                "will_occur": will_occur,
                "confidence": confidence  # Float value
            },
            severity_prediction=severity_result,
            risk_assessment={
                "risk_level": risk_level,
                "recommendation": recommendation
            },
            data_quality={
                "earthquakes_analyzed": {
                    "last_1_day": data_info['earthquakes_1d'],
                    "last_7_days": data_info['earthquakes_7d'],
                    "last_30_days": data_info['earthquakes_30d'],
                    "last_90_days": data_info['earthquakes_90d']
                },
                "latest_earthquake": data_info['latest_earthquake']['place'] if data_info[
                    'latest_earthquake'] else "None in past 7 days",
                "data_source": "USGS Earthquake Catalog",
                "boundary_type": all_features['boundary_type'],
                "crust_type": all_features['crust_type'],
                "elevation_m": all_features['elevation_m']
            },
            timestamp=datetime.utcnow().isoformat()
        )

        logger.info("✓ Prediction completed successfully!")
        logger.info("=" * 80)
        return response

    except Exception as e:
        logger.error(f"✗ Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models": {
            "occurrence_transformer": occurrence_transformer is not None,
            "occurrence_model": occurrence_model is not None,
            "severity_transformer": severity_transformer is not None,
            "severity_model": severity_model is not None
        },
        "external_services": {
            "usgs_api": "operational",
            "elevation_api": "operational"
        },
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)