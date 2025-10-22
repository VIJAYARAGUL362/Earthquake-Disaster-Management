# app/main.py
import numpy as np
import pandas as pd
import requests
import joblib
from datetime import datetime, timedelta
from catboost import CatBoostClassifier
from geopy.distance import geodesic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional
import uvicorn
import logging
from functools import lru_cache
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Earthquake Prediction API", version="1.0.0")

# Global variables for models and transformers
occurrence_model = None
severity_model = None
occ_trans = None
sev_trans = None


# Configure requests session with retry logic
def get_session():
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=(500, 502, 504),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


session = get_session()


# =====================================================
#               HELPER FUNCTIONS
# =====================================================

def fetch_recent_quakes(lat, lon, days, radius_km=300):
    """Fetch recent earthquakes from USGS API within a given radius."""
    try:
        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        params = {
            "format": "geojson",
            "starttime": start,
            "latitude": lat,
            "longitude": lon,
            "maxradiuskm": radius_km,
            "minmagnitude": 2.5
        }
        response = session.get(
            "https://earthquake.usgs.gov/fdsnws/event/1/query",
            params=params,
            timeout=15
        )
        response.raise_for_status()

        data = response.json()
        events = []
        for f in data.get("features", []):
            props = f.get("properties", {})
            if 'mag' in props and props['mag'] is not None:
                time_ms = props.get('time')
                if time_ms:
                    time = datetime.utcfromtimestamp(time_ms / 1000.0)
                    events.append({'time': time, 'mag': props['mag']})

        return sorted(events, key=lambda x: x['time'], reverse=True)

    except requests.Timeout:
        logger.error(f"Timeout fetching earthquakes for ({lat}, {lon})")
        return []
    except requests.RequestException as e:
        logger.error(f"Error fetching earthquakes: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in fetch_recent_quakes: {e}")
        return []


def fetch_all_recent_quakes(lat, lon, radius_km=300):
    """Fetch earthquakes once for 90 days and filter for different windows."""
    try:
        all_events = fetch_recent_quakes(lat, lon, 90, radius_km)
        now = datetime.utcnow()

        events_1d = [e for e in all_events if (now - e['time']).days < 1]
        events_7d = [e for e in all_events if (now - e['time']).days < 7]
        events_30d = [e for e in all_events if (now - e['time']).days < 30]
        events_90d = all_events

        return events_1d, events_7d, events_30d, events_90d
    except Exception as e:
        logger.error(f"Error in fetch_all_recent_quakes: {e}")
        return [], [], [], []


def compute_rolling_features(events):
    """Compute statistical features from events."""
    if not events or len(events) == 0:
        return {"count": 0, "mean_mag": 0, "max_mag": 0, "log_energy": 0}

    mags = [e['mag'] for e in events]
    try:
        energy = [10 ** (1.5 * m + 4.8) for m in mags]
        total_energy = sum(energy)
        log_energy = np.log10(total_energy) if total_energy > 0 else 0
    except (ValueError, OverflowError):
        log_energy = 0

    return {
        "count": len(mags),
        "mean_mag": float(np.mean(mags)),
        "max_mag": float(np.max(mags)),
        "log_energy": float(log_energy)
    }


def get_elevation(lat, lon):
    """Fetch elevation from Open-Elevation API."""
    try:
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
        response = session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return float(data['results'][0]['elevation'])
    except (requests.RequestException, KeyError, IndexError, ValueError) as e:
        logger.warning(f"Error fetching elevation for ({lat}, {lon}): {e}")
        return 0.0


def simplify_boundary_type(code):
    """Simplify boundary code to category string."""
    convergent = ['SUB', 'CCB', 'CRB', 'OCB']
    divergent = ['OSR', 'OBR']
    transform = ['CTF', 'OTF']

    if code in convergent:
        return 'convergent'
    elif code in divergent:
        return 'divergent'
    elif code in transform:
        return 'transform'
    else:
        return 'Other'


@lru_cache(maxsize=1)
def get_tectonic_boundaries():
    """Fetch and cache tectonic boundaries data."""
    boundaries_url = "https://raw.githubusercontent.com/fraxen/tectonicplates/master/GeoJSON/PB2002_boundaries.json"
    try:
        response = session.get(boundaries_url, timeout=30)
        response.raise_for_status()
        return response.json().get('features', [])
    except requests.RequestException as e:
        logger.error(f"Error fetching tectonic boundaries: {e}")
        return []


def get_boundary_features(lat, lon):
    """Fetch tectonic boundary distance and simplified type."""
    boundaries = get_tectonic_boundaries()

    if not boundaries:
        logger.warning("No tectonic boundaries data available")
        return 5000.0, 'Other'

    min_dist = float('inf')
    nearest_code = None
    point = (lat, lon)

    try:
        for feature in boundaries:
            code = feature.get('properties', {}).get('Code', '')
            geom = feature.get('geometry', {})
            coords = []

            if geom.get('type') == 'LineString':
                coords = geom.get('coordinates', [])
            elif geom.get('type') == 'MultiLineString':
                for line in geom.get('coordinates', []):
                    coords.extend(line)

            for coord in coords:
                if len(coord) >= 2:
                    dist_point = (coord[1], coord[0])
                    dist = geodesic(point, dist_point).km
                    if dist < min_dist:
                        min_dist = dist
                        nearest_code = code
    except Exception as e:
        logger.error(f"Error calculating boundary distance: {e}")
        return 5000.0, 'Other'

    if min_dist == float('inf') or nearest_code is None:
        return 5000.0, 'Other'

    boundary_type_str = simplify_boundary_type(nearest_code)
    return float(min_dist), boundary_type_str


def get_static_features(lat, lon):
    """Get static spatial features."""
    dist_to_boundary_km, boundary_type_str = get_boundary_features(lat, lon)
    elevation_m = get_elevation(lat, lon)
    crust_str = "oceanic" if elevation_m <= 0 else "continental"
    return dist_to_boundary_km, boundary_type_str, crust_str, elevation_m


def extract_features(lat, lon, date):
    """Construct the full raw feature vector."""
    now = datetime.utcnow()

    # Fetch events once for all windows
    events_1d, events_7d, events_30d, events_90d = fetch_all_recent_quakes(lat, lon)

    # Days since last event - FIXED BUG
    if len(events_90d) > 0:
        days_since_last_event = (now - events_90d[0]['time']).days
        days_since_last_event = max(0, days_since_last_event)  # Ensure non-negative
    else:
        days_since_last_event = 90

    # Compute rolling features
    feats_1d = compute_rolling_features(events_1d)
    feats_7d = compute_rolling_features(events_7d)
    feats_30d = compute_rolling_features(events_30d)
    feats_90d = compute_rolling_features(events_90d)

    # Temporal features
    try:
        month = pd.to_datetime(date).month
    except Exception as e:
        logger.error(f"Error parsing date {date}: {e}")
        month = datetime.utcnow().month

    # Rate change with protection against division issues
    count_30d = feats_30d["count"] if feats_30d["count"] > 0 else 1
    rate_change = feats_7d["count"] / count_30d

    # Spatial features
    dist_to_boundary_km, boundary_type_str, crust_str, elevation_m = get_static_features(lat, lon)

    # Create DataFrame
    df = pd.DataFrame([{
        "count_prev_1d": feats_1d["count"],
        "meanmag_prev_1d": feats_1d["mean_mag"],
        "maxmag_prev_1d": feats_1d["max_mag"],
        "log_energy_prev_1d": feats_1d["log_energy"],
        "count_prev_7d": feats_7d["count"],
        "meanmag_prev_7d": feats_7d["mean_mag"],
        "maxmag_prev_7d": feats_7d["max_mag"],
        "log_energy_prev_7d": feats_7d["log_energy"],
        "count_prev_30d": feats_30d["count"],
        "meanmag_prev_30d": feats_30d["mean_mag"],
        "maxmag_prev_30d": feats_30d["max_mag"],
        "log_energy_prev_30d": feats_30d["log_energy"],
        "count_prev_90d": feats_90d["count"],
        "meanmag_prev_90d": feats_90d["mean_mag"],
        "maxmag_prev_90d": feats_90d["max_mag"],
        "log_energy_prev_90d": feats_90d["log_energy"],
        "days_since_last_event": days_since_last_event,
        "rate_change_7d_vs_30d": rate_change,
        "dist_to_boundary_km": dist_to_boundary_km,
        "boundary_type": boundary_type_str,
        "elevation_m": elevation_m,
        "crust_type": crust_str,
        "month": month
    }])

    return df


def encode_boundary_type(boundary_str):
    """Encode boundary type as integer."""
    mapping = {
        'convergent': 0,
        'divergent': 1,
        'transform': 2,
        'Other': 3
    }
    return mapping.get(boundary_str, 3)


def preprocess_features(X, transformer):
    """Apply preprocessing steps."""
    try:
        # Skewed features transformation
        skewed_cols = [
            'count_prev_1d', 'meanmag_prev_1d', 'maxmag_prev_1d', 'log_energy_prev_1d',
            'count_prev_7d', 'meanmag_prev_7d', 'maxmag_prev_7d', 'log_energy_prev_7d',
            'count_prev_30d', 'count_prev_90d', 'days_since_last_event', 'rate_change_7d_vs_30d',
            'dist_to_boundary_km'
        ]
        X[skewed_cols] = transformer.transform(X[skewed_cols])

        # Cyclic encoding for month
        month = X['month'].iloc[0]
        X['month_sin'] = np.sin(2 * np.pi * month / 12)
        X['month_cos'] = np.cos(2 * np.pi * month / 12)
        X = X.drop('month', axis=1)

        # Encode boundary_type
        boundary_str = X['boundary_type'].iloc[0]
        X['boundary_type'] = encode_boundary_type(boundary_str)

        # Encode crust_type
        crust_str = X['crust_type'].iloc[0]
        X['crust_type'] = 0 if crust_str == "oceanic" else 1

        # Select final features
        selected_features = [
            'meanmag_prev_1d', 'maxmag_prev_1d',
            'meanmag_prev_7d', 'log_energy_prev_7d',
            'meanmag_prev_30d', 'log_energy_prev_30d',
            'meanmag_prev_90d', 'log_energy_prev_90d',
            'days_since_last_event', 'rate_change_7d_vs_30d',
            'dist_to_boundary_km', 'elevation_m',
            'boundary_type', 'crust_type', 'month_sin', 'month_cos'
        ]
        X = X[selected_features]

        return X
    except Exception as e:
        logger.error(f"Error in preprocess_features: {e}")
        raise


def predict_occurrence(lat, lon, date, model, transformer):
    """Predict earthquake occurrence."""
    X = extract_features(lat, lon, date)
    X_processed = preprocess_features(X, transformer)
    prob = model.predict_proba(X_processed)[0, 1]
    pred = int(prob >= 0.5)
    return {"type": "occurrence", "prediction": pred, "probability": float(prob)}


def predict_severity(lat, lon, date, model, transformer):
    """Predict earthquake severity."""
    X = extract_features(lat, lon, date)
    X_processed = preprocess_features(X, transformer)
    prob = model.predict_proba(X_processed)[0, 1]
    pred = int(prob >= 0.5)
    return {"type": "severity", "prediction": pred, "probability": float(prob)}


# Pydantic model with validation
class PredictionRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Latitude between -90 and 90")
    lon: float = Field(..., ge=-180, le=180, description="Longitude between -180 and 180")
    date: str = Field(..., description="Date in YYYY-MM-DD format")

    @validator('date')
    def validate_date(cls, v):
        try:
            parsed_date = datetime.strptime(v, "%Y-%m-%d")
            # Check if date is reasonable (not too far in past or future)
            now = datetime.utcnow()
            if parsed_date > now + timedelta(days=365):
                raise ValueError("Date cannot be more than 1 year in the future")
            if parsed_date < now - timedelta(days=365 * 10):
                raise ValueError("Date cannot be more than 10 years in the past")
            return v
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD. Error: {str(e)}")


@app.on_event("startup")
async def load_models():
    """Load models on startup."""
    global occurrence_model, severity_model, occ_trans, sev_trans
    try:
        logger.info("Loading transformers...")
        occ_trans = joblib.load("occurence_transformer.joblib")
        sev_trans = joblib.load("severity_transformer.joblib")

        logger.info("Loading occurrence model...")
        occurrence_model = CatBoostClassifier()
        occurrence_model.load_model("occurence_model.cbm")

        logger.info("Loading severity model...")
        severity_model = CatBoostClassifier()
        severity_model.load_model("severity_model.cbm")

        logger.info("All models loaded successfully!")

        # Pre-cache tectonic boundaries
        logger.info("Pre-caching tectonic boundaries...")
        get_tectonic_boundaries()
        logger.info("Tectonic boundaries cached!")

    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        logger.error("Please ensure model files exist in the 'models' directory")
    except Exception as e:
        logger.error(f"Error loading models: {e}")


@app.post("/predict/occurrence", response_model=Dict[str, Any])
async def predict_occurrence_endpoint(request: PredictionRequest):
    """Predict earthquake occurrence probability."""
    if occurrence_model is None or occ_trans is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Please check server logs.")

    try:
        logger.info(f"Occurrence prediction request: lat={request.lat}, lon={request.lon}, date={request.date}")
        result = predict_occurrence(request.lat, request.lon, request.date, occurrence_model, occ_trans)
        logger.info(f"Occurrence prediction result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in occurrence prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/severity", response_model=Dict[str, Any])
async def predict_severity_endpoint(request: PredictionRequest):
    """Predict earthquake severity probability."""
    if severity_model is None or sev_trans is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Please check server logs.")

    try:
        logger.info(f"Severity prediction request: lat={request.lat}, lon={request.lon}, date={request.date}")
        result = predict_severity(request.lat, request.lon, request.date, severity_model, sev_trans)
        logger.info(f"Severity prediction result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in severity prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    models_loaded = all([
        occurrence_model is not None,
        severity_model is not None,
        occ_trans is not None,
        sev_trans is not None
    ])

    return {
        "status": "healthy" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "occurrence_model": occurrence_model is not None,
        "severity_model": severity_model is not None,
        "transformers": occ_trans is not None and sev_trans is not None
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Earthquake Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict_occurrence": "/predict/occurrence",
            "predict_severity": "/predict/severity"
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)