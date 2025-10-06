"""
FastAPI Backend for Airline Customer Satisfaction Prediction System
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from utils.preprocessing import preprocess_passenger_data, calculate_service_metrics
from utils.mongodb import get_mongodb_manager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and database
model = None
scaler = None
mongodb_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    startup_event()
    yield
    # Shutdown
    shutdown_event()

# Initialize FastAPI app
app = FastAPI(
    title="Airline Customer Satisfaction Prediction API",
    description="ML-powered API for predicting airline customer satisfaction with enhanced personal traveler logic",
    version="1.0.0",
    lifespan=lifespan
)

class PassengerInput(BaseModel):
    """Input model for passenger data"""
    # Customer Demographics
    gender: str = Field(..., description="Male or Female")
    customer_type: str = Field(..., description="Loyal Customer or disloyal Customer")
    age: int = Field(..., ge=7, le=85, description="Age between 7-85")
    type_of_travel: str = Field(..., description="Business travel or Personal Travel")
    travel_class: str = Field(..., description="Business, Eco Plus, or Eco")
    
    # Flight Information
    flight_distance: int = Field(..., ge=31, le=5000, description="Flight distance in miles")
    departure_delay_in_minutes: int = Field(..., ge=0, description="Departure delay in minutes")
    arrival_delay_in_minutes: float = Field(..., ge=0.0, description="Arrival delay in minutes")
    
    # Service Ratings (1-5 scale)
    inflight_wifi_service: int = Field(..., ge=0, le=5, description="WiFi service rating")
    departure_arrival_time_convenient: int = Field(..., ge=1, le=5, description="Time convenience rating")
    ease_of_online_booking: int = Field(..., ge=1, le=5, description="Online booking ease rating")
    gate_location: int = Field(..., ge=1, le=5, description="Gate location rating")
    food_and_drink: int = Field(..., ge=1, le=5, description="Food and drink rating")
    online_boarding: int = Field(..., ge=1, le=5, description="Online boarding rating")
    seat_comfort: int = Field(..., ge=1, le=5, description="Seat comfort rating")
    inflight_entertainment: int = Field(..., ge=1, le=5, description="Entertainment rating")
    on_board_service: int = Field(..., ge=1, le=5, description="Onboard service rating")
    leg_room_service: int = Field(..., ge=1, le=5, description="Legroom service rating")
    baggage_handling: int = Field(..., ge=1, le=5, description="Baggage handling rating")
    checkin_service: int = Field(..., ge=1, le=5, description="Check-in service rating")
    inflight_service: int = Field(..., ge=1, le=5, description="Inflight service rating")
    cleanliness: int = Field(..., ge=1, le=5, description="Cleanliness rating")

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    base_probability: float = Field(..., description="Original model probability")
    adjusted_probability: float = Field(..., description="Enhanced probability after adjustments")
    threshold: float = Field(..., description="Decision threshold used")
    prediction: str = Field(..., description="Satisfied or Dissatisfied")
    reasons: List[str] = Field(..., description="List of applied enhancement reasons")
    confidence_level: str = Field(..., description="High, Medium, or Low confidence")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    passenger_type: str = Field(..., description="Business travel or Personal Travel")

def startup_event():
    """Initialize model, preprocessor, and database connection"""
    global model, scaler, mongodb_manager
    
    try:
        # Load trained model
        logger.info("Loading Random Forest model...")
        if os.path.exists('rf_model.pkl'):
            model = joblib.load('rf_model.pkl')
        elif os.path.exists('models/model.joblib'):
            model = joblib.load('models/model.joblib')
        else:
            raise FileNotFoundError("Model file not found. Please ensure rf_model.pkl exists.")
        logger.info("Model loaded successfully")
        
        # Load scaler if available
        try:
            if os.path.exists('models/scaler.joblib'):
                scaler = joblib.load('models/scaler.joblib')
                logger.info("Scaler loaded successfully")
        except Exception as e:
            logger.warning(f"Scaler not loaded: {str(e)}")
            scaler = None
        
        # Initialize MongoDB manager
        logger.info("Initializing MongoDB connection...")
        mongodb_manager = get_mongodb_manager()
        if mongodb_manager.connect():
            logger.info("MongoDB connection established successfully")
        else:
            logger.warning("MongoDB connection failed - predictions will not be stored")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

def shutdown_event():
    """Close database connection"""
    global mongodb_manager
    if mongodb_manager:
        mongodb_manager.disconnect()
        logger.info("MongoDB connection closed")

def preprocess_input(passenger_data: PassengerInput) -> np.ndarray:
    """
    Preprocess passenger input for model prediction
    """
    try:
        # Convert to dictionary
        data_dict = passenger_data.dict()
        
        # Use preprocessing utility
        df = preprocess_passenger_data(data_dict)
        
        # Basic preprocessing - encode categorical variables
        categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
        df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
        
        # Ensure all expected columns are present
        expected_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
        
        if expected_features is not None:
            for col in expected_features:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            df_encoded = df_encoded[expected_features]
        
        # Apply scaler if available
        if scaler is not None:
            return scaler.transform(df_encoded.values)
        
        return df_encoded.values
        
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(e)}")

def apply_enhancement_logic(base_probability: float, passenger_data: PassengerInput) -> tuple:
    """
    Apply enhanced prediction logic for personal travelers
    
    Returns:
        tuple: (adjusted_probability, reasons_list)
    """
    reasons = []
    adjusted_probability = base_probability
    
    # Calculate service quality metrics using utility function
    data_dict = passenger_data.dict()
    metrics = calculate_service_metrics(data_dict)
    
    is_personal_travel = passenger_data.type_of_travel == "Personal Travel"
    no_delays = (passenger_data.departure_delay_in_minutes == 0 and 
                passenger_data.arrival_delay_in_minutes == 0)
    
    # Skip adjustments if already high confidence
    if base_probability >= 0.85:
        reasons.append("High base confidence - no adjustments needed")
        return adjusted_probability, reasons
    
    # Enhancement 1: High service quality for personal travelers
    if (is_personal_travel and 
        metrics['high_quality_ratio'] >= 0.7 and 
        metrics['key_services_avg'] >= 4.0):
        adjustment_factor = 1.0 + (0.15 * (1.0 - base_probability))
        adjusted_probability = base_probability * adjustment_factor
        reasons.append(f"Personal travel high-quality service boost (avg: {metrics['average_rating']:.1f}/5)")
    
    # Enhancement 2: Excellent key services
    elif (metrics['wifi_service'] >= 5 and metrics['online_boarding'] >= 5):
        adjustment_factor = 1.0 + (0.10 * (1.0 - base_probability))
        adjusted_probability = base_probability * adjustment_factor
        reasons.append("Excellent key services boost (WiFi & Online Boarding = 5)")
    
    # Enhancement 3: Good service + no delays
    elif (metrics['average_rating'] >= 4.0 and no_delays):
        adjustment_factor = 1.0 + (0.08 * (1.0 - base_probability))
        adjusted_probability = base_probability * adjustment_factor
        reasons.append(f"Good service + punctuality boost (avg: {metrics['average_rating']:.1f}/5, no delays)")
    
    # Clip probabilities to safe range [0.05, 0.95]
    adjusted_probability = max(0.05, min(0.95, adjusted_probability))
    
    # Model drift detection
    if base_probability < 0.1 and metrics['average_rating'] >= 4.5:
        reasons.append("⚠️ Model drift warning: Low prediction despite high service quality")
    
    return adjusted_probability, reasons

@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        db_status = "connected" if mongodb_manager and mongodb_manager.is_connected() else "disconnected"
        model_status = "loaded" if model is not None else "not loaded"
        
        return {
            "status": "ok", 
            "database": db_status, 
            "model": model_status,
            "mongodb_uri": mongodb_manager.mongodb_uri if mongodb_manager else "not configured"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/predict", response_model=PredictionResponse)
def predict_satisfaction(passenger_data: PassengerInput):
    """
    Predict customer satisfaction with enhanced logic
    """
    try:
        # Preprocess input data
        processed_data = preprocess_input(passenger_data)
        
        # Make base prediction
        base_probabilities = model.predict_proba(processed_data)[0]
        base_probability = float(base_probabilities[1])  # Probability of satisfaction
        
        # Apply enhancement logic
        adjusted_probability, reasons = apply_enhancement_logic(base_probability, passenger_data)
        
        # Determine final prediction with dynamic threshold
        threshold = 0.60 if passenger_data.type_of_travel == "Personal Travel" else 0.50
        prediction = "Satisfied" if adjusted_probability >= threshold else "Dissatisfied"
        
        # Calculate confidence level
        confidence = max(adjusted_probability, 1.0 - adjusted_probability)
        if confidence >= 0.8:
            confidence_level = "High"
        elif confidence >= 0.6:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        # Create response
        response = PredictionResponse(
            base_probability=base_probability,
            adjusted_probability=adjusted_probability,
            threshold=threshold,
            prediction=prediction,
            reasons=reasons if reasons else ["No enhancements applied"],
            confidence_level=confidence_level,
            timestamp=datetime.utcnow(),
            passenger_type=passenger_data.type_of_travel
        )
        
        # Calculate service metrics for storage
        data_dict = passenger_data.dict()
        metrics = calculate_service_metrics(data_dict)
        
        # Store prediction in MongoDB using the new module
        additional_data = {
            "base_probability": base_probability,
            "adjusted_probability": adjusted_probability,
            "threshold": threshold,
            "reasons": reasons if reasons else ["No enhancements applied"],
            "confidence_level": confidence_level,
            "passenger_type": passenger_data.type_of_travel,
            "service_quality_average": metrics['average_rating']
        }
        
        if mongodb_manager and mongodb_manager.is_connected():
            doc_id = mongodb_manager.store_prediction(
                input_data=data_dict,
                predicted_label=prediction,
                prediction_probability=adjusted_probability,
                additional_data=additional_data
            )
            if doc_id:
                logger.info(f"Prediction stored with ID: {doc_id}")
            else:
                logger.warning("Failed to store prediction in MongoDB")
        else:
            logger.warning("MongoDB not connected - prediction not stored")
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/analytics")
def get_analytics():
    """Get analytics summary from stored predictions"""
    try:
        if not mongodb_manager or not mongodb_manager.is_connected():
            raise HTTPException(status_code=503, detail="Database not available")
        
        analytics = mongodb_manager.get_analytics_summary()
        
        if not analytics:
            return {
                "total_predictions": 0,
                "overall_satisfaction_rate": 0,
                "average_probability": 0,
                "travel_type_breakdown": []
            }
        
        return {
            "total_predictions": analytics.get("total_predictions", 0),
            "overall_satisfaction_rate": analytics.get("satisfaction_rate", 0),
            "average_probability": analytics.get("average_probability", 0),
            "average_service_quality": 0,  # This field is not available in the new analytics
            "travel_type_breakdown": analytics.get("travel_type_breakdown", [])
        }
        
    except Exception as e:
        logger.error(f"Analytics error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")

@app.get("/predictions")
def get_recent_predictions(limit: int = 10):
    """Get recent predictions from database"""
    try:
        if not mongodb_manager or not mongodb_manager.is_connected():
            raise HTTPException(status_code=503, detail="Database not available")
        
        predictions = mongodb_manager.get_recent_predictions(limit)
        return {"predictions": predictions}
        
    except Exception as e:
        logger.error(f"Error fetching predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch predictions: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)