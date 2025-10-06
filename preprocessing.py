"""
Preprocessing utilities for airline satisfaction prediction
"""
import pandas as pd
import numpy as np
from typing import Dict, Any

def preprocess_passenger_data(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocess passenger input data for model prediction
    
    Args:
        data: Dictionary containing passenger information
        
    Returns:
        pd.DataFrame: Preprocessed data ready for model
    """
    # Create DataFrame with expected column names
    df = pd.DataFrame([{
        'Gender': data['gender'],
        'Customer Type': data['customer_type'],
        'Age': data['age'],
        'Type of Travel': data['type_of_travel'],
        'Class': data['travel_class'],
        'Flight Distance': data['flight_distance'],
        'Departure Delay in Minutes': data['departure_delay_in_minutes'],
        'Arrival Delay in Minutes': data['arrival_delay_in_minutes'],
        'Inflight wifi service': data['inflight_wifi_service'],
        'Departure/Arrival time convenient': data['departure_arrival_time_convenient'],
        'Ease of Online booking': data['ease_of_online_booking'],
        'Gate location': data['gate_location'],
        'Food and drink': data['food_and_drink'],
        'Online boarding': data['online_boarding'],
        'Seat comfort': data['seat_comfort'],
        'Inflight entertainment': data['inflight_entertainment'],
        'On-board service': data['on_board_service'],
        'Leg room service': data['leg_room_service'],
        'Baggage handling': data['baggage_handling'],
        'Checkin service': data['checkin_service'],
        'Inflight service': data['inflight_service'],
        'Cleanliness': data['cleanliness']
    }])
    
    return df

def calculate_service_metrics(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate various service quality metrics
    
    Args:
        data: Dictionary containing passenger service ratings
        
    Returns:
        Dict containing calculated metrics
    """
    service_ratings = [
        data.get('inflight_wifi_service', 0),
        data.get('departure_arrival_time_convenient', 1),
        data.get('ease_of_online_booking', 1),
        data.get('gate_location', 1),
        data.get('food_and_drink', 1),
        data.get('online_boarding', 1),
        data.get('seat_comfort', 1),
        data.get('inflight_entertainment', 1),
        data.get('on_board_service', 1),
        data.get('leg_room_service', 1),
        data.get('baggage_handling', 1),
        data.get('checkin_service', 1),
        data.get('inflight_service', 1),
        data.get('cleanliness', 1)
    ]
    
    # Filter out zero ratings (WiFi can be 0)
    non_zero_ratings = [r for r in service_ratings if r > 0]
    
    return {
        'average_rating': sum(non_zero_ratings) / len(non_zero_ratings) if non_zero_ratings else 0,
        'high_quality_count': sum(1 for r in non_zero_ratings if r >= 4),
        'high_quality_ratio': sum(1 for r in non_zero_ratings if r >= 4) / len(non_zero_ratings) if non_zero_ratings else 0,
        'key_services_avg': (data.get('inflight_wifi_service', 0) + data.get('online_boarding', 1)) / 2,
        'wifi_service': data.get('inflight_wifi_service', 0),
        'online_boarding': data.get('online_boarding', 1)
    }