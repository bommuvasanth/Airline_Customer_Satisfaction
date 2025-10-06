#!/usr/bin/env python3
"""
MongoDB utility module for Airline Customer Satisfaction System
Provides centralized database operations and connection management
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class MongoDBManager:
    """
    MongoDB connection and operations manager
    Provides centralized database operations for the airline satisfaction system
    """
    
    def __init__(self, 
                 mongodb_uri: str = None, 
                 database_name: str = None, 
                 collection_name: str = None):
        """
        Initialize MongoDB manager with configuration
        
        Args:
            mongodb_uri: MongoDB connection URI
            database_name: Database name
            collection_name: Collection name for predictions
        """
        # Load configuration from environment or use defaults
        self.mongodb_uri = mongodb_uri or os.getenv("MONGODB_URL", "mongodb://localhost:27017")
        self.database_name = database_name or os.getenv("MONGODB_DB_NAME", "airline_satisfaction")
        self.collection_name = collection_name or os.getenv("MONGODB_COLLECTION", "predictions")
        
        # Optional authentication
        self.username = os.getenv("MONGODB_USERNAME")
        self.password = os.getenv("MONGODB_PASSWORD")
        self.auth_source = os.getenv("MONGODB_AUTH_SOURCE", "admin")
        
        # Connection objects
        self.client = None
        self.database = None
        self.collection = None
        self._is_connected = False
    
    def connect(self) -> bool:
        """
        Establish connection to MongoDB
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Create connection options
            connection_options = {
                'serverSelectionTimeoutMS': 5000,  # 5 second timeout
                'connectTimeoutMS': 10000,         # 10 second connection timeout
                'socketTimeoutMS': 30000,          # 30 second socket timeout
            }
            
            # Add authentication if credentials are provided
            if self.username and self.password:
                self.mongodb_uri = self.mongodb_uri.replace(
                    'mongodb://', 
                    f'mongodb://{self.username}:{self.password}@'
                )
                connection_options['authSource'] = self.auth_source
            
            # Connect to MongoDB
            self.client = MongoClient(self.mongodb_uri, **connection_options)
            
            # Test the connection
            self.client.server_info()
            
            # Initialize database and collection
            self.database = self.client[self.database_name]
            self.collection = self.database[self.collection_name]
            
            self._is_connected = True
            logger.info(f"Successfully connected to MongoDB at {self.mongodb_uri}")
            return True
            
        except ServerSelectionTimeoutError as e:
            logger.error(f"MongoDB connection timeout: {str(e)}")
            self._is_connected = False
            return False
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {str(e)}")
            self._is_connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {str(e)}")
            self._is_connected = False
            return False
    
    def disconnect(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self._is_connected = False
            logger.info("MongoDB connection closed")
    
    def is_connected(self) -> bool:
        """Check if MongoDB is connected"""
        return self._is_connected
    
    def store_prediction(self, input_data: Dict[str, Any], 
                        predicted_label: str, 
                        prediction_probability: float,
                        additional_data: Dict[str, Any] = None) -> Optional[str]:
        """
        Store prediction data to MongoDB
        
        Args:
            input_data: Dictionary containing passenger input data
            predicted_label: Prediction result ('Satisfied' or 'Not Satisfied')
            prediction_probability: Confidence score of prediction
            additional_data: Optional additional data to store
        
        Returns:
            str: Inserted document ID if successful, None if failed
        """
        try:
            if not self._is_connected:
                if not self.connect():
                    return None
            
            # Create document with all required fields
            document = {
                "timestamp": datetime.now(),
                # Passenger input fields
                "gender": input_data.get('Gender'),
                "customer_type": input_data.get('Customer Type'),
                "age": input_data.get('Age'),
                "type_of_travel": input_data.get('Type of Travel'),
                "class": input_data.get('Class'),
                "flight_distance": input_data.get('Flight Distance'),
                "departure_delay_in_minutes": input_data.get('Departure Delay in Minutes'),
                "arrival_delay_in_minutes": input_data.get('Arrival Delay in Minutes'),
                "inflight_wifi_service": input_data.get('Inflight wifi service'),
                "departure_arrival_time_convenient": input_data.get('Departure/Arrival time convenient'),
                "ease_of_online_booking": input_data.get('Ease of Online booking'),
                "gate_location": input_data.get('Gate location'),
                "food_and_drink": input_data.get('Food and drink'),
                "online_boarding": input_data.get('Online boarding'),
                "seat_comfort": input_data.get('Seat comfort'),
                "inflight_entertainment": input_data.get('Inflight entertainment'),
                "on_board_service": input_data.get('On-board service'),
                "leg_room_service": input_data.get('Leg room service'),
                "baggage_handling": input_data.get('Baggage handling'),
                "checkin_service": input_data.get('Checkin service'),
                "inflight_service": input_data.get('Inflight service'),
                "cleanliness": input_data.get('Cleanliness'),
                # Prediction results
                "predicted_label": predicted_label,
                "prediction_probability": prediction_probability
            }
            
            # Add additional data if provided
            if additional_data:
                document.update(additional_data)
            
            # Insert document
            result = self.collection.insert_one(document)
            logger.info(f"Prediction stored with ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Failed to store prediction: {e}")
            return None
    
    def get_recent_predictions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve recent predictions from database
        
        Args:
            limit: Maximum number of predictions to retrieve
        
        Returns:
            List of prediction documents
        """
        try:
            if not self._is_connected:
                if not self.connect():
                    return []
            
            cursor = self.collection.find().sort("timestamp", -1).limit(limit)
            predictions = []
            
            for doc in cursor:
                doc['_id'] = str(doc['_id'])  # Convert ObjectId to string
                predictions.append(doc)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to retrieve predictions: {e}")
            return []
    
    def get_prediction_count(self) -> int:
        """
        Get total number of predictions stored
        
        Returns:
            int: Total count of predictions
        """
        try:
            if not self._is_connected:
                if not self.connect():
                    return 0
            
            return self.collection.count_documents({})
            
        except Exception as e:
            logger.error(f"Failed to count predictions: {e}")
            return 0
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """
        Get analytics summary from stored predictions
        
        Returns:
            Dictionary containing analytics data
        """
        try:
            if not self._is_connected:
                if not self.connect():
                    return {}
            
            # Total predictions
            total_predictions = self.collection.count_documents({})
            
            if total_predictions == 0:
                return {
                    "total_predictions": 0,
                    "satisfaction_rate": 0,
                    "average_probability": 0,
                    "travel_type_breakdown": []
                }
            
            # Satisfaction rate
            satisfied_count = self.collection.count_documents({"predicted_label": "Satisfied"})
            satisfaction_rate = (satisfied_count / total_predictions * 100)
            
            # Average prediction probability
            pipeline = [
                {"$group": {
                    "_id": None,
                    "avg_probability": {"$avg": "$prediction_probability"}
                }}
            ]
            
            avg_stats = list(self.collection.aggregate(pipeline))
            avg_probability = avg_stats[0]["avg_probability"] if avg_stats else 0
            
            # Travel type breakdown
            travel_type_pipeline = [
                {"$group": {
                    "_id": "$type_of_travel",
                    "count": {"$sum": 1},
                    "satisfaction_rate": {
                        "$avg": {"$cond": [{"$eq": ["$predicted_label", "Satisfied"]}, 1, 0]}
                    }
                }}
            ]
            
            travel_breakdown = list(self.collection.aggregate(travel_type_pipeline))
            
            return {
                "total_predictions": total_predictions,
                "satisfaction_rate": round(satisfaction_rate, 2),
                "average_probability": round(avg_probability, 3),
                "travel_type_breakdown": travel_breakdown
            }
            
        except Exception as e:
            logger.error(f"Failed to get analytics: {e}")
            return {}
    
    def delete_test_data(self) -> int:
        """
        Delete test data from collection
        
        Returns:
            int: Number of documents deleted
        """
        try:
            if not self._is_connected:
                if not self.connect():
                    return 0
            
            result = self.collection.delete_many({"test": True})
            logger.info(f"Deleted {result.deleted_count} test documents")
            return result.deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete test data: {e}")
            return 0
    
    def create_indexes(self):
        """Create database indexes for better performance"""
        try:
            if not self._is_connected:
                if not self.connect():
                    return False
            
            # Create index on timestamp for faster sorting
            self.collection.create_index("timestamp")
            
            # Create index on prediction fields for analytics
            self.collection.create_index("predicted_label")
            self.collection.create_index("type_of_travel")
            
            logger.info("Database indexes created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            return False


# Global MongoDB manager instance
_mongodb_manager = None

def get_mongodb_manager() -> MongoDBManager:
    """
    Get global MongoDB manager instance (singleton pattern)
    
    Returns:
        MongoDBManager: Global MongoDB manager instance
    """
    global _mongodb_manager
    if _mongodb_manager is None:
        _mongodb_manager = MongoDBManager()
    return _mongodb_manager

def store_prediction_to_mongodb(input_data: Dict[str, Any], 
                               predicted_label: str, 
                               prediction_probability: float,
                               additional_data: Dict[str, Any] = None) -> bool:
    """
    Convenience function to store prediction using global manager
    
    Args:
        input_data: Dictionary containing passenger input data
        predicted_label: Prediction result
        prediction_probability: Confidence score
        additional_data: Optional additional data
    
    Returns:
        bool: True if successful, False otherwise
    """
    manager = get_mongodb_manager()
    result = manager.store_prediction(input_data, predicted_label, prediction_probability, additional_data)
    return result is not None

def test_mongodb_connection() -> bool:
    """
    Test MongoDB connection
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    manager = get_mongodb_manager()
    return manager.connect()

# Context manager for MongoDB operations
class MongoDBContext:
    """Context manager for MongoDB operations"""
    
    def __init__(self, mongodb_uri: str = None, database_name: str = None, collection_name: str = None):
        self.manager = MongoDBManager(mongodb_uri, database_name, collection_name)
    
    def __enter__(self) -> MongoDBManager:
        self.manager.connect()
        return self.manager
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.manager.disconnect()