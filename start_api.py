#!/usr/bin/env python3
"""
Simple startup script for the FastAPI Airline Satisfaction Prediction API
"""

import uvicorn
import sys
import os

def main():
    """Start the FastAPI server"""
    print("🚀 Starting Airline Customer Satisfaction Prediction API...")
    print("📊 Loading model and connecting to MongoDB...")
    
    try:
        # Run the FastAPI app with uvicorn
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n✋ Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()