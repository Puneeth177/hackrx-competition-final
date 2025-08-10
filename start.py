"""
Startup script for Railway deployment
"""

import os
import sys
import time

def main():
    """Start the application"""
    try:
        # Get port from environment
        port = int(os.environ.get("PORT", 8000))
        
        print(f"🚀 Starting HackRX API on port {port}")
        print(f"📁 Working directory: {os.getcwd()}")
        print(f"🐍 Python version: {sys.version}")
        print(f"📦 Available files: {os.listdir('.')}")
        
        # Test imports first
        print("🔍 Testing imports...")
        try:
            import fastapi
            print(f"✅ FastAPI: {fastapi.__version__}")
        except ImportError as e:
            print(f"❌ FastAPI import failed: {e}")
            sys.exit(1)
        
        try:
            import uvicorn
            print(f"✅ Uvicorn: {uvicorn.__version__}")
        except ImportError as e:
            print(f"❌ Uvicorn import failed: {e}")
            sys.exit(1)
        
        try:
            from hackrx_api_simple import app
            print("✅ App imported successfully")
        except ImportError as e:
            print(f"❌ App import failed: {e}")
            sys.exit(1)
        
        # Small delay to ensure everything is ready
        time.sleep(2)
        
        print(f"🌐 Starting server on 0.0.0.0:{port}")
        
        # Start the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()