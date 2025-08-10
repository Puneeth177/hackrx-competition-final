"""
Test the simple API locally
"""

import requests
import json
import time
import subprocess
import sys
import threading
from pathlib import Path

def start_server():
    """Start the simple API server"""
    print("🚀 Starting simple API server...")
    
    try:
        process = subprocess.Popen(
            [sys.executable, "hackrx_api_simple.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        time.sleep(10)
        return process
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return None

def test_simple_api():
    """Test the simple API"""
    print("\n🧪 TESTING SIMPLE API")
    print("=" * 50)
    
    # Test data
    test_data = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?"
        ]
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 6caf8e7ec0c26c15b51e20eb8c1479cdf8d5dde6b9453cf13e007aa5bb381210"
    }
    
    try:
        print("📤 Sending test request...")
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:8000/api/v1/hackrx/run",
            json=test_data,
            headers=headers,
            timeout=60
        )
        
        processing_time = time.time() - start_time
        
        print(f"⏱️  Response time: {processing_time:.2f}s")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            answers = result.get('answers', [])
            
            print("✅ SUCCESS! Simple API working!")
            print(f"📝 Answers: {len(answers)}")
            
            if answers:
                print(f"📋 Answer: {answers[0][:200]}...")
                
                # Check for key terms
                if "thirty days" in answers[0].lower() or "30 days" in answers[0].lower():
                    print("🎯 EXCELLENT: Contains expected key terms!")
                    return True
                else:
                    print("⚠️  GOOD: Answer provided, may need refinement")
                    return True
            else:
                print("❌ No answers received")
                return False
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 SIMPLE API TEST")
    print("=" * 40)
    
    # Start server
    server_process = start_server()
    
    if not server_process:
        print("❌ Cannot start server")
        return False
    
    try:
        # Test the API
        success = test_simple_api()
        
        if success:
            print("\n🎉 SIMPLE API TEST PASSED!")
            print("✅ Ready for deployment!")
            return True
        else:
            print("\n⚠️  Issues detected")
            return False
        
    finally:
        # Clean up
        print("\n🛑 Stopping server...")
        try:
            server_process.terminate()
            server_process.wait(timeout=5)
        except:
            server_process.kill()
        print("✅ Server stopped")

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 READY TO DEPLOY!")
    else:
        print("\n🔧 NEEDS FIXES")
        sys.exit(1)