"""
Quick start script for Llama server
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

def find_model():
    """Find Llama model file."""
    possible_paths = [
        "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "models/llama-3.2-3b-instruct.Q4_K_M.gguf",
        "models/llama-3.2-3b-instruct.gguf",
        "models/llama-3.2-3b.Q4_K_M.gguf",
        "models/llama-3.2-3b.gguf",
        "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "llama-3.2-3b-instruct.Q4_K_M.gguf",
        "llama-3.2-3b-instruct.gguf"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            return path
    
    return None

def start_server(model_path, port=8010):
    """Start Llama server."""
    print(f"🚀 Starting Llama server...")
    print(f"Model: {model_path}")
    print(f"Port: {port}")
    
    cmd = [
        sys.executable, "-m", "llama_cpp.server",
        "--model", model_path,
        "--port", str(port),
        "--host", "127.0.0.1",
        "--n_ctx", "2048",
        "--n_gpu_layers", "0",  # CPU only
        "--verbose"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\n⏳ Starting server... (this may take 30-60 seconds)")
    
    try:
        process = subprocess.Popen(cmd)
        
        # Wait for server to start
        print("⏳ Waiting for server to initialize...")
        for i in range(30):  # Wait up to 30 seconds
            time.sleep(1)
            try:
                response = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=2)
                if response.status_code == 200:
                    print(f"✅ Server is ready! (took {i+1} seconds)")
                    print(f"✅ Server PID: {process.pid}")
                    return process
            except requests.exceptions.RequestException:
                continue
        
        print("❌ Server failed to start within 30 seconds")
        process.terminate()
        return None
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return None

def test_server(port=8010):
    """Quick server test."""
    print(f"\n🧪 Testing server...")
    
    try:
        response = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=5)
        if response.status_code == 200:
            print("✅ Server is responding")
            return True
        else:
            print(f"❌ Server error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server test failed: {e}")
        return False

def main():
    """Main function."""
    print("="*50)
    print("QUICK LLAMA SERVER START")
    print("="*50)
    
    # Find model
    model_path = find_model()
    if not model_path:
        print("❌ No Llama model found!")
        print("\n📋 Please download a Llama model:")
        print("1. Go to: https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF")
        print("2. Download: Llama-3.2-3B-Instruct-Q4_K_M.gguf")
        print("3. Save to: models/ directory")
        print("\nOr run: python download_llama_model.py")
        return False
    
    print(f"✅ Found model: {model_path}")
    
    # Start server
    server_process = start_server(model_path)
    if not server_process:
        return False
    
    # Test server
    if test_server():
        print("\n🎉 Llama server is ready!")
        print(f"✅ URL: http://127.0.0.1:8010")
        print(f"✅ PID: {server_process.pid}")
        
        print("\n📋 Now you can run:")
        print("  python test_real_rag.py")
        print("  python evaluation_main.py")
        
        print("\n⚠️  Keep this terminal open!")
        print("Press Ctrl+C to stop server")
        
        # Keep running
        try:
            server_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping server...")
            server_process.terminate()
            server_process.wait()
            print("✅ Server stopped")
        
        return True
    else:
        print("❌ Server test failed")
        server_process.terminate()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Failed to start server")
        sys.exit(1)