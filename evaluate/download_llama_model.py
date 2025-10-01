"""
Download Llama 3.2 3B model quickly
"""

import os
import subprocess
import sys
from pathlib import Path

def install_huggingface_hub():
    """Install huggingface-hub if needed."""
    try:
        import huggingface_hub
        print("✅ huggingface-hub is installed")
        return True
    except ImportError:
        print("📦 Installing huggingface-hub...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "huggingface-hub"], check=True)
            print("✅ huggingface-hub installed")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install huggingface-hub")
            return False

def download_model():
    """Download Llama model."""
    print("📥 Downloading Llama 3.2 3B model...")
    print("This will download ~2.3 GB")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    try:
        from huggingface_hub import hf_hub_download
        
        print("⏳ Downloading from Hugging Face...")
        print("This may take 5-15 minutes depending on your internet speed...")
        
        downloaded_path = hf_hub_download(
            repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
            filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            local_dir=str(models_dir),
            local_dir_use_symlinks=False
        )
        
        print(f"✅ Model downloaded successfully!")
        print(f"📁 Location: {downloaded_path}")
        
        # Check file size
        file_size = Path(downloaded_path).stat().st_size / (1024**3)  # GB
        print(f"📊 File size: {file_size:.2f} GB")
        
        return downloaded_path
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

def main():
    """Main function."""
    print("="*50)
    print("LLAMA MODEL DOWNLOADER")
    print("="*50)
    
    # Check if model already exists
    model_path = Path("models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
    if model_path.exists():
        print(f"✅ Model already exists: {model_path}")
        print(f"📊 File size: {model_path.stat().st_size / (1024**3):.2f} GB")
        
        response = input("\nDo you want to re-download? (y/N): ").strip().lower()
        if response != 'y':
            print("✅ Using existing model")
            return str(model_path)
    
    # Install huggingface-hub
    if not install_huggingface_hub():
        print("❌ Cannot download without huggingface-hub")
        return None
    
    # Download model
    model_path = download_model()
    if model_path:
        print("\n🎉 Model download complete!")
        print("\n📋 Next steps:")
        print("1. Run: python start_llama_server.py")
        print("2. Or run: python setup_llama_server.py")
        return model_path
    else:
        print("\n❌ Download failed")
        print("\n📋 Manual download:")
        print("1. Go to: https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF")
        print("2. Download: Llama-3.2-3B-Instruct-Q4_K_M.gguf")
        print("3. Save to: models/ directory")
        return None

if __name__ == "__main__":
    model_path = main()
    if model_path:
        print(f"\n✅ Ready! Model at: {model_path}")
    else:
        print("\n❌ Download failed")
        sys.exit(1)