"""
Manually download BLIP VQA config file
"""
from huggingface_hub import hf_hub_download
import os

print("Attempting to download BLIP VQA config file...")
print("This may take a few moments...")

try:
    # Download preprocessor_config.json
    config_path = hf_hub_download(
        repo_id="Salesforce/blip-vqa-base",
        filename="preprocessor_config.json",
        force_download=True,
        resume_download=True
    )
    print(f"✅ Config downloaded to: {config_path}")
    
    # Verify the file exists
    if os.path.exists(config_path):
        print(f"✅ File verified at: {config_path}")
        with open(config_path, 'r') as f:
            print(f"File content preview: {f.read()[:200]}...")
    else:
        print("❌ File not found after download")
        
except Exception as e:
    print(f"❌ Download failed: {e}")
    print("\nTrying alternative approach...")
    
    # Try using transformers to load the model (will auto-download)
    try:
        from transformers import BlipProcessor
        print("Loading BLIP VQA processor (will auto-download config)...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        print("✅ BLIP VQA processor loaded successfully!")
        print("Config files should now be cached")
    except Exception as e2:
        print(f"❌ Alternative approach also failed: {e2}")
