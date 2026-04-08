#!/usr/bin/env python3
"""
Test script to validate the pangenome analysis setup.
Run this script to check if all dependencies are installed correctly.

Usage: python test_setup.py
"""

import sys
import importlib
import subprocess
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    
    required_packages = [
        'streamlit',
        'torch', 
        'transformers',
        'Bio',  # biopython
        'umap',
        'plotly',
        'pandas',
        'numpy',
        'sklearn'
    ]
    
    print("Testing package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    return failed_imports


def test_torch_mps():
    """Test PyTorch MPS backend availability."""
    
    try:
        import torch
        
        print(f"\nPyTorch version: {torch.__version__}")
        
        if torch.backends.mps.is_available():
            print("✅ MPS backend is available")
            print("✅ Apple Silicon GPU acceleration ready")
            return True
        else:
            print("⚠️  MPS backend not available - will fall back to CPU")
            return False
            
    except Exception as e:
        print(f"❌ Error testing MPS: {e}")
        return False


def test_model_access():
    """Test access to Hugging Face transformers."""
    
    try:
        from transformers import T5Tokenizer
        
        print("\nTesting Hugging Face model access...")
        
        # Try to load tokenizer (small download)
        model_name = "Rostlab/prot_t5_xl_uniref50"
        tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        
        print("✅ ProtT5 tokenizer loaded successfully")
        print("✅ Hugging Face access working")
        
        return True
        
    except Exception as e:
        print(f"❌ Error accessing model: {e}")
        print("💡 Note: Model will be downloaded on first use (~3GB)")
        return False


def test_file_structure():
    """Test that all required files are present."""
    
    print("\nTesting file structure...")
    
    required_files = [
        'app.py',
        'config.py', 
        'requirements.txt',
        'environment.yml',
        'src/__init__.py',
        'src/data_loading.py',
        'src/embedding.py',
        'src/analysis.py',
        'src/visualization.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    return missing_files


def main():
    """Run all tests."""
    
    print("🧬 Pangenome Analysis Setup Validation")
    print("=" * 40)
    
    # Test imports
    failed_imports = test_imports()
    
    # Test PyTorch MPS
    mps_available = test_torch_mps()
    
    # Test model access (optional - may require internet)
    try:
        model_accessible = test_model_access()
    except:
        model_accessible = False
        print("⚠️  Skipping model access test (offline?)")
    
    # Test file structure
    missing_files = test_file_structure()
    
    # Summary
    print("\n" + "=" * 40)
    print("📋 SUMMARY")
    print("=" * 40)
    
    if not failed_imports and not missing_files:
        print("✅ All core components are ready!")
        
        if mps_available:
            print("🚀 Apple Silicon acceleration enabled")
        else:
            print("⚠️  Will use CPU (slower but functional)")
        
        print("\n🎯 Ready to run:")
        print("   conda activate pangenome  # (if using conda)")
        print("   streamlit run app.py")
        
    else:
        print("❌ Setup incomplete:")
        
        if failed_imports:
            print(f"   Missing packages: {failed_imports}")
            print("   Run: conda env create -f environment.yml")
            print("   Or:  pip install -r requirements.txt")
        
        if missing_files:
            print(f"   Missing files: {missing_files}")
    
    print("\n📖 See README.md for detailed setup instructions")


if __name__ == "__main__":
    main()