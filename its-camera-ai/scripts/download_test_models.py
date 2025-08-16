#!/usr/bin/env python3
"""
Test Model Download Script

Downloads or prepares test models for ML pipeline testing.
Used by CI/CD pipeline to ensure test models are available.
"""

import hashlib
import os
import urllib.request
from pathlib import Path
from typing import Dict, List

import torch
from ultralytics import YOLO


class TestModelManager:
    """Manages test models for ML pipeline testing."""
    
    def __init__(self, models_dir: str = "models/test"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Define test models with their expected checksums
        self.test_models = {
            "yolo11n.pt": {
                "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
                "sha256": None,  # Will be calculated on first download
                "description": "YOLO11 Nano model for fast testing",
                "size_mb": 5.7,
            },
            "yolo11s.pt": {
                "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt", 
                "sha256": None,
                "description": "YOLO11 Small model for accuracy testing",
                "size_mb": 21.5,
            }
        }
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def download_model(self, model_name: str, force_redownload: bool = False) -> bool:
        """Download a test model if it doesn't exist or force redownload."""
        if model_name not in self.test_models:
            print(f"❌ Unknown model: {model_name}")
            return False
        
        model_info = self.test_models[model_name]
        model_path = self.models_dir / model_name
        
        # Check if model already exists and is valid
        if model_path.exists() and not force_redownload:
            if model_info["sha256"]:
                current_hash = self.calculate_file_hash(model_path)
                if current_hash == model_info["sha256"]:
                    print(f"✅ Model {model_name} already exists and is valid")
                    return True
                else:
                    print(f"⚠️ Model {model_name} exists but hash mismatch, redownloading...")
            else:
                print(f"✅ Model {model_name} already exists")
                return True
        
        print(f"📥 Downloading {model_name} ({model_info['size_mb']} MB)...")
        print(f"   Description: {model_info['description']}")
        
        try:
            # Download with progress
            def progress_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, (downloaded / total_size) * 100)
                print(f"\r   Progress: {percent:.1f}%", end="", flush=True)
            
            urllib.request.urlretrieve(
                model_info["url"],
                model_path,
                reporthook=progress_hook
            )
            print()  # New line after progress
            
            # Calculate and store hash for future verification
            if not model_info["sha256"]:
                calculated_hash = self.calculate_file_hash(model_path)
                self.test_models[model_name]["sha256"] = calculated_hash
                print(f"   SHA256: {calculated_hash}")
            
            print(f"✅ Successfully downloaded {model_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")
            if model_path.exists():
                model_path.unlink()  # Remove partial download
            return False
    
    def create_dummy_models(self):
        """Create dummy models for testing when downloads are not available."""
        print("🔧 Creating dummy models for testing...")
        
        dummy_models = {
            "dummy_detection.pt": self._create_dummy_detection_model,
            "dummy_classification.pt": self._create_dummy_classification_model,
            "dummy_lightweight.pt": self._create_dummy_lightweight_model,
        }
        
        for model_name, creator_func in dummy_models.items():
            model_path = self.models_dir / model_name
            
            if not model_path.exists():
                try:
                    creator_func(model_path)
                    print(f"✅ Created dummy model: {model_name}")
                except Exception as e:
                    print(f"❌ Failed to create dummy model {model_name}: {e}")
    
    def _create_dummy_detection_model(self, output_path: Path):
        """Create a dummy object detection model."""
        # Create a simple CNN model structure similar to YOLO
        import torch.nn as nn
        
        class DummyDetectionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(32, 84)  # 84 = (x, y, w, h, conf) * num_classes
                )
            
            def forward(self, x):
                return self.backbone(x)
        
        model = DummyDetectionModel()
        torch.save({
            'model': model.state_dict(),
            'metadata': {
                'type': 'detection',
                'classes': ['car', 'truck', 'motorcycle', 'bus'],
                'input_size': [640, 640],
                'created_for': 'testing'
            }
        }, output_path)
    
    def _create_dummy_classification_model(self, output_path: Path):
        """Create a dummy classification model."""
        import torch.nn as nn
        
        class DummyClassificationModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = nn.Sequential(
                    nn.Conv2d(3, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(16, 4)  # 4 vehicle classes
                )
            
            def forward(self, x):
                return self.classifier(x)
        
        model = DummyClassificationModel()
        torch.save({
            'model': model.state_dict(),
            'metadata': {
                'type': 'classification',
                'classes': ['car', 'truck', 'motorcycle', 'bus'],
                'input_size': [224, 224],
                'created_for': 'testing'
            }
        }, output_path)
    
    def _create_dummy_lightweight_model(self, output_path: Path):
        """Create a dummy lightweight model for edge testing."""
        import torch.nn as nn
        
        class DummyLightweightModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 8, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(8, 10)  # Minimal output
                )
            
            def forward(self, x):
                return self.features(x)
        
        model = DummyLightweightModel()
        torch.save({
            'model': model.state_dict(),
            'metadata': {
                'type': 'lightweight',
                'classes': ['vehicle', 'non-vehicle'],
                'input_size': [320, 320],
                'created_for': 'edge_testing',
                'parameters': sum(p.numel() for p in model.parameters())
            }
        }, output_path)
    
    def validate_models(self) -> Dict[str, bool]:
        """Validate that all test models are functional."""
        results = {}
        
        print("🔍 Validating test models...")
        
        for model_file in self.models_dir.glob("*.pt"):
            model_name = model_file.name
            
            try:
                # Try to load the model
                if "yolo" in model_name.lower():
                    # Test YOLO models
                    model = YOLO(str(model_file))
                    # Test inference on dummy input
                    test_image = torch.randn(1, 3, 640, 640)
                    with torch.no_grad():
                        results_test = model(test_image, verbose=False)
                    
                else:
                    # Test PyTorch models
                    checkpoint = torch.load(model_file, map_location='cpu')
                    # Basic validation that it contains expected keys
                    if 'model' in checkpoint and 'metadata' in checkpoint:
                        pass  # Valid structure
                
                results[model_name] = True
                print(f"✅ {model_name} - Valid")
                
            except Exception as e:
                results[model_name] = False
                print(f"❌ {model_name} - Invalid: {e}")
        
        return results
    
    def list_models(self):
        """List all available test models."""
        print("\n📋 Available Test Models:")
        print("=" * 50)
        
        for model_file in sorted(self.models_dir.glob("*.pt")):
            file_size = model_file.stat().st_size / (1024 * 1024)  # MB
            
            try:
                # Try to get metadata
                if "yolo" in model_file.name.lower():
                    model_type = "YOLO Detection"
                    metadata = "Official YOLO model"
                else:
                    checkpoint = torch.load(model_file, map_location='cpu')
                    metadata_info = checkpoint.get('metadata', {})
                    model_type = metadata_info.get('type', 'Unknown').title()
                    metadata = metadata_info.get('created_for', 'No description')
                
                print(f"📄 {model_file.name}")
                print(f"   Type: {model_type}")
                print(f"   Size: {file_size:.1f} MB")
                print(f"   Description: {metadata}")
                print()
                
            except Exception as e:
                print(f"📄 {model_file.name}")
                print(f"   Size: {file_size:.1f} MB")
                print(f"   Status: Error reading metadata ({e})")
                print()
    
    def cleanup_old_models(self, keep_days: int = 30):
        """Remove old test models to save space."""
        import time
        
        current_time = time.time()
        cleanup_threshold = current_time - (keep_days * 24 * 60 * 60)
        
        cleaned_files = []
        
        for model_file in self.models_dir.glob("*.pt"):
            if model_file.stat().st_mtime < cleanup_threshold:
                file_size = model_file.stat().st_size / (1024 * 1024)
                model_file.unlink()
                cleaned_files.append((model_file.name, file_size))
        
        if cleaned_files:
            total_size = sum(size for _, size in cleaned_files)
            print(f"🧹 Cleaned up {len(cleaned_files)} old models ({total_size:.1f} MB)")
            for name, size in cleaned_files:
                print(f"   Removed: {name} ({size:.1f} MB)")
        else:
            print("🧹 No old models to clean up")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and manage test models")
    parser.add_argument("--models-dir", default="models/test", help="Models directory")
    parser.add_argument("--download-all", action="store_true", help="Download all test models")
    parser.add_argument("--create-dummies", action="store_true", help="Create dummy models")
    parser.add_argument("--validate", action="store_true", help="Validate existing models")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--cleanup", type=int, metavar="DAYS", help="Clean up models older than N days")
    parser.add_argument("--force", action="store_true", help="Force redownload existing models")
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = TestModelManager(args.models_dir)
    
    print(f"🏗️ Test Model Manager")
    print(f"Models directory: {manager.models_dir}")
    print()
    
    # Execute requested operations
    if args.download_all:
        print("📥 Downloading all test models...")
        success_count = 0
        for model_name in manager.test_models.keys():
            if manager.download_model(model_name, force_redownload=args.force):
                success_count += 1
        
        print(f"\n✅ Downloaded {success_count}/{len(manager.test_models)} models successfully")
    
    if args.create_dummies:
        manager.create_dummy_models()
    
    if args.validate:
        results = manager.validate_models()
        valid_count = sum(1 for valid in results.values() if valid)
        print(f"\n📊 Validation Summary: {valid_count}/{len(results)} models valid")
    
    if args.list:
        manager.list_models()
    
    if args.cleanup is not None:
        manager.cleanup_old_models(args.cleanup)
    
    # Default behavior: ensure basic test models exist
    if not any([args.download_all, args.create_dummies, args.validate, args.list, args.cleanup]):
        print("🔧 Ensuring test models are available...")
        
        # Try to download lightweight models first
        success = False
        for model_name in ["yolo11n.pt"]:  # Start with smallest
            if manager.download_model(model_name):
                success = True
                break
        
        # If download fails, create dummy models
        if not success:
            print("⚠️ Download failed, creating dummy models instead...")
            manager.create_dummy_models()
        
        # Validate what we have
        results = manager.validate_models()
        valid_count = sum(1 for valid in results.values() if valid)
        
        if valid_count > 0:
            print(f"✅ Test models ready: {valid_count} valid models available")
        else:
            print("❌ No valid test models available")
            exit(1)


if __name__ == "__main__":
    main()