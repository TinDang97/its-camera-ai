#!/usr/bin/env python3
"""Generate Python protobuf files from .proto definitions.

This script compiles the protobuf definitions into Python modules
for use with the gRPC serialization system.
"""

import os
import subprocess
import sys
from pathlib import Path


def generate_proto_files():
    """Generate Python protobuf files from .proto definitions."""
    
    # Get project root directory
    project_root = Path(__file__).parent.parent
    proto_dir = project_root / "proto"
    output_dir = project_root / "src" / "its_camera_ai" / "proto"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create __init__.py
    init_file = output_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("# Generated protobuf modules\n")
    
    # Find all .proto files
    proto_files = list(proto_dir.glob("*.proto"))
    
    if not proto_files:
        print(f"No .proto files found in {proto_dir}")
        return False
    
    print(f"Found {len(proto_files)} .proto files:")
    for proto_file in proto_files:
        print(f"  - {proto_file.name}")
    
    # Generate Python files
    for proto_file in proto_files:
        print(f"\nGenerating Python code for {proto_file.name}...")
        
        try:
            # Run protoc to generate Python files
            cmd = [
                "python", "-m", "grpc_tools.protoc",
                f"--proto_path={proto_dir}",
                f"--python_out={output_dir}",
                f"--grpc_python_out={output_dir}",
                str(proto_file)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=project_root
            )
            
            if result.returncode == 0:
                print(f"  ✓ Successfully generated {proto_file.stem}_pb2.py")
                if "service" in proto_file.name.lower():
                    print(f"  ✓ Successfully generated {proto_file.stem}_pb2_grpc.py")
            else:
                print(f"  ✗ Error generating {proto_file.name}:")
                print(f"    {result.stderr}")
                return False
                
        except FileNotFoundError:
            print("  ✗ Error: grpcio-tools not found. Install with: pip install grpcio-tools")
            return False
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            return False
    
    # Fix imports in generated files
    print("\nFixing imports in generated files...")
    fix_imports(output_dir)
    
    print("\n✓ Protobuf generation completed successfully!")
    return True


def fix_imports(output_dir: Path):
    """Fix relative imports in generated protobuf files."""
    
    for py_file in output_dir.glob("*_pb2*.py"):
        content = py_file.read_text()
        
        # Fix imports to be relative
        if "import processed_frame_pb2" in content:
            content = content.replace(
                "import processed_frame_pb2",
                "from . import processed_frame_pb2"
            )
        
        if "import streaming_service_pb2" in content:
            content = content.replace(
                "import streaming_service_pb2",
                "from . import streaming_service_pb2"
            )
        
        py_file.write_text(content)
        print(f"  ✓ Fixed imports in {py_file.name}")


def check_dependencies():
    """Check if required dependencies are installed."""
    
    try:
        import grpc
        import grpc_tools
        print("✓ gRPC dependencies found")
        return True
    except ImportError as e:
        print(f"✗ Missing dependencies: {e}")
        print("Install with: pip install grpcio grpcio-tools")
        return False


if __name__ == "__main__":
    print("ITS Camera AI - Protobuf Generator")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Generate protobuf files
    if generate_proto_files():
        print("\nNext steps:")
        print("1. Install the updated dependencies: uv sync")
        print("2. Run tests to verify the implementation")
        print("3. Update your application configuration to use Redis queues")
    else:
        print("\n✗ Protobuf generation failed")
        sys.exit(1)
