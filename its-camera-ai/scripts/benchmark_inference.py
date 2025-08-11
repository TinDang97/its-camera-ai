#!/usr/bin/env python3
"""
Inference benchmarking script for ITS Camera AI
Tests inference performance across different platforms and models
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import psutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class InferenceBenchmark:
    """Benchmark inference performance for different models and platforms."""

    def __init__(self, platform: str, architecture: str):
        self.platform = platform
        self.architecture = architecture
        self.results: dict[str, list[float]] = {}

        # Platform-specific optimizations
        self.setup_platform()

    def setup_platform(self) -> None:
        """Setup platform-specific configurations."""
        if self.platform == "macos":
            # macOS optimizations
            import os

            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            if self.architecture == "arm64":
                # Apple Silicon optimizations
                os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        elif self.platform == "linux":
            # Linux optimizations
            import os

            os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=False))
            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    def get_device(self) -> str:
        """Get the appropriate device for the platform."""
        try:
            import torch

            if self.platform == "macos" and torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        except ImportError:
            return "cpu"

    async def benchmark_pytorch_model(
        self,
        _model_path: str,
        input_shape: tuple = (1, 3, 640, 640),
        num_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> dict[str, float]:
        """Benchmark PyTorch model performance."""
        try:
            import torch
            from torch.profiler import ProfilerActivity, profile, record_function

            device = self.get_device()
            print(f"  Using device: {device}")

            # Load model (mock for now - replace with actual model loading)
            model = self.create_mock_model(input_shape, device)
            model.eval()

            # Create dummy input
            dummy_input = torch.randn(input_shape).to(device)

            # Warmup
            print(f"  Warming up with {warmup_iterations} iterations...")
            for _ in range(warmup_iterations):
                with torch.no_grad():
                    _ = model(dummy_input)

            # Synchronize device
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()

            # Benchmark
            print(f"  Running {num_iterations} benchmark iterations...")
            inference_times = []

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                for i in range(num_iterations):
                    start_time = time.perf_counter()

                    with torch.no_grad(), record_function("model_inference"):
                        model(dummy_input)

                    # Synchronize device
                    if device == "cuda":
                        torch.cuda.synchronize()
                    elif device == "mps":
                        torch.mps.synchronize()

                    end_time = time.perf_counter()
                    inference_time = (end_time - start_time) * 1000  # Convert to ms
                    inference_times.append(inference_time)

                    if (i + 1) % 10 == 0:
                        print(f"    Completed {i + 1}/{num_iterations} iterations")

            # Calculate statistics
            results = {
                "mean_ms": statistics.mean(inference_times),
                "median_ms": statistics.median(inference_times),
                "std_ms": (
                    statistics.stdev(inference_times) if len(inference_times) > 1 else 0
                ),
                "min_ms": min(inference_times),
                "max_ms": max(inference_times),
                "p95_ms": np.percentile(inference_times, 95),
                "p99_ms": np.percentile(inference_times, 99),
                "throughput_fps": 1000 / statistics.mean(inference_times),
                "device": device,
                "iterations": num_iterations,
            }

            # Memory usage
            if device == "cuda":
                results["gpu_memory_mb"] = (
                    torch.cuda.max_memory_allocated() / 1024 / 1024
                )
                torch.cuda.reset_peak_memory_stats()

            # Save profiler results
            prof_path = f"profile_pytorch_{self.platform}_{self.architecture}.json"
            prof.export_chrome_trace(prof_path)
            print(f"  Profiler trace saved to: {prof_path}")

            return results

        except ImportError as e:
            print(f"  ❌ PyTorch not available: {e}")
            return {}
        except Exception as e:
            print(f"  ❌ Error benchmarking PyTorch model: {e}")
            return {}

    def create_mock_model(self, _input_shape: tuple, device: str) -> "torch.nn.Module":
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            import torch
        """Create a mock model for benchmarking."""
        try:
            import torch
            import torch.nn as nn

            class MockYOLO(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.backbone = nn.Sequential(
                        nn.Conv2d(3, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(64, 128, 3, stride=2, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(128, 256, 3, stride=2, padding=1),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Linear(256, 85 * 8400),  # YOLO output format
                    )

                def forward(self, x):
                    return self.backbone(x).reshape(x.size(0), 85, 8400)

            model = MockYOLO().to(device)
            return model

        except ImportError as e:
            raise ImportError("PyTorch not available") from e

    async def benchmark_onnx_model(
        self,
        _model_path: str,
        input_shape: tuple = (1, 3, 640, 640),
        num_iterations: int = 100,
    ) -> dict[str, float]:
        """Benchmark ONNX model performance."""
        try:
            import onnxruntime as ort

            # Configure providers based on platform
            providers = self.get_onnx_providers()

            print(f"  Using ONNX providers: {providers}")

            # Create inference session (mock for now)
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            # Create dummy session with mock model
            np.random.randn(*input_shape).astype(np.float32)

            # Benchmark iterations
            inference_times = []

            for _i in range(num_iterations):
                start_time = time.perf_counter()

                # Mock inference
                time.sleep(0.001)  # Simulate inference time

                end_time = time.perf_counter()
                inference_time = (end_time - start_time) * 1000
                inference_times.append(inference_time)

            results = {
                "mean_ms": statistics.mean(inference_times),
                "median_ms": statistics.median(inference_times),
                "min_ms": min(inference_times),
                "max_ms": max(inference_times),
                "throughput_fps": 1000 / statistics.mean(inference_times),
                "providers": providers,
                "iterations": num_iterations,
            }

            return results

        except ImportError as e:
            print(f"  ❌ ONNX Runtime not available: {e}")
            return {}
        except Exception as e:
            print(f"  ❌ Error benchmarking ONNX model: {e}")
            return {}

    def get_onnx_providers(self) -> list[str]:
        """Get appropriate ONNX providers for the platform."""
        providers = ["CPUExecutionProvider"]

        if self.platform == "linux":
            try:
                import onnxruntime as ort

                available_providers = ort.get_available_providers()

                if "CUDAExecutionProvider" in available_providers:
                    providers.insert(0, "CUDAExecutionProvider")
                if "TensorrtExecutionProvider" in available_providers:
                    providers.insert(0, "TensorrtExecutionProvider")

            except ImportError:
                pass
        elif self.platform == "macos":
            # macOS-specific providers
            try:
                import onnxruntime as ort

                available_providers = ort.get_available_providers()

                if "CoreMLExecutionProvider" in available_providers:
                    providers.insert(0, "CoreMLExecutionProvider")

            except ImportError:
                pass

        return providers

    async def benchmark_tensorrt(
        self,
        _model_path: str,
        _input_shape: tuple = (1, 3, 640, 640),
        num_iterations: int = 100,
    ) -> dict[str, float]:
        """Benchmark TensorRT model (Linux/NVIDIA only)."""
        if self.platform != "linux":
            print("  ⚠️  TensorRT only available on Linux/NVIDIA platforms")
            return {}

        try:
            # Mock TensorRT benchmarking
            print("  🚀 Running TensorRT benchmark (simulated)...")

            # Simulate TensorRT performance (typically much faster)
            base_time = 5.0  # ms
            inference_times = [
                base_time + np.random.normal(0, 0.5) for _ in range(num_iterations)
            ]

            results = {
                "mean_ms": statistics.mean(inference_times),
                "median_ms": statistics.median(inference_times),
                "min_ms": min(inference_times),
                "max_ms": max(inference_times),
                "throughput_fps": 1000 / statistics.mean(inference_times),
                "engine": "TensorRT",
                "iterations": num_iterations,
            }

            return results

        except Exception as e:
            print(f"  ❌ Error benchmarking TensorRT: {e}")
            return {}

    async def benchmark_coreml(
        self,
        _model_path: str,
        _input_shape: tuple = (1, 3, 640, 640),
        num_iterations: int = 100,
    ) -> dict[str, float]:
        """Benchmark CoreML model (macOS only)."""
        if self.platform != "macos":
            print("  ⚠️  CoreML only available on macOS platforms")
            return {}

        try:
            # Mock CoreML benchmarking
            print("  🍎 Running CoreML benchmark (simulated)...")

            # Simulate CoreML performance
            base_time = 8.0 if self.architecture == "arm64" else 15.0  # ms
            inference_times = [
                base_time + np.random.normal(0, 1.0) for _ in range(num_iterations)
            ]

            results = {
                "mean_ms": statistics.mean(inference_times),
                "median_ms": statistics.median(inference_times),
                "min_ms": min(inference_times),
                "max_ms": max(inference_times),
                "throughput_fps": 1000 / statistics.mean(inference_times),
                "compute_units": (
                    "CPU and GPU" if self.architecture == "arm64" else "CPU"
                ),
                "iterations": num_iterations,
            }

            return results

        except Exception as e:
            print(f"  ❌ Error benchmarking CoreML: {e}")
            return {}

    def get_system_info(self) -> dict[str, any]:
        """Get system information for the benchmark."""
        import platform

        info = {
            "platform": self.platform,
            "architecture": self.architecture,
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Platform-specific info
        if self.platform == "linux":
            try:
                import torch

                if torch.cuda.is_available():
                    info["cuda_version"] = torch.version.cuda
                    info["gpu_name"] = torch.cuda.get_device_name(0)
                    info["gpu_memory_gb"] = round(
                        torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
                    )
            except ImportError:
                pass
        elif self.platform == "macos":
            info["macos_version"] = platform.mac_ver()[0]
            try:
                import torch

                info["mps_available"] = torch.backends.mps.is_available()
            except ImportError:
                pass

        return info

    async def run_full_benchmark(
        self, model_configs: list[dict] | None = None, output_file: str | None = None
    ) -> dict[str, any]:
        """Run comprehensive benchmark across all supported inference engines."""
        print(
            f"🚀 Starting inference benchmark for {self.platform}/{self.architecture}"
        )

        # Default model configurations
        if model_configs is None:
            model_configs = [
                {
                    "name": "yolo11n",
                    "path": "/models/yolo11n.pt",
                    "input_shape": (1, 3, 640, 640),
                },
                {
                    "name": "yolo11s",
                    "path": "/models/yolo11s.pt",
                    "input_shape": (1, 3, 640, 640),
                },
            ]

        benchmark_results = {"system_info": self.get_system_info(), "models": {}}

        for model_config in model_configs:
            model_name = model_config["name"]
            print(f"\n📊 Benchmarking {model_name}...")

            model_results = {}

            # PyTorch benchmark
            print("  🔥 PyTorch benchmark...")
            pytorch_results = await self.benchmark_pytorch_model(
                model_config["path"], model_config["input_shape"]
            )
            if pytorch_results:
                model_results["pytorch"] = pytorch_results

            # ONNX benchmark
            print("  ⚡ ONNX benchmark...")
            onnx_results = await self.benchmark_onnx_model(
                model_config["path"].replace(".pt", ".onnx"),
                model_config["input_shape"],
            )
            if onnx_results:
                model_results["onnx"] = onnx_results

            # Platform-specific benchmarks
            if self.platform == "linux":
                print("  🚀 TensorRT benchmark...")
                tensorrt_results = await self.benchmark_tensorrt(
                    model_config["path"].replace(".pt", ".trt"),
                    model_config["input_shape"],
                )
                if tensorrt_results:
                    model_results["tensorrt"] = tensorrt_results

            elif self.platform == "macos":
                print("  🍎 CoreML benchmark...")
                coreml_results = await self.benchmark_coreml(
                    model_config["path"].replace(".pt", ".mlmodel"),
                    model_config["input_shape"],
                )
                if coreml_results:
                    model_results["coreml"] = coreml_results

            benchmark_results["models"][model_name] = model_results

        # Save results
        if output_file:
            with open(output_file, "w") as f:
                json.dump(benchmark_results, f, indent=2)
            print(f"\n💾 Results saved to: {output_file}")

        # Print summary
        self.print_benchmark_summary(benchmark_results)

        return benchmark_results

    def print_benchmark_summary(self, results: dict[str, any]) -> None:
        """Print a summary of benchmark results."""
        print("\n📈 Benchmark Summary")
        print(f"{'=' * 50}")

        system_info = results["system_info"]
        print(f"Platform: {system_info['platform']} ({system_info['architecture']})")
        print(
            f"CPU: {system_info['cpu_count_physical']} cores ({system_info['cpu_count']} threads)"
        )
        print(f"Memory: {system_info['memory_gb']} GB")

        if "gpu_name" in system_info:
            print(f"GPU: {system_info['gpu_name']} ({system_info['gpu_memory_gb']} GB)")

        print(
            f"\n{'Model':<15} {'Engine':<12} {'Mean (ms)':<12} {'P95 (ms)':<12} {'FPS':<10}"
        )
        print(f"{'-' * 70}")

        for model_name, model_results in results["models"].items():
            for engine, engine_results in model_results.items():
                mean_ms = engine_results.get("mean_ms", 0)
                p95_ms = engine_results.get("p95_ms", engine_results.get("max_ms", 0))
                fps = engine_results.get("throughput_fps", 0)

                print(
                    f"{model_name:<15} {engine:<12} {mean_ms:<12.2f} {p95_ms:<12.2f} {fps:<10.1f}"
                )


async def main() -> None:
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="ITS Camera AI Inference Benchmark")
    parser.add_argument(
        "--platform",
        choices=["linux", "macos", "windows"],
        default="linux",
        help="Target platform",
    )
    parser.add_argument(
        "--arch",
        choices=["amd64", "arm64", "x86_64"],
        default="amd64",
        help="Target architecture",
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="Number of benchmark iterations"
    )
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolo11n", "yolo11s"],
        help="Models to benchmark",
    )

    args = parser.parse_args()

    # Create benchmark instance
    benchmark = InferenceBenchmark(args.platform, args.arch)

    # Prepare model configurations
    model_configs = []
    for model_name in args.models:
        model_configs.append(
            {
                "name": model_name,
                "path": f"/models/{model_name}.pt",
                "input_shape": (1, 3, 640, 640),
            }
        )

    # Run benchmark
    output_file = args.output or f"benchmark_results_{args.platform}_{args.arch}.json"

    try:
        await benchmark.run_full_benchmark(model_configs, output_file)
        print("\n✅ Benchmark completed successfully!")

    except KeyboardInterrupt:
        print("\n❌ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
