#!/bin/bash
# GPU-Accelerated Environment Startup Script for ITS Camera AI
# Platform Engineering: GPU-optimized deployment with performance monitoring

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
GPU_ENV_FILE="${SCRIPT_DIR}/.env.gpu"
COMPOSE_FILES="-f docker-compose.yml -f docker-compose.gpu.yml"
PROFILE="gpu"
LOG_FILE="${SCRIPT_DIR}/logs/startup-gpu.log"

# Functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1" >> "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1" >> "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Create log directory
mkdir -p "$(dirname "$LOG_FILE")"

log "Starting ITS Camera AI GPU-Accelerated Environment..."

# Check GPU prerequisites
check_gpu_prerequisites() {
    log "Checking GPU prerequisites..."
    
    # Check Docker
    if ! command -v docker >/dev/null 2>&1; then
        error "Docker is not installed. Please install Docker and try again."
        exit 1
    fi
    
    # Check Docker Compose
    if ! docker compose version >/dev/null 2>&1; then
        error "Docker Compose is not available. Please install Docker Compose and try again."
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        error "Docker daemon is not running. Please start Docker and try again."
        exit 1
    fi
    
    # Check NVIDIA Docker runtime
    if ! docker info 2>/dev/null | grep -i nvidia >/dev/null; then
        error "NVIDIA Docker runtime not found. Please install nvidia-docker2 and restart Docker."
        error "Installation guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        exit 1
    fi
    
    # Check nvidia-smi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        error "nvidia-smi not found. Please install NVIDIA drivers."
        exit 1
    fi
    
    # Check CUDA availability
    local cuda_version
    if cuda_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1); then
        log "NVIDIA Driver version: $cuda_version"
    else
        error "Unable to detect NVIDIA driver. Please check your GPU installation."
        exit 1
    fi
    
    # Check GPU count and memory
    local gpu_count
    local gpu_memory
    gpu_count=$(nvidia-smi --list-gpus | wc -l)
    gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    
    log "Detected $gpu_count GPU(s) with ${gpu_memory}MB memory"
    
    if [[ $gpu_count -eq 0 ]]; then
        error "No GPUs detected. Please check your NVIDIA GPU installation."
        exit 1
    fi
    
    # Recommend minimum memory for ML workloads
    if [[ $gpu_memory -lt 6144 ]]; then  # 6GB
        warn "GPU memory is less than 6GB. Performance may be limited for large models."
    fi
    
    # Check CUDA compute capability
    local compute_capability
    if compute_capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1); then
        log "CUDA Compute Capability: $compute_capability"
        
        # Check if it meets minimum requirements
        local major_version
        major_version=$(echo "$compute_capability" | cut -d. -f1)
        if [[ $major_version -lt 6 ]]; then
            warn "GPU compute capability is less than 6.0. Some optimizations may not be available."
        fi
    fi
    
    log "GPU prerequisites check passed ✓"
}

# Setup GPU environment
setup_gpu_environment() {
    log "Setting up GPU environment..."
    
    # Create GPU environment file if it doesn't exist
    if [[ ! -f "$GPU_ENV_FILE" ]]; then
        log "Creating GPU environment file..."
        cat > "$GPU_ENV_FILE" << EOF
# GPU Environment Configuration
ENVIRONMENT=production
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility

# GPU Performance Settings
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.6
CUDA_LAUNCH_BLOCKING=0
CUDA_CACHE_DISABLE=0
CUDA_CACHE_MAXSIZE=1073741824
GPU_MEMORY_FRACTION=0.8
MIXED_PRECISION=true
TORCH_CUDNN_BENCHMARK=true

# Inference Optimization
GPU_BATCH_SIZE=32
GPU_MAX_BATCH_DELAY=50
GPU_MODEL_CACHE_SIZE=2048
TENSORRT_ENABLED=true
ONNX_RUNTIME_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider

# Multi-GPU Settings
NCCL_DEBUG=INFO
NCCL_SOCKET_IFNAME=eth0

# Monitoring
NVIDIA_ML_PY=1
GPU_MONITORING_ENABLED=true

# Ports
GPU_APP_PORT=8000
GPU_METRICS_PORT=8001
GPU_PROFILER_PORT=8002
NVIDIA_EXPORTER_PORT=9835
TRITON_HTTP_PORT=8100
TRITON_GRPC_PORT=8101
TRITON_METRICS_PORT=8102

# Resource Limits
GPU_APP_MEMORY_LIMIT=16G
GPU_APP_CPU_LIMIT=8.0
GPU_APP_MEMORY_RESERVATION=8G
GPU_APP_CPU_RESERVATION=4.0

# Build settings
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse HEAD 2>/dev/null || echo 'unknown')
EOF
        log "Created GPU environment file"
    fi
    
    # Create GPU-specific directories
    local dirs=(
        "data/gpu"
        "logs/gpu"
        "models/gpu"
        "benchmark_results"
        "cache/gpu"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "${SCRIPT_DIR}/$dir"
    done
    
    log "GPU environment setup completed ✓"
}

# GPU system optimization
optimize_gpu_system() {
    log "Optimizing system for GPU workloads..."
    
    # Set GPU performance mode (if available)
    if command -v nvidia-smi >/dev/null 2>&1; then
        info "Setting GPU performance mode..."
        nvidia-smi -pm 1 2>/dev/null || warn "Could not set GPU persistence mode (may require sudo)"
        nvidia-smi -ac $(nvidia-smi --query-supported-clocks=memory,graphics --format=csv,noheader,nounits | head -1 | tr ',' ' ') 2>/dev/null || warn "Could not set GPU application clocks"
    fi
    
    # Check GPU utilization before starting
    info "Current GPU status:"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | while IFS=',' read -r name util mem_used mem_total temp; do
        info "GPU: $name | Utilization: $util% | Memory: ${mem_used}MB/${mem_total}MB | Temp: ${temp}°C"
    done
    
    log "GPU system optimization completed ✓"
}

# Pull GPU-optimized images
pull_gpu_images() {
    log "Pulling GPU-optimized Docker images..."
    
    # Pull images with retry logic
    local max_retries=3
    local retry_count=0
    
    while [[ $retry_count -lt $max_retries ]]; do
        if docker compose $COMPOSE_FILES pull --ignore-pull-failures; then
            break
        else
            ((retry_count++))
            warn "Image pull failed, retry $retry_count/$max_retries"
            sleep 10
        fi
    done
    
    if [[ $retry_count -eq $max_retries ]]; then
        warn "Some images failed to pull after $max_retries retries"
    fi
    
    log "GPU images pulled ✓"
}

# Build GPU-optimized images
build_gpu_images() {
    log "Building GPU-optimized Docker images..."
    
    # Detect GPU architecture for optimization
    local gpu_arch
    if gpu_arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1); then
        export TORCH_CUDA_ARCH_LIST="$(echo "$gpu_arch" | tr '.' '')+PTX"
        info "Building for GPU architecture: $gpu_arch"
    fi
    
    # Build with GPU optimizations
    docker compose $COMPOSE_FILES build --parallel \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse HEAD 2>/dev/null || echo 'unknown')" \
        --build-arg CUDA_VERSION="${CUDA_VERSION:-12.6}" \
        --build-arg CUDNN_VERSION="${CUDNN_VERSION:-8}"
    
    log "GPU images built ✓"
}

# Start GPU services
start_gpu_services() {
    log "Starting GPU services..."
    
    # Start infrastructure services first
    info "Starting infrastructure services..."
    docker compose $COMPOSE_FILES --profile "$PROFILE" up -d \
        postgres redis minio timescaledb
    
    # Wait for infrastructure
    sleep 15
    
    # Initialize MinIO
    docker compose $COMPOSE_FILES --profile "$PROFILE" up --no-deps minio-init
    
    # Start GPU monitoring
    if [[ "${ENABLE_GPU_MONITORING:-true}" == "true" ]]; then
        info "Starting GPU monitoring..."
        docker compose $COMPOSE_FILES --profile "$PROFILE" up -d nvidia-exporter
    fi
    
    # Start main GPU application
    info "Starting GPU-accelerated application..."
    docker compose $COMPOSE_FILES --profile "$PROFILE" up -d app-gpu
    
    # Start Triton Inference Server if enabled
    if [[ "${ENABLE_TRITON:-false}" == "true" ]]; then
        info "Starting Triton Inference Server..."
        docker compose $COMPOSE_FILES --profile triton up -d triton-inference
    fi
    
    # Start Jupyter for GPU development if enabled
    if [[ "${ENABLE_JUPYTER:-false}" == "true" ]]; then
        info "Starting GPU-enabled Jupyter..."
        docker compose $COMPOSE_FILES --profile "$PROFILE" up -d jupyter-gpu
    fi
    
    log "GPU services started ✓"
}

# Run GPU performance tests
run_gpu_performance_tests() {
    log "Running GPU performance tests..."
    
    # Wait for application to be ready
    sleep 30
    
    # Run GPU benchmark
    info "Running GPU benchmark..."
    docker compose $COMPOSE_FILES --profile benchmark up --no-deps gpu-benchmark || warn "GPU benchmark failed"
    
    # Test GPU memory allocation
    info "Testing GPU memory allocation..."
    if docker compose $COMPOSE_FILES exec -T app-gpu python -c "
import torch
if torch.cuda.is_available():
    device = torch.cuda.get_device_name(0)
    memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
    memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f'GPU: {device}')
    print(f'Memory Allocated: {memory_allocated:.2f} GB')
    print(f'Memory Reserved: {memory_reserved:.2f} GB')
    # Test tensor operations
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print('GPU tensor operations successful')
else:
    print('CUDA not available')
    exit(1)
" 2>/dev/null; then
        log "GPU memory allocation test passed ✓"
    else
        error "GPU memory allocation test failed"
        return 1
    fi
    
    # Test ML model inference
    info "Testing ML model inference..."
    if docker compose $COMPOSE_FILES exec -T app-gpu python -c "
import torch
import torchvision.models as models
import time

if torch.cuda.is_available():
    model = models.resnet18().cuda()
    model.eval()
    
    # Warm up
    dummy_input = torch.randn(1, 3, 224, 224).cuda()
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(100):
            output = model(dummy_input)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100 * 1000  # ms
    print(f'Average inference time: {avg_time:.2f} ms')
    
    if avg_time < 100:  # Sub-100ms requirement
        print('Performance requirement met ✓')
    else:
        print(f'Warning: Inference time {avg_time:.2f}ms exceeds 100ms target')
else:
    print('CUDA not available')
    exit(1)
" 2>/dev/null; then
        log "ML model inference test passed ✓"
    else
        warn "ML model inference test failed or performance below target"
    fi
    
    log "GPU performance tests completed ✓"
}

# GPU health checks
run_gpu_health_checks() {
    log "Running GPU health checks..."
    
    local services=("app-gpu" "postgres" "redis" "minio")
    if [[ "${ENABLE_GPU_MONITORING:-true}" == "true" ]]; then
        services+=("nvidia-exporter")
    fi
    if [[ "${ENABLE_TRITON:-false}" == "true" ]]; then
        services+=("triton-inference")
    fi
    
    local failed_services=()
    local timeout=180
    local count=0
    
    while [[ $count -lt $timeout ]]; do
        failed_services=()
        
        for service in "${services[@]}"; do
            local health_status
            health_status=$(docker compose $COMPOSE_FILES ps --format json | jq -r --arg service "$service" '.[] | select(.Service == $service) | .Health // "unknown"')
            
            if [[ "$health_status" != "healthy" ]]; then
                failed_services+=("$service:$health_status")
            fi
        done
        
        if [[ ${#failed_services[@]} -eq 0 ]]; then
            break
        fi
        
        sleep 5
        ((count += 5))
        info "Waiting for services to be healthy... ($count/$timeout seconds)"
    done
    
    if [[ ${#failed_services[@]} -gt 0 ]]; then
        error "The following services are not healthy: ${failed_services[*]}"
        return 1
    fi
    
    # GPU-specific health checks
    info "Checking GPU accessibility..."
    if ! docker compose $COMPOSE_FILES exec -T app-gpu python -c "import torch; assert torch.cuda.is_available(); print(f'GPUs available: {torch.cuda.device_count()}')" >/dev/null 2>&1; then
        error "GPU accessibility check failed"
        return 1
    fi
    
    log "All GPU health checks passed ✓"
}

# Monitor GPU performance
monitor_gpu_performance() {
    log "Starting GPU performance monitoring..."
    
    # Display current GPU status
    info "=== GPU Status ==="
    nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | while IFS=',' read -r name gpu_util mem_util mem_used mem_total temp power; do
        echo "🔥 GPU: $name"
        echo "   📊 GPU Utilization: $gpu_util%"
        echo "   💾 Memory Utilization: $mem_util%"
        echo "   💻 Memory: ${mem_used}MB/${mem_total}MB"
        echo "   🌡️  Temperature: ${temp}°C"
        echo "   ⚡ Power: ${power}W"
        echo
    done
    
    # Show running processes
    info "=== GPU Processes ==="
    nvidia-smi pmon -c 1 -s um || true
    
    log "GPU performance monitoring active ✓"
}

# Display GPU service information
show_gpu_service_info() {
    log "GPU environment is ready! 🚀🔥"
    echo
    info "=== GPU Service URLs ==="
    echo "🔗 Main Application:     http://localhost:${GPU_APP_PORT:-8000}"
    echo "📊 Metrics:              http://localhost:${GPU_METRICS_PORT:-8001}/metrics"
    echo "🔬 GPU Profiler:         http://localhost:${GPU_PROFILER_PORT:-8002}"
    echo "🔍 API Documentation:    http://localhost:${GPU_APP_PORT:-8000}/docs"
    
    if [[ "${ENABLE_GPU_MONITORING:-true}" == "true" ]]; then
        echo "📈 GPU Metrics:          http://localhost:${NVIDIA_EXPORTER_PORT:-9835}/metrics"
    fi
    
    if [[ "${ENABLE_TRITON:-false}" == "true" ]]; then
        echo "🚀 Triton HTTP:          http://localhost:${TRITON_HTTP_PORT:-8100}/v2/health/ready"
        echo "🔌 Triton gRPC:          localhost:${TRITON_GRPC_PORT:-8101}"
        echo "📊 Triton Metrics:       http://localhost:${TRITON_METRICS_PORT:-8102}/metrics"
    fi
    
    if [[ "${ENABLE_JUPYTER:-false}" == "true" ]]; then
        echo "🐍 Jupyter Lab:          http://localhost:${JUPYTER_GPU_PORT:-8888}?token=jupyter-gpu-token-12345"
    fi
    
    echo
    info "=== GPU Commands ==="
    echo "📋 View logs:            docker compose $COMPOSE_FILES logs -f app-gpu"
    echo "🔧 Execute shell:        docker compose $COMPOSE_FILES exec app-gpu bash"
    echo "🧪 Run tests:            docker compose $COMPOSE_FILES exec app-gpu pytest -m gpu"
    echo "📊 GPU status:           nvidia-smi"
    echo "🔥 GPU monitoring:       watch -n 1 nvidia-smi"
    echo "🛑 Stop all:             docker compose $COMPOSE_FILES down"
    echo
    info "=== GPU Features ==="
    echo "🔥 CUDA-accelerated inference"
    echo "⚡ Sub-100ms inference latency target"
    echo "📊 Real-time GPU monitoring"
    echo "🧠 Mixed precision training"
    echo "🚀 TensorRT optimization"
    if [[ "${ENABLE_TRITON:-false}" == "true" ]]; then
        echo "🏎️  Triton Inference Server"
    fi
    echo
    
    # Show GPU specifications
    info "=== GPU Specifications ==="
    nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader,nounits | while IFS=',' read -r name compute_cap driver memory; do
        echo "🔥 GPU: $name"
        echo "   🧮 Compute Capability: $compute_cap"
        echo "   🔧 Driver Version: $driver"
        echo "   💾 Total Memory: ${memory}MB"
    done
    echo
}

# Cleanup function
cleanup() {
    if [[ "${1:-}" == "error" ]]; then
        error "GPU setup failed. Cleaning up..."
        docker compose $COMPOSE_FILES down --remove-orphans 2>/dev/null || true
    fi
}

# Main execution
main() {
    # Setup error handling
    trap 'cleanup error; exit 1' ERR
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --triton)
                export ENABLE_TRITON=true
                shift
                ;;
            --jupyter)
                export ENABLE_JUPYTER=true
                shift
                ;;
            --no-monitoring)
                export ENABLE_GPU_MONITORING=false
                shift
                ;;
            --benchmark)
                export RUN_BENCHMARK=true
                shift
                ;;
            --build)
                FORCE_BUILD=true
                shift
                ;;
            --clean)
                log "Cleaning up existing GPU containers..."
                docker compose $COMPOSE_FILES down --volumes --remove-orphans
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo
                echo "Options:"
                echo "  --triton         Enable Triton Inference Server"
                echo "  --jupyter        Enable GPU-enabled Jupyter Lab"
                echo "  --no-monitoring  Disable GPU monitoring"
                echo "  --benchmark      Run performance benchmarks"
                echo "  --build          Force rebuild of images"
                echo "  --clean          Clean up before starting"
                echo "  --help           Show this help message"
                exit 0
                ;;
            *)
                warn "Unknown option: $1"
                shift
                ;;
        esac
    done
    
    # Execute setup steps
    check_gpu_prerequisites
    setup_gpu_environment
    optimize_gpu_system
    
    # Load environment files
    if [[ -f "$ENV_FILE" ]]; then
        set -a
        source "$ENV_FILE"
        set +a
    fi
    
    if [[ -f "$GPU_ENV_FILE" ]]; then
        set -a
        source "$GPU_ENV_FILE"
        set +a
    fi
    
    # Pull and build images
    pull_gpu_images
    
    if [[ "${FORCE_BUILD:-false}" == "true" ]]; then
        build_gpu_images
    fi
    
    # Start GPU services
    start_gpu_services
    
    # Wait for services to initialize
    sleep 20
    
    # Run health checks
    if ! run_gpu_health_checks; then
        error "GPU health checks failed. Please check the logs."
        exit 1
    fi
    
    # Run performance tests
    if [[ "${RUN_BENCHMARK:-true}" == "true" ]]; then
        run_gpu_performance_tests
    fi
    
    # Monitor GPU performance
    monitor_gpu_performance
    
    # Show service information
    show_gpu_service_info
    
    log "GPU environment startup completed successfully!"
}

# Run main function
main "$@"