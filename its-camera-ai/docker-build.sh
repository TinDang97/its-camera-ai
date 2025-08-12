#!/bin/bash
# Production-ready Docker build script for ITS Camera AI
# Supports multi-stage builds with different optimization targets

set -euo pipefail

# Configuration
IMAGE_NAME="its-camera-ai"
REGISTRY="${REGISTRY:-ghcr.io/your-org}"
VERSION="${VERSION:-0.1.0}"
BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
PLATFORM="${PLATFORM:-linux/amd64,linux/arm64}"

# Build arguments
BUILD_ARGS="
  --build-arg BUILD_DATE=${BUILD_DATE}
  --build-arg VCS_REF=${VCS_REF}
  --build-arg VERSION=${VERSION}
  --build-arg PYTHON_VERSION=3.12
  --build-arg UV_VERSION=0.5.8
"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" >&2
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Function to build a specific target
build_target() {
    local target=$1
    local tag_suffix=$2
    local platforms=${3:-"linux/amd64"}
    local push=${4:-false}
    
    log "Building ${target} target for platforms: ${platforms}"
    
    local full_tag="${REGISTRY}/${IMAGE_NAME}:${VERSION}-${tag_suffix}"
    local latest_tag="${REGISTRY}/${IMAGE_NAME}:${tag_suffix}"
    
    local push_args=""
    if [[ "$push" == "true" ]]; then
        push_args="--push"
        info "Will push to registry: ${full_tag}"
    else
        push_args="--load"
        info "Building locally: ${full_tag}"
    fi
    
    docker buildx build \
        --target "${target}" \
        --platform "${platforms}" \
        --tag "${full_tag}" \
        --tag "${latest_tag}" \
        ${BUILD_ARGS} \
        ${push_args} \
        --progress=plain \
        --cache-from="type=registry,ref=${REGISTRY}/${IMAGE_NAME}:cache-${tag_suffix}" \
        --cache-to="type=registry,ref=${REGISTRY}/${IMAGE_NAME}:cache-${tag_suffix},mode=max" \
        .
    
    log "✅ Successfully built ${target} target"
}

# Function to run tests in Docker
run_tests() {
    log "Running tests in Docker container"
    
    docker buildx build \
        --target testing \
        --platform linux/amd64 \
        --tag "${IMAGE_NAME}:testing" \
        ${BUILD_ARGS} \
        --load \
        .
    
    # Create directories for test results
    mkdir -p test-results coverage
    
    # Run tests and copy results
    docker run --rm \
        -v "$(pwd)/test-results:/app/test-results" \
        -v "$(pwd)/coverage:/app/coverage" \
        "${IMAGE_NAME}:testing" \
        pytest -v \
            --cov=its_camera_ai \
            --cov-report=html:/app/coverage \
            --cov-report=xml:/app/test-results/coverage.xml \
            --junit-xml=/app/test-results/junit.xml \
            --cov-fail-under=90
    
    log "✅ Tests completed successfully"
}

# Function to security scan
security_scan() {
    local image=$1
    
    log "Running security scan on ${image}"
    
    # Check if trivy is installed
    if ! command -v trivy &> /dev/null; then
        warn "Trivy not installed, skipping security scan"
        warn "Install with: brew install aquasecurity/trivy/trivy"
        return 0
    fi
    
    trivy image \
        --severity HIGH,CRITICAL \
        --format json \
        --output "security-report-$(date +%Y%m%d).json" \
        "${image}"
    
    log "✅ Security scan completed"
}

# Function to show image sizes
show_image_info() {
    log "Docker images built:"
    docker images "${IMAGE_NAME}" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}"
    echo ""
    
    # Show layer analysis if dive is available
    if command -v dive &> /dev/null; then
        info "Run 'dive ${IMAGE_NAME}:production' to analyze image layers"
    else
        info "Install 'dive' tool for detailed image layer analysis: brew install dive"
    fi
}

# Main build function
main() {
    local command=${1:-"help"}
    
    # Ensure buildx is available
    if ! docker buildx version &> /dev/null; then
        error "Docker buildx is required but not available"
    fi
    
    # Create builder if it doesn't exist
    docker buildx create --name its-builder --use --bootstrap 2>/dev/null || true
    
    case $command in
        "development" | "dev")
            log "Building development environment"
            build_target "development" "dev" "linux/amd64"
            ;;
        
        "gpu-dev")
            log "Building GPU development environment"
            if ! docker info | grep -q "nvidia"; then
                warn "NVIDIA Docker runtime not detected. GPU features may not work."
            fi
            build_target "gpu-development" "gpu-dev" "linux/amd64"
            ;;
        
        "production" | "prod")
            log "Building production image"
            build_target "production" "production" "${PLATFORM}"
            ;;
        
        "gpu-prod")
            log "Building GPU production image"
            build_target "gpu-production" "gpu-prod" "linux/amd64"
            ;;
        
        "edge")
            log "Building edge deployment image"
            build_target "edge" "edge" "${PLATFORM}"
            ;;
        
        "triton")
            log "Building Triton inference server"
            build_target "triton-inference" "triton" "linux/amd64"
            ;;
        
        "test")
            run_tests
            ;;
        
        "all")
            log "Building all targets"
            build_target "development" "dev" "linux/amd64"
            build_target "production" "production" "${PLATFORM}"
            build_target "edge" "edge" "${PLATFORM}"
            build_target "gpu-development" "gpu-dev" "linux/amd64"
            build_target "gpu-production" "gpu-prod" "linux/amd64"
            run_tests
            show_image_info
            ;;
        
        "push")
            log "Building and pushing all production images"
            build_target "production" "production" "${PLATFORM}" true
            build_target "edge" "edge" "${PLATFORM}" true
            build_target "gpu-production" "gpu-prod" "linux/amd64" true
            ;;
        
        "scan")
            local target_image=${2:-"${IMAGE_NAME}:production"}
            security_scan "$target_image"
            ;;
        
        "clean")
            log "Cleaning up Docker build cache"
            docker buildx prune -f
            docker system prune -f
            log "✅ Cleanup completed"
            ;;
        
        "help" | *)
            cat << EOF
${GREEN}ITS Camera AI Docker Build Script${NC}

Usage: $0 <command> [options]

Commands:
  ${BLUE}development, dev${NC}     - Build development environment
  ${BLUE}gpu-dev${NC}              - Build GPU development environment
  ${BLUE}production, prod${NC}     - Build production image
  ${BLUE}gpu-prod${NC}             - Build GPU production image
  ${BLUE}edge${NC}                 - Build edge deployment image
  ${BLUE}triton${NC}               - Build Triton inference server
  ${BLUE}test${NC}                 - Run tests in Docker
  ${BLUE}all${NC}                  - Build all targets
  ${BLUE}push${NC}                 - Build and push production images
  ${BLUE}scan [image]${NC}         - Security scan (requires trivy)
  ${BLUE}clean${NC}                - Clean Docker build cache
  ${BLUE}help${NC}                 - Show this help message

Environment variables:
  ${YELLOW}VERSION${NC}              - Image version (default: 0.1.0)
  ${YELLOW}PLATFORM${NC}             - Target platforms (default: linux/amd64,linux/arm64)
  ${YELLOW}REGISTRY${NC}             - Docker registry (default: ghcr.io/your-org)

Examples:
  $0 development                    # Build dev environment
  $0 production                     # Build production image
  VERSION=1.0.0 $0 push            # Build and push v1.0.0
  $0 scan its-camera-ai:production  # Security scan

Build targets available:
  • ${GREEN}development${NC}    - Full dev environment with hot reload
  • ${GREEN}gpu-development${NC} - GPU-enabled dev environment
  • ${GREEN}production${NC}     - Optimized production runtime
  • ${GREEN}gpu-production${NC}  - GPU-optimized production runtime
  • ${GREEN}edge${NC}           - Lightweight edge deployment
  • ${GREEN}triton-inference${NC} - High-performance GPU inference
  • ${GREEN}testing${NC}        - Testing environment with full test suite

Performance characteristics:
  🚀 Production: <100ms inference latency, optimized memory usage
  ⚡ GPU: Hardware acceleration for ML inference
  📱 Edge: Minimal footprint for resource-constrained devices
  🔬 Triton: High-throughput batch inference
EOF
            ;;
    esac
}

# Run main function with all arguments
main "$@"