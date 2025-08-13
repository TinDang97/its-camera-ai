#!/bin/bash
# Development Environment Startup Script for ITS Camera AI
# Platform Engineering: Automated development environment provisioning

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
DEV_ENV_FILE="${SCRIPT_DIR}/.env.dev"
COMPOSE_FILES="-f docker-compose.yml -f docker-compose.dev.yml"
PROFILE="dev"
LOG_FILE="${SCRIPT_DIR}/logs/startup-dev.log"

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

log "Starting ITS Camera AI Development Environment..."

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
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
    
    log "Prerequisites check passed ✓"
}

# Setup environment files
setup_environment() {
    log "Setting up environment files..."
    
    # Copy .env.example if .env doesn't exist
    if [[ ! -f "$ENV_FILE" ]]; then
        if [[ -f "${SCRIPT_DIR}/.env.example" ]]; then
            cp "${SCRIPT_DIR}/.env.example" "$ENV_FILE"
            log "Created .env file from .env.example"
        else
            warn "No .env.example found. Creating minimal .env file..."
            cat > "$ENV_FILE" << EOF
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
BUILD_TARGET=development
EOF
        fi
    fi
    
    # Create development-specific environment file
    if [[ ! -f "$DEV_ENV_FILE" ]]; then
        log "Creating development environment file..."
        cat > "$DEV_ENV_FILE" << EOF
# Development Environment Overrides
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
RELOAD=true
DEV_MODE=true

# Development ports
APP_PORT=8000
METRICS_PORT=8001
POSTGRES_PORT=5432
REDIS_PORT=6379
INFLUXDB_PORT=8086
KAFKA_PORT=9092
MINIO_API_PORT=9000
MINIO_CONSOLE_PORT=9001
ADMINER_PORT=8080
REDIS_COMMANDER_PORT=8081
MAILHOG_SMTP_PORT=1025
MAILHOG_UI_PORT=8025

# Development credentials
SECRET_KEY=dev-secret-key-change-in-production
POSTGRES_PASSWORD=dev_password
REDIS_PASSWORD=dev_redis_password
MINIO_ROOT_PASSWORD=DevMinIO2024Pass!

# Build settings
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse HEAD 2>/dev/null || echo 'unknown')
EOF
        log "Created development environment file"
    fi
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    local dirs=(
        "data/dev"
        "logs/dev"
        "models/dev"
        "temp"
        "backups"
        "data/postgres/dev"
        "data/redis/dev"
        "data/minio/dev"
        "cache"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "${SCRIPT_DIR}/$dir"
    done
    
    log "Directories created ✓"
}

# Pull latest images
pull_images() {
    log "Pulling latest Docker images..."
    docker compose $COMPOSE_FILES pull --ignore-pull-failures || warn "Some images failed to pull"
}

# Build custom images
build_images() {
    log "Building custom Docker images..."
    docker compose $COMPOSE_FILES build --parallel --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" --build-arg VCS_REF="$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
}

# Start services
start_services() {
    log "Starting development services..."
    
    # Start infrastructure services first
    info "Starting infrastructure services (databases, cache, etc.)..."
    docker compose $COMPOSE_FILES --profile "$PROFILE" up -d postgres redis minio timescaledb
    
    # Wait for infrastructure to be ready
    info "Waiting for infrastructure services to be ready..."
    sleep 15
    
    # Run MinIO initialization
    docker compose $COMPOSE_FILES --profile "$PROFILE" up --no-deps minio-init
    
    # Start main application
    info "Starting main application..."
    docker compose $COMPOSE_FILES --profile "$PROFILE" up -d app
    
    # Start development tools
    info "Starting development tools..."
    docker compose $COMPOSE_FILES --profile "$PROFILE" up -d adminer redis-commander mailhog
    
    # Start optional services
    if [[ "${START_KAFKA:-false}" == "true" ]]; then
        info "Starting Kafka services..."
        docker compose $COMPOSE_FILES --profile "$PROFILE" up -d zookeeper kafka
    fi
    
    if [[ "${START_JUPYTER:-false}" == "true" ]]; then
        info "Starting Jupyter service..."
        docker compose $COMPOSE_FILES --profile "$PROFILE" up -d jupyter
    fi
}

# Health checks
run_health_checks() {
    log "Running health checks..."
    
    local services=("app" "postgres" "redis" "minio")
    local failed_services=()
    
    for service in "${services[@]}"; do
        info "Checking health of $service..."
        if ! docker compose $COMPOSE_FILES ps --format json | jq -r --arg service "$service" '.[] | select(.Service == $service) | .Health' | grep -q "healthy"; then
            failed_services+=("$service")
        fi
    done
    
    if [[ ${#failed_services[@]} -gt 0 ]]; then
        error "The following services are not healthy: ${failed_services[*]}"
        warn "Check logs with: docker compose $COMPOSE_FILES logs <service_name>"
        return 1
    fi
    
    log "All services are healthy ✓"
}

# Display service information
show_service_info() {
    log "Development environment is ready! 🚀"
    echo
    info "=== Service URLs ==="
    echo "🔗 Main Application:     http://localhost:${APP_PORT:-8000}"
    echo "📊 Metrics:              http://localhost:${METRICS_PORT:-8001}/metrics"
    echo "🔍 API Documentation:    http://localhost:${APP_PORT:-8000}/docs"
    echo "🗄️  Database (Adminer):   http://localhost:${ADMINER_PORT:-8080}"
    echo "🔴 Redis Commander:      http://localhost:${REDIS_COMMANDER_PORT:-8081}"
    echo "📦 MinIO Console:        http://localhost:${MINIO_CONSOLE_PORT:-9001}"
    echo "📧 MailHog (Email):      http://localhost:${MAILHOG_UI_PORT:-8025}"
    if [[ "${START_JUPYTER:-false}" == "true" ]]; then
        echo "🐍 Jupyter Lab:          http://localhost:${JUPYTER_PORT:-8888}?token=jupyter-token-12345"
    fi
    echo
    info "=== Useful Commands ==="
    echo "📋 View logs:            docker compose $COMPOSE_FILES logs -f [service]"
    echo "🔧 Execute shell:        docker compose $COMPOSE_FILES exec app bash"
    echo "🧪 Run tests:            docker compose $COMPOSE_FILES exec app pytest"
    echo "🛑 Stop all:             docker compose $COMPOSE_FILES down"
    echo "🔄 Restart app:          docker compose $COMPOSE_FILES restart app"
    echo
    info "=== Development Features ==="
    echo "🔄 Hot reload enabled for Python files"
    echo "🐛 Debugger available on port 5678"
    echo "📝 Live configuration reloading"
    echo "🧪 Test database isolated from production"
    echo
}

# Cleanup function
cleanup() {
    if [[ "${1:-}" == "error" ]]; then
        error "Setup failed. Cleaning up..."
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
            --kafka)
                export START_KAFKA=true
                shift
                ;;
            --jupyter)
                export START_JUPYTER=true
                shift
                ;;
            --build)
                FORCE_BUILD=true
                shift
                ;;
            --pull)
                FORCE_PULL=true
                shift
                ;;
            --clean)
                log "Cleaning up existing containers..."
                docker compose $COMPOSE_FILES down --volumes --remove-orphans
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo
                echo "Options:"
                echo "  --kafka      Start Kafka services"
                echo "  --jupyter    Start Jupyter Lab"
                echo "  --build      Force rebuild of images"
                echo "  --pull       Force pull of base images"
                echo "  --clean      Clean up before starting"
                echo "  --help       Show this help message"
                exit 0
                ;;
            *)
                warn "Unknown option: $1"
                shift
                ;;
        esac
    done
    
    # Execute setup steps
    check_prerequisites
    setup_environment
    create_directories
    
    # Load environment files
    if [[ -f "$ENV_FILE" ]]; then
        set -a
        source "$ENV_FILE"
        set +a
    fi
    
    if [[ -f "$DEV_ENV_FILE" ]]; then
        set -a
        source "$DEV_ENV_FILE"
        set +a
    fi
    
    # Pull and build images if requested
    if [[ "${FORCE_PULL:-false}" == "true" ]]; then
        pull_images
    fi
    
    if [[ "${FORCE_BUILD:-false}" == "true" ]]; then
        build_images
    fi
    
    # Start services
    start_services
    
    # Wait a bit for services to initialize
    sleep 10
    
    # Run health checks
    if ! run_health_checks; then
        error "Health checks failed. Please check the logs."
        exit 1
    fi
    
    # Show service information
    show_service_info
    
    log "Development environment startup completed successfully!"
}

# Run main function
main "$@"