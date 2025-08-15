#!/bin/bash

# ITS Camera AI - One-Command Quickstart Script
# Complete setup and verification for development and production environments

set -euo pipefail

# Configuration
DEFAULT_ENV="development"
ENABLE_GPU=false
ENABLE_MONITORING=false
ENABLE_ML=false
SKIP_DEPS=false
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_usage() {
    cat << EOF
ITS Camera AI - Quickstart Script

Usage: $0 [OPTIONS]

OPTIONS:
    --env ENV               Environment (development|production) [default: development]
    --gpu                   Enable GPU support and dependencies
    --monitoring            Enable monitoring stack (Prometheus, Grafana)
    --ml                    Enable ML dependencies and model downloads
    --skip-deps             Skip dependency installation
    --verbose               Enable verbose output
    -h, --help              Show this help message

EXAMPLES:
    $0                                  # Basic development setup
    $0 --env production --gpu           # Production with GPU support
    $0 --monitoring --ml                # Development with monitoring and ML
    $0 --env production --gpu --monitoring --ml  # Full production setup

EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --env)
                ENV="$2"
                shift 2
                ;;
            --gpu)
                ENABLE_GPU=true
                shift
                ;;
            --monitoring)
                ENABLE_MONITORING=true
                shift
                ;;
            --ml)
                ENABLE_ML=true
                shift
                ;;
            --skip-deps)
                SKIP_DEPS=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                set -x
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    ENV=${ENV:-$DEFAULT_ENV}
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found. Please install Python 3.12+"
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ "$(printf '%s\n' "3.12" "$PYTHON_VERSION" | sort -V | head -n1)" != "3.12" ]]; then
        log_error "Python 3.12+ required. Found: $PYTHON_VERSION"
        exit 1
    fi

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker"
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose not found. Please install Docker Compose"
        exit 1
    fi

    # Check uv package manager
    if ! command -v uv &> /dev/null; then
        log_warning "uv not found. Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source ~/.bashrc || source ~/.zshrc || true
        export PATH="$HOME/.cargo/bin:$PATH"
    fi

    # Check NVIDIA drivers if GPU enabled
    if [[ "$ENABLE_GPU" == true ]]; then
        if command -v nvidia-smi &> /dev/null; then
            log_success "NVIDIA drivers detected: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)"
        else
            log_warning "NVIDIA drivers not found. GPU features will be disabled."
            ENABLE_GPU=false
        fi
    fi

    log_success "Prerequisites check passed"
}

setup_environment() {
    log_info "Setting up environment for: $ENV"

    # Create .env file if it doesn't exist
    if [[ ! -f .env ]]; then
        if [[ -f .env.example ]]; then
            cp .env.example .env
            log_success "Created .env from .env.example"
        else
            log_info "Creating basic .env file"
            cat > .env << EOF
# ITS Camera AI Configuration
ENVIRONMENT=$ENV
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Database Configuration
DATABASE_URL=postgresql+asyncpg://its_camera_ai:dev_password@localhost:5432/its_camera_ai
REDIS_URL=redis://localhost:6379/0

# MinIO Configuration
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
MINIO_BUCKET=its-camera-ai

# Security
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
JWT_EXPIRATION_MINUTES=60

# ML Configuration
MODEL_CONFIDENCE_THRESHOLD=0.5
BATCH_SIZE=8
GPU_MEMORY_FRACTION=0.8
EOF
        fi
    fi

    # Update environment-specific settings
    if [[ "$ENV" == "production" ]]; then
        sed -i.bak 's/LOG_LEVEL=INFO/LOG_LEVEL=WARNING/' .env
        sed -i.bak 's/JWT_EXPIRATION_MINUTES=60/JWT_EXPIRATION_MINUTES=30/' .env
        log_info "Updated .env for production settings"
    fi

    log_success "Environment configuration completed"
}

install_dependencies() {
    if [[ "$SKIP_DEPS" == true ]]; then
        log_info "Skipping dependency installation"
        return 0
    fi

    log_info "Installing Python dependencies..."

    # Build dependency groups
    GROUPS="dev"

    if [[ "$ENABLE_ML" == true ]]; then
        GROUPS="$GROUPS ml"
    fi

    if [[ "$ENABLE_GPU" == true ]]; then
        GROUPS="$GROUPS gpu"
    fi

    # Convert to uv sync format
    UV_GROUPS=""
    for group in $GROUPS; do
        UV_GROUPS="$UV_GROUPS --group $group"
    done

    log_info "Installing dependency groups: $GROUPS"
    uv sync $UV_GROUPS

    log_success "Dependencies installed successfully"
}

setup_services() {
    log_info "Starting required services..."

    # Determine which compose file to use
    COMPOSE_FILE="docker-compose.yml"
    if [[ "$ENV" == "production" ]]; then
        COMPOSE_FILE="docker-compose.prod.yml"
    fi

    # Start core services first
    log_info "Starting database and cache services..."
    if [[ "$ENV" == "production" ]]; then
        docker-compose -f $COMPOSE_FILE up -d postgres redis-master redis-replica zookeeper
    else
        docker-compose -f $COMPOSE_FILE up -d postgres redis zookeeper
    fi

    # Wait for database to be ready
    log_info "Waiting for PostgreSQL to be ready..."
    sleep 30

    # Start remaining services
    log_info "Starting remaining services..."
    if [[ "$ENABLE_MONITORING" == true ]]; then
        docker-compose -f $COMPOSE_FILE up -d
    else
        # Start without monitoring services
        if [[ "$ENV" == "production" ]]; then
            docker-compose -f $COMPOSE_FILE up -d postgres redis-master redis-replica minio1 minio2 kafka1 kafka2
        else
            docker-compose -f $COMPOSE_FILE up -d postgres redis minio kafka
        fi
    fi

    log_success "Services started successfully"
}

run_migrations() {
    log_info "Running database migrations..."

    # Wait for database to be fully ready
    sleep 15

    # Run migrations
    if command -v alembic &> /dev/null; then
        alembic upgrade head
        log_success "Database migrations completed"
    else
        log_warning "Alembic not found. Run 'alembic upgrade head' manually after installation"
    fi
}

download_models() {
    if [[ "$ENABLE_ML" == true ]]; then
        log_info "Downloading ML models..."

        # Create models directory
        mkdir -p models

        # Download YOLO11 models if they don't exist
        if [[ ! -f models/yolo11n.pt ]]; then
            log_info "Downloading YOLO11 nano model..."
            python3 -c "
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.save('models/yolo11n.pt')
print('YOLO11 nano model downloaded')
" 2>/dev/null || log_warning "Failed to download YOLO11 model"
        fi

        log_success "Model download completed"
    fi
}

verify_setup() {
    log_info "Verifying setup..."

    # Check service health
    log_info "Checking service health..."

    # Check database
    if docker-compose ps postgres | grep -q "Up"; then
        log_success "PostgreSQL is running"
    else
        log_error "PostgreSQL is not running"
    fi

    # Check Redis
    if docker-compose ps redis | grep -q "Up" || docker-compose ps redis-master | grep -q "Up"; then
        log_success "Redis is running"
    else
        log_error "Redis is not running"
    fi

    # Test database connection
    log_info "Testing database connection..."
    python3 -c "
import asyncio
import asyncpg
import os
from dotenv import load_dotenv

async def test_db():
    load_dotenv()
    db_url = os.getenv('DATABASE_URL', 'postgresql://its_camera_ai:dev_password@localhost:5432/its_camera_ai')
    # Convert asyncpg URL format
    db_url = db_url.replace('postgresql+asyncpg://', 'postgresql://')
    try:
        conn = await asyncpg.connect(db_url)
        result = await conn.fetchval('SELECT 1')
        await conn.close()
        print('✅ Database connection successful')
        return True
    except Exception as e:
        print(f'❌ Database connection failed: {e}')
        return False

if asyncio.run(test_db()):
    exit(0)
else:
    exit(1)
" 2>/dev/null && log_success "Database connection verified" || log_warning "Database connection failed"

    # Check API if running
    if curl -s http://localhost:8000/health &> /dev/null; then
        log_success "API is responding"
    else
        log_info "API not running (start with: uvicorn its_camera_ai.api.app:app --reload)"
    fi

    log_success "Setup verification completed"
}

show_next_steps() {
    echo
    log_success "🎉 ITS Camera AI setup completed successfully!"
    echo
    log_info "Next steps:"
    echo
    echo "  1. Start the application:"
    echo "     uvicorn its_camera_ai.api.app:app --reload --host 0.0.0.0 --port 8000"
    echo
    echo "  2. Access the services:"
    echo "     • API Documentation: http://localhost:8000/docs"
    echo "     • MinIO Console: http://localhost:9000 (minioadmin/minioadmin123)"

    if [[ "$ENABLE_MONITORING" == true ]]; then
        echo "     • Grafana Dashboard: http://localhost:3000 (admin/admin123)"
        echo "     • Prometheus Metrics: http://localhost:9090"
    fi

    echo
    echo "  3. Use the CLI:"
    echo "     its-camera-ai dashboard"
    echo "     its-camera-ai services status"

    if [[ "$ENABLE_ML" == true ]]; then
        echo "     its-camera-ai ml deploy --model-path ./models/yolo11n.pt"
    fi

    echo
    echo "  4. Run tests:"
    echo "     pytest --cov=src/its_camera_ai --cov-report=html"
    echo
    echo "  5. View logs:"
    echo "     docker-compose logs -f"
    echo

    if [[ "$ENV" == "production" ]]; then
        log_warning "Production environment configured. Remember to:"
        echo "     • Update security credentials in .env"
        echo "     • Configure SSL/TLS certificates"
        echo "     • Set up backup procedures"
        echo "     • Configure monitoring alerts"
    fi

    echo
    log_info "For more information, see README.md and CLAUDE.md"
}

main() {
    echo "🚀 ITS Camera AI - Quickstart Setup"
    echo "=================================="
    echo

    parse_args "$@"

    log_info "Configuration:"
    echo "  Environment: $ENV"
    echo "  GPU Support: $ENABLE_GPU"
    echo "  Monitoring: $ENABLE_MONITORING"
    echo "  ML Models: $ENABLE_ML"
    echo

    check_prerequisites
    setup_environment
    install_dependencies
    setup_services
    run_migrations
    download_models
    verify_setup
    show_next_steps
}

# Handle script interruption
trap 'log_error "Script interrupted"; exit 1' INT TERM

# Run main function
main "$@"
