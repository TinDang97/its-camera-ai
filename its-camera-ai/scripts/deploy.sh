#!/bin/bash
# Task 5.4.4: Automated deployment script with Docker Compose
# Supports multiple environments, health checks, and rollback mechanisms

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_TIMEOUT=600  # 10 minutes

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Help function
show_help() {
    cat << EOF
🚀 ITS Camera AI Deployment Script

Usage: $0 [OPTIONS]

Options:
    -e, --environment ENVIRONMENT    Target environment (development|staging|production)
    -s, --strategy STRATEGY         Deployment strategy (rolling|blue-green|canary)
    -v, --version VERSION           Docker image version/tag
    -f, --force                     Force deployment (skip confirmations)
    --skip-backup                   Skip database backup
    --skip-tests                    Skip health checks
    --rollback                      Rollback to previous version
    --dry-run                       Show what would be deployed without executing
    -h, --help                      Show this help message

Examples:
    $0 -e development                           # Deploy to development
    $0 -e staging -s rolling -v latest          # Rolling deployment to staging
    $0 -e production -s blue-green -v v1.2.3    # Blue-green deployment to production
    $0 --rollback -e production                 # Rollback production deployment

Environment Variables:
    DOCKER_REGISTRY                 Docker registry URL (default: ghcr.io)
    IMAGE_NAME                      Docker image name (default: repository name)
    DEPLOYMENT_TIMEOUT              Deployment timeout in seconds (default: 600)

EOF
}

# Parse command line arguments
parse_arguments() {
    ENVIRONMENT=""
    STRATEGY="rolling"
    VERSION="latest"
    FORCE=false
    SKIP_BACKUP=false
    SKIP_TESTS=false
    ROLLBACK=false
    DRY_RUN=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -s|--strategy)
                STRATEGY="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            --skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --rollback)
                ROLLBACK=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Validate required arguments
    if [[ -z "$ENVIRONMENT" ]]; then
        log_error "Environment is required. Use -e or --environment"
        show_help
        exit 1
    fi

    # Validate environment
    if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
        log_error "Invalid environment: $ENVIRONMENT. Must be development, staging, or production"
        exit 1
    fi

    # Validate strategy
    if [[ ! "$STRATEGY" =~ ^(rolling|blue-green|canary)$ ]]; then
        log_error "Invalid strategy: $STRATEGY. Must be rolling, blue-green, or canary"
        exit 1
    fi

    # Force blue-green for production
    if [[ "$ENVIRONMENT" == "production" && "$STRATEGY" != "blue-green" ]]; then
        log_warning "Forcing blue-green strategy for production environment"
        STRATEGY="blue-green"
    fi
}

# Validate prerequisites
validate_prerequisites() {
    log_info "Validating deployment prerequisites..."

    # Check Docker and Docker Compose
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi

    # Check if Docker daemon is running
    if ! docker ps &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi

    # Check environment-specific requirements
    case "$ENVIRONMENT" in
        production)
            if [[ "$FORCE" != true ]]; then
                read -p "⚠️  You are about to deploy to PRODUCTION. Are you sure? (yes/no): " confirm
                if [[ "$confirm" != "yes" ]]; then
                    log_info "Deployment cancelled"
                    exit 0
                fi
            fi
            ;;
    esac

    log_success "Prerequisites validation completed"
}

# Set up environment variables
setup_environment() {
    log_info "Setting up environment variables for $ENVIRONMENT..."

    # Set default values
    export DOCKER_REGISTRY="${DOCKER_REGISTRY:-ghcr.io}"
    export IMAGE_NAME="${IMAGE_NAME:-$(basename "$(git remote get-url origin)" .git)}"
    export COMPOSE_PROJECT_NAME="its-camera-ai-$ENVIRONMENT"
    export COMPOSE_FILE=""

    # Environment-specific configuration
    case "$ENVIRONMENT" in
        development)
            export COMPOSE_FILE="docker/docker-compose.yml:docker/docker-compose.dev.yml"
            export APP_PORT=8000
            export LOG_LEVEL=DEBUG
            ;;
        staging)
            export COMPOSE_FILE="docker/docker-compose.yml:docker/docker-compose.prod.yml"
            export APP_PORT=8000
            export LOG_LEVEL=INFO
            ;;
        production)
            export COMPOSE_FILE="docker/docker-compose.yml:docker/docker-compose.prod.yml"
            export APP_PORT=8000
            export LOG_LEVEL=WARNING
            ;;
    esac

    # Load environment-specific variables
    ENV_FILE="$PROJECT_ROOT/.env.$ENVIRONMENT"
    if [[ -f "$ENV_FILE" ]]; then
        log_info "Loading environment variables from $ENV_FILE"
        set -a
        source "$ENV_FILE"
        set +a
    else
        log_warning "Environment file $ENV_FILE not found"
    fi

    log_success "Environment setup completed"
}

# Create database backup
create_backup() {
    if [[ "$SKIP_BACKUP" == true || "$ENVIRONMENT" == "development" ]]; then
        log_info "Skipping database backup"
        return 0
    fi

    log_info "Creating database backup for $ENVIRONMENT..."

    BACKUP_DIR="$PROJECT_ROOT/backups/$ENVIRONMENT"
    BACKUP_FILE="$BACKUP_DIR/backup-$(date +%Y%m%d-%H%M%S).sql"

    mkdir -p "$BACKUP_DIR"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would create backup: $BACKUP_FILE"
        return 0
    fi

    # Create database backup (adjust connection parameters as needed)
    if docker-compose exec -T postgres pg_dump -U its_user its_camera_ai > "$BACKUP_FILE"; then
        log_success "Database backup created: $BACKUP_FILE"

        # Keep only last 10 backups
        ls -t "$BACKUP_DIR"/backup-*.sql | tail -n +11 | xargs -r rm
    else
        log_error "Failed to create database backup"
        exit 1
    fi
}

# Pull latest images
pull_images() {
    log_info "Pulling latest Docker images..."

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would pull images with version: $VERSION"
        return 0
    fi

    if ! docker-compose pull; then
        log_error "Failed to pull Docker images"
        exit 1
    fi

    log_success "Docker images pulled successfully"
}

# Rolling deployment
deploy_rolling() {
    log_info "Starting rolling deployment..."

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would perform rolling deployment"
        return 0
    fi

    # Graceful rolling update
    if ! docker-compose up -d --remove-orphans; then
        log_error "Rolling deployment failed"
        exit 1
    fi

    log_success "Rolling deployment completed"
}

# Blue-green deployment
deploy_blue_green() {
    log_info "Starting blue-green deployment..."

    # Determine current and next environments
    CURRENT_COLOR=$(docker-compose ps -q app 2>/dev/null | head -1 | xargs docker inspect --format '{{.Config.Labels.deployment_color}}' 2>/dev/null || echo "blue")
    NEXT_COLOR=$([ "$CURRENT_COLOR" = "blue" ] && echo "green" || echo "blue")

    log_info "Current deployment: $CURRENT_COLOR"
    log_info "Deploying to: $NEXT_COLOR"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would deploy to $NEXT_COLOR environment"
        return 0
    fi

    # Set deployment color
    export DEPLOYMENT_COLOR="$NEXT_COLOR"

    # Deploy to next environment
    if ! docker-compose -p "$COMPOSE_PROJECT_NAME-$NEXT_COLOR" up -d --remove-orphans; then
        log_error "Blue-green deployment to $NEXT_COLOR failed"
        exit 1
    fi

    # Wait for new environment to be healthy
    log_info "Waiting for new environment to be healthy..."
    wait_for_health "$COMPOSE_PROJECT_NAME-$NEXT_COLOR"

    # Switch traffic (this would typically involve load balancer reconfiguration)
    log_info "Switching traffic to $NEXT_COLOR environment"

    # Update main project to point to new environment
    docker-compose -p "$COMPOSE_PROJECT_NAME" down
    docker-compose -p "$COMPOSE_PROJECT_NAME-$NEXT_COLOR" up -d

    # Cleanup old environment
    log_info "Cleaning up $CURRENT_COLOR environment"
    docker-compose -p "$COMPOSE_PROJECT_NAME-$CURRENT_COLOR" down --remove-orphans

    log_success "Blue-green deployment completed"
}

# Canary deployment
deploy_canary() {
    log_info "Starting canary deployment..."

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would perform canary deployment"
        return 0
    fi

    # Deploy canary with reduced replicas
    export CANARY_REPLICAS=1
    export PRODUCTION_REPLICAS=3

    # Start canary deployment
    if ! docker-compose -f docker-compose.canary.yml up -d; then
        log_error "Canary deployment failed"
        exit 1
    fi

    log_info "Canary deployment started. Monitor metrics and promote when ready."
    log_success "Canary deployment completed"
}

# Wait for service health
wait_for_health() {
    local project_name="${1:-$COMPOSE_PROJECT_NAME}"
    local max_attempts=30
    local attempt=1

    log_info "Waiting for services to be healthy..."

    while [[ $attempt -le $max_attempts ]]; do
        if docker-compose -p "$project_name" ps | grep -q "Up (healthy)"; then
            log_success "Services are healthy"
            return 0
        fi

        log_info "Attempt $attempt/$max_attempts - waiting for services..."
        sleep 20
        ((attempt++))
    done

    log_error "Services failed to become healthy within timeout"
    return 1
}

# Run health checks
run_health_checks() {
    if [[ "$SKIP_TESTS" == true ]]; then
        log_info "Skipping health checks"
        return 0
    fi

    log_info "Running health checks..."

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would run health checks"
        return 0
    fi

    # Wait for services to be ready
    wait_for_health

    # API health check
    local api_url="http://localhost:${APP_PORT}/health"
    local max_attempts=10
    local attempt=1

    while [[ $attempt -le $max_attempts ]]; do
        if curl -f "$api_url" &> /dev/null; then
            log_success "API health check passed"
            break
        fi

        log_info "API health check attempt $attempt/$max_attempts"
        sleep 10
        ((attempt++))
    done

    if [[ $attempt -gt $max_attempts ]]; then
        log_error "API health check failed"
        return 1
    fi

    # Additional environment-specific health checks
    case "$ENVIRONMENT" in
        staging|production)
            # Database connectivity check
            if ! docker-compose exec -T app python -c "
import asyncio
from src.its_camera_ai.database import database
async def test_db():
    await database.connect()
    await database.disconnect()
    print('Database connectivity: OK')
asyncio.run(test_db())
"; then
                log_error "Database connectivity check failed"
                return 1
            fi

            # Redis connectivity check
            if ! docker-compose exec -T app python -c "
import redis
r = redis.from_url('${REDIS_URL}')
r.ping()
print('Redis connectivity: OK')
"; then
                log_error "Redis connectivity check failed"
                return 1
            fi
            ;;
    esac

    log_success "All health checks passed"
}

# Rollback deployment
rollback_deployment() {
    log_info "Starting deployment rollback for $ENVIRONMENT..."

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would rollback deployment"
        return 0
    fi

    # Get previous deployment
    BACKUP_DIR="$PROJECT_ROOT/backups/$ENVIRONMENT"
    LATEST_BACKUP=$(ls -t "$BACKUP_DIR"/backup-*.sql 2>/dev/null | head -1)

    if [[ -n "$LATEST_BACKUP" ]]; then
        log_info "Rolling back to previous version with backup: $LATEST_BACKUP"

        # Stop current deployment
        docker-compose down

        # Restore database backup
        if [[ -f "$LATEST_BACKUP" ]]; then
            log_info "Restoring database from backup..."
            docker-compose exec -T postgres psql -U its_user its_camera_ai < "$LATEST_BACKUP"
        fi

        # Start previous deployment
        docker-compose up -d

        log_success "Rollback completed"
    else
        log_error "No backup found for rollback"
        exit 1
    fi
}

# Main deployment function
main() {
    log_info "Starting deployment for ITS Camera AI"
    log_info "Environment: $ENVIRONMENT"
    log_info "Strategy: $STRATEGY"
    log_info "Version: $VERSION"

    if [[ "$DRY_RUN" == true ]]; then
        log_warning "DRY RUN MODE - No actual changes will be made"
    fi

    cd "$PROJECT_ROOT"

    validate_prerequisites
    setup_environment

    if [[ "$ROLLBACK" == true ]]; then
        rollback_deployment
        exit 0
    fi

    create_backup
    pull_images

    # Execute deployment strategy
    case "$STRATEGY" in
        rolling)
            deploy_rolling
            ;;
        blue-green)
            deploy_blue_green
            ;;
        canary)
            deploy_canary
            ;;
    esac

    run_health_checks

    log_success "🎉 Deployment completed successfully!"
    log_info "Environment: $ENVIRONMENT"
    log_info "Version: $VERSION"
    log_info "Strategy: $STRATEGY"

    # Show service status
    docker-compose ps
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    parse_arguments "$@"
    main
fi