#!/bin/bash

# ITS Camera AI - Frontend Deployment Script
# Handles both development and production deployment scenarios

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
ENVIRONMENT="development"
BUILD_ONLY=false
SKIP_DEPS=false
SKIP_SSL=false
CLEAN_BUILD=false
DETACHED=false

# Help function
show_help() {
    cat << EOF
ITS Camera AI - Frontend Deployment Script

Usage: $0 [OPTIONS]

OPTIONS:
    -e, --env ENVIRONMENT     Set environment (development|staging|production) [default: development]
    -b, --build-only          Build images only, don't start services
    -s, --skip-deps           Skip dependency installation
    --skip-ssl                Skip SSL certificate generation
    -c, --clean               Clean build (remove existing images and volumes)
    -d, --detached            Run in detached mode
    -h, --help                Show this help message

EXAMPLES:
    $0                        # Start development environment
    $0 -e production -d       # Start production environment in background
    $0 -b -c                  # Clean build without starting services
    $0 --skip-ssl             # Start without generating SSL certificates

ENVIRONMENT VARIABLES:
    DOCKER_BUILDKIT=1         Enable Docker BuildKit for faster builds
    COMPOSE_PARALLEL_LIMIT=4  Limit parallel container starts
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -b|--build-only)
            BUILD_ONLY=true
            shift
            ;;
        -s|--skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --skip-ssl)
            SKIP_SSL=true
            shift
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -d|--detached)
            DETACHED=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
    echo -e "${RED}Error: Environment must be one of: development, staging, production${NC}"
    exit 1
fi

echo -e "${BLUE}🚀 ITS Camera AI - Frontend Deployment${NC}"
echo -e "${BLUE}Environment: ${YELLOW}$ENVIRONMENT${NC}"
echo -e "${BLUE}Project Root: ${YELLOW}$PROJECT_ROOT${NC}"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Check prerequisites
echo -e "${BLUE}📋 Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed${NC}"
    exit 1
fi

# Determine docker-compose command
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

echo -e "${GREEN}✅ Docker and Docker Compose are available${NC}"

# Generate SSL certificates for development/staging
if [[ "$ENVIRONMENT" != "production" && "$SKIP_SSL" != true ]]; then
    echo -e "${BLUE}🔐 Generating SSL certificates for $ENVIRONMENT...${NC}"

    SSL_SCRIPT="$PROJECT_ROOT/config/nginx/ssl/generate-dev-certs.sh"
    if [[ -f "$SSL_SCRIPT" ]]; then
        bash "$SSL_SCRIPT"
    else
        echo -e "${YELLOW}⚠️  SSL generation script not found, skipping...${NC}"
    fi
fi

# Clean build if requested
if [[ "$CLEAN_BUILD" == true ]]; then
    echo -e "${BLUE}🧹 Cleaning existing containers and images...${NC}"

    # Stop and remove containers
    if [[ "$ENVIRONMENT" == "production" ]]; then
        $DOCKER_COMPOSE -f docker-compose.prod.yml down --remove-orphans --volumes || true
    else
        $DOCKER_COMPOSE down --remove-orphans --volumes || true
    fi

    # Remove images
    docker images | grep its-camera-ai | awk '{print $3}' | xargs -r docker rmi -f || true

    # Clean build cache
    docker builder prune -f || true

    echo -e "${GREEN}✅ Cleanup completed${NC}"
fi

# Install dependencies if not skipped
if [[ "$SKIP_DEPS" != true ]]; then
    echo -e "${BLUE}📦 Installing frontend dependencies...${NC}"

    cd "$PROJECT_ROOT/web"

    if [[ -f "yarn.lock" ]]; then
        yarn install --frozen-lockfile
    elif [[ -f "package-lock.json" ]]; then
        npm ci
    else
        npm install
    fi

    cd "$PROJECT_ROOT"
    echo -e "${GREEN}✅ Dependencies installed${NC}"
fi

# Set environment variables
export DOCKER_BUILDKIT=1
export COMPOSE_PARALLEL_LIMIT=4

# Choose compose file based on environment
COMPOSE_FILE=""
case $ENVIRONMENT in
    development)
        COMPOSE_FILE="docker-compose.yml"
        ;;
    staging)
        COMPOSE_FILE="docker-compose.yml"
        export NODE_ENV=staging
        ;;
    production)
        COMPOSE_FILE="docker-compose.prod.yml"
        export NODE_ENV=production
        ;;
esac

echo -e "${BLUE}🏗️  Building and starting services...${NC}"
echo -e "${BLUE}Using compose file: ${YELLOW}$COMPOSE_FILE${NC}"

# Build images
if [[ "$BUILD_ONLY" == true ]]; then
    echo -e "${BLUE}🔨 Building images only...${NC}"
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" build frontend

    if [[ "$ENVIRONMENT" == "production" ]]; then
        $DOCKER_COMPOSE -f "$COMPOSE_FILE" build nginx
    fi

    echo -e "${GREEN}✅ Build completed${NC}"
    exit 0
fi

# Start services
START_ARGS=""
if [[ "$DETACHED" == true ]]; then
    START_ARGS="-d"
fi

# For development, start just the frontend and its dependencies
if [[ "$ENVIRONMENT" == "development" ]]; then
    echo -e "${BLUE}🚀 Starting development environment...${NC}"
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" up $START_ARGS frontend
else
    echo -e "${BLUE}🚀 Starting $ENVIRONMENT environment...${NC}"
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" up $START_ARGS
fi

# Wait for services to be healthy
if [[ "$DETACHED" == true ]]; then
    echo -e "${BLUE}⏳ Waiting for services to be healthy...${NC}"

    # Wait for frontend health check
    for i in {1..30}; do
        if curl -f http://localhost:3000/api/health &>/dev/null; then
            echo -e "${GREEN}✅ Frontend is healthy${NC}"
            break
        fi

        if [[ $i -eq 30 ]]; then
            echo -e "${RED}❌ Frontend health check failed${NC}"
            exit 1
        fi

        echo -e "${YELLOW}⏳ Waiting for frontend... ($i/30)${NC}"
        sleep 2
    done
fi

# Show service status
echo -e "${BLUE}📊 Service Status:${NC}"
$DOCKER_COMPOSE -f "$COMPOSE_FILE" ps

# Show access URLs
echo ""
echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo ""
echo -e "${BLUE}📱 Access URLs:${NC}"

if [[ "$ENVIRONMENT" == "development" ]]; then
    echo -e "  ${GREEN}Frontend:${NC} http://localhost:3000"
    echo -e "  ${GREEN}API:${NC} http://localhost:8000"
    echo -e "  ${GREEN}Grafana:${NC} http://localhost:3001"
else
    echo -e "  ${GREEN}Application:${NC} http://localhost (HTTP)"
    echo -e "  ${GREEN}Application:${NC} https://localhost (HTTPS)"
fi

echo ""
echo -e "${BLUE}🛠️  Useful Commands:${NC}"
echo -e "  View logs: ${YELLOW}$DOCKER_COMPOSE -f $COMPOSE_FILE logs -f frontend${NC}"
echo -e "  Stop services: ${YELLOW}$DOCKER_COMPOSE -f $COMPOSE_FILE down${NC}"
echo -e "  Restart frontend: ${YELLOW}$DOCKER_COMPOSE -f $COMPOSE_FILE restart frontend${NC}"
echo -e "  Shell access: ${YELLOW}$DOCKER_COMPOSE -f $COMPOSE_FILE exec frontend sh${NC}"

if [[ "$ENVIRONMENT" == "development" ]]; then
    echo ""
    echo -e "${YELLOW}💡 Development Tips:${NC}"
    echo -e "  - Hot reload is enabled for code changes"
    echo -e "  - Check health: curl http://localhost:3000/api/health"
    echo -e "  - API docs: http://localhost:8000/docs"
fi