#!/bin/bash

# ITS Camera AI - Docker Setup Validation Script
# Validates the complete Docker deployment configuration

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

# Test configuration
ENVIRONMENT=${1:-development}
TIMEOUT=120

echo -e "${BLUE}🧪 ITS Camera AI - Docker Setup Validation${NC}"
echo -e "${BLUE}Environment: ${YELLOW}$ENVIRONMENT${NC}"
echo -e "${BLUE}Project Root: ${YELLOW}$PROJECT_ROOT${NC}"
echo ""

cd "$PROJECT_ROOT"

# Validation functions
validate_docker() {
    echo -e "${BLUE}📋 Validating Docker installation...${NC}"

    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker is not installed${NC}"
        return 1
    fi

    if ! docker version &> /dev/null; then
        echo -e "${RED}❌ Docker daemon is not running${NC}"
        return 1
    fi

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        echo -e "${RED}❌ Docker Compose is not available${NC}"
        return 1
    fi

    echo -e "${GREEN}✅ Docker and Docker Compose are available${NC}"
    return 0
}

validate_files() {
    echo -e "${BLUE}📁 Validating required files...${NC}"

    local required_files=(
        "docker-compose.yml"
        "web/Dockerfile"
        "web/package.json"
        "config/nginx/nginx.conf"
        "scripts/deploy-frontend.sh"
        "web/app/api/health/route.ts"
    )

    local missing_files=()

    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            missing_files+=("$file")
        fi
    done

    if [[ ${#missing_files[@]} -gt 0 ]]; then
        echo -e "${RED}❌ Missing required files:${NC}"
        for file in "${missing_files[@]}"; do
            echo -e "  ${RED}- $file${NC}"
        done
        return 1
    fi

    echo -e "${GREEN}✅ All required files are present${NC}"
    return 0
}

validate_dockerfile() {
    echo -e "${BLUE}🐳 Validating Dockerfile...${NC}"

    # Check if Dockerfile has required stages
    local required_stages=("base" "development" "production")
    local dockerfile="web/Dockerfile"

    for stage in "${required_stages[@]}"; do
        if ! grep -q "FROM .* AS $stage" "$dockerfile"; then
            echo -e "${RED}❌ Missing stage '$stage' in Dockerfile${NC}"
            return 1
        fi
    done

    # Check for security best practices
    if ! grep -q "USER nextjs" "$dockerfile"; then
        echo -e "${YELLOW}⚠️  Warning: Dockerfile should run as non-root user${NC}"
    fi

    if ! grep -q "HEALTHCHECK" "$dockerfile"; then
        echo -e "${YELLOW}⚠️  Warning: Dockerfile should include health checks${NC}"
    fi

    echo -e "${GREEN}✅ Dockerfile validation passed${NC}"
    return 0
}

validate_compose_file() {
    echo -e "${BLUE}📝 Validating Docker Compose configuration...${NC}"

    local compose_file=""
    case $ENVIRONMENT in
        development)
            compose_file="docker-compose.yml"
            ;;
        production)
            compose_file="docker-compose.prod.yml"
            ;;
        *)
            compose_file="docker-compose.yml"
            ;;
    esac

    # Check if compose file is valid YAML
    if ! docker-compose -f "$compose_file" config > /dev/null 2>&1; then
        echo -e "${RED}❌ Invalid Docker Compose configuration${NC}"
        return 1
    fi

    # Check for required services
    local required_services=("frontend")
    if [[ "$ENVIRONMENT" == "production" ]]; then
        required_services+=("nginx")
    fi

    for service in "${required_services[@]}"; do
        if ! docker-compose -f "$compose_file" config | grep -q "^  $service:"; then
            echo -e "${RED}❌ Missing service '$service' in compose file${NC}"
            return 1
        fi
    done

    echo -e "${GREEN}✅ Docker Compose configuration is valid${NC}"
    return 0
}

build_images() {
    echo -e "${BLUE}🏗️  Building Docker images...${NC}"

    local compose_file=""
    case $ENVIRONMENT in
        development)
            compose_file="docker-compose.yml"
            ;;
        production)
            compose_file="docker-compose.prod.yml"
            ;;
        *)
            compose_file="docker-compose.yml"
            ;;
    esac

    # Build frontend image
    if ! docker-compose -f "$compose_file" build frontend; then
        echo -e "${RED}❌ Failed to build frontend image${NC}"
        return 1
    fi

    echo -e "${GREEN}✅ Images built successfully${NC}"
    return 0
}

test_health_endpoint() {
    echo -e "${BLUE}🏥 Testing health endpoint...${NC}"

    # Start frontend service temporarily
    local compose_file=""
    case $ENVIRONMENT in
        development)
            compose_file="docker-compose.yml"
            ;;
        production)
            compose_file="docker-compose.prod.yml"
            ;;
        *)
            compose_file="docker-compose.yml"
            ;;
    esac

    echo -e "${BLUE}Starting frontend service for health check...${NC}"
    docker-compose -f "$compose_file" up -d frontend

    # Wait for service to be ready
    local max_attempts=30
    local attempt=1

    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s http://localhost:3000/api/health > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Health endpoint is responding${NC}"

            # Test health endpoint response format
            local health_response=$(curl -s http://localhost:3000/api/health)

            if echo "$health_response" | jq -e '.status' > /dev/null 2>&1; then
                echo -e "${GREEN}✅ Health endpoint returns valid JSON${NC}"
            else
                echo -e "${YELLOW}⚠️  Health endpoint response is not valid JSON${NC}"
            fi

            # Stop the service
            docker-compose -f "$compose_file" down
            return 0
        fi

        echo -e "${YELLOW}⏳ Waiting for health endpoint... ($attempt/$max_attempts)${NC}"
        sleep 5
        ((attempt++))
    done

    echo -e "${RED}❌ Health endpoint failed to respond${NC}"
    docker-compose -f "$compose_file" logs frontend
    docker-compose -f "$compose_file" down
    return 1
}

test_ssl_config() {
    echo -e "${BLUE}🔐 Testing SSL configuration...${NC}"

    if [[ "$ENVIRONMENT" == "development" ]]; then
        # Check if SSL generation script exists and is executable
        local ssl_script="config/nginx/ssl/generate-dev-certs.sh"

        if [[ ! -x "$ssl_script" ]]; then
            echo -e "${YELLOW}⚠️  SSL generation script is not executable${NC}"
            return 0
        fi

        # Test SSL certificate generation
        if bash "$ssl_script" <<< "y" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ SSL certificates can be generated${NC}"
        else
            echo -e "${YELLOW}⚠️  SSL certificate generation failed${NC}"
        fi
    fi

    # Check nginx configuration
    if [[ -f "config/nginx/nginx.conf" ]]; then
        # Basic syntax check (if nginx is available)
        if command -v nginx &> /dev/null; then
            if nginx -t -c "$(pwd)/config/nginx/nginx.conf" 2>/dev/null; then
                echo -e "${GREEN}✅ Nginx configuration is valid${NC}"
            else
                echo -e "${YELLOW}⚠️  Nginx configuration has syntax errors${NC}"
            fi
        else
            echo -e "${BLUE}ℹ️  Nginx not available for configuration testing${NC}"
        fi
    fi

    return 0
}

run_comprehensive_test() {
    echo -e "${BLUE}🚀 Running comprehensive deployment test...${NC}"

    local deployment_script="scripts/deploy-frontend.sh"

    if [[ ! -x "$deployment_script" ]]; then
        echo -e "${RED}❌ Deployment script is not executable${NC}"
        return 1
    fi

    echo -e "${BLUE}Testing deployment script (build only)...${NC}"

    if bash "$deployment_script" -e "$ENVIRONMENT" --build-only; then
        echo -e "${GREEN}✅ Deployment script executed successfully${NC}"
    else
        echo -e "${RED}❌ Deployment script failed${NC}"
        return 1
    fi

    return 0
}

# Main validation sequence
main() {
    local exit_code=0

    echo -e "${BLUE}Starting validation process...${NC}"
    echo ""

    # Run all validations
    validate_docker || exit_code=1
    echo ""

    validate_files || exit_code=1
    echo ""

    validate_dockerfile || exit_code=1
    echo ""

    validate_compose_file || exit_code=1
    echo ""

    build_images || exit_code=1
    echo ""

    test_health_endpoint || exit_code=1
    echo ""

    test_ssl_config || exit_code=1
    echo ""

    run_comprehensive_test || exit_code=1
    echo ""

    # Final result
    if [[ $exit_code -eq 0 ]]; then
        echo -e "${GREEN}🎉 All validations passed! Docker setup is ready for deployment.${NC}"
        echo ""
        echo -e "${BLUE}📋 Next steps:${NC}"
        echo -e "  1. Configure environment variables in .env.docker"
        echo -e "  2. Run: ${YELLOW}./scripts/deploy-frontend.sh -e $ENVIRONMENT${NC}"
        echo -e "  3. Access application at http://localhost:3000"
    else
        echo -e "${RED}❌ Some validations failed. Please fix the issues above.${NC}"
    fi

    exit $exit_code
}

# Run main function
main