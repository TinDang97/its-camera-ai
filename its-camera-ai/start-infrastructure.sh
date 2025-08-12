#!/bin/bash

# ITS Camera AI - Infrastructure Startup Script
# Starts all validated infrastructure services

set -e

echo "========================================"
echo "  ITS Camera AI Infrastructure Startup  "
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    echo "Please start Docker and try again"
    exit 1
fi

# Function to check if a container is running
check_container() {
    if docker ps --format '{{.Names}}' | grep -q "^$1$"; then
        echo -e "${GREEN}✓${NC} $1 is running"
        return 0
    else
        return 1
    fi
}

# Function to start a service
start_service() {
    local service_name=$1
    local docker_command=$2

    if check_container "$service_name"; then
        echo -e "${YELLOW}→${NC} $service_name already running, skipping..."
    else
        echo -e "${YELLOW}→${NC} Starting $service_name..."
        eval "$docker_command"
        sleep 2
        if check_container "$service_name"; then
            echo -e "${GREEN}✓${NC} $service_name started successfully"
        else
            echo -e "${RED}✗${NC} Failed to start $service_name"
            exit 1
        fi
    fi
}

# Start services in order
echo "Starting Infrastructure Services..."
echo ""

# 1. PostgreSQL
start_service "its-postgres" "docker run -d --name its-postgres \
    -e POSTGRES_DB=its_camera_ai \
    -e POSTGRES_USER=postgres \
    -e POSTGRES_PASSWORD=postgres_password \
    -p 5432:5432 \
    postgres:15-alpine"

# 2. TimescaleDB
start_service "its-timescaledb" "docker run -d --name its-timescaledb \
    -e POSTGRES_DB=its_metrics \
    -e POSTGRES_USER=postgres \
    -e POSTGRES_PASSWORD=timescale_password \
    -e TIMESCALEDB_TELEMETRY=off \
    -p 5433:5432 \
    timescale/timescaledb:latest-pg15"

# 3. Redis
start_service "its-redis" "docker run -d --name its-redis \
    -p 6379:6379 \
    redis:7-alpine redis-server --requirepass redis_password"

# 4. Zookeeper
start_service "its-zookeeper" "docker run -d --name its-zookeeper \
    -e ZOOKEEPER_CLIENT_PORT=2181 \
    -e ZOOKEEPER_TICK_TIME=2000 \
    -p 2181:2181 \
    confluentinc/cp-zookeeper:7.5.0"

# Wait for Zookeeper to be ready
echo -e "${YELLOW}→${NC} Waiting for Zookeeper to be ready..."
sleep 10

# 5. Kafka
start_service "its-kafka" "docker run -d --name its-kafka \
    -e KAFKA_BROKER_ID=1 \
    -e KAFKA_ZOOKEEPER_CONNECT=host.docker.internal:2181 \
    -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
    -e KAFKA_LISTENERS=PLAINTEXT://0.0.0.0:9092 \
    -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 \
    -e KAFKA_AUTO_CREATE_TOPICS_ENABLE=true \
    -p 9092:9092 \
    confluentinc/cp-kafka:7.5.0"

# 6. MinIO
start_service "its-minio" "docker run -d --name its-minio \
    -e MINIO_ROOT_USER=minioadmin \
    -e MINIO_ROOT_PASSWORD=minioadmin123 \
    -p 9000:9000 \
    -p 9001:9001 \
    minio/minio server /data --console-address ':9001'"

# 7. Prometheus
start_service "its-prometheus" "docker run -d --name its-prometheus \
    -p 9090:9090 \
    prom/prometheus:latest"

# 8. Grafana
start_service "its-grafana" "docker run -d --name its-grafana \
    -e GF_SECURITY_ADMIN_USER=admin \
    -e GF_SECURITY_ADMIN_PASSWORD=grafana_password \
    -p 3000:3000 \
    grafana/grafana:latest"

echo ""
echo "========================================"
echo "  Infrastructure Status                "
echo "========================================"
echo ""

# Show running containers
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep its- || true

echo ""
echo "========================================"
echo "  Access Points                        "
echo "========================================"
echo ""
echo "PostgreSQL:      localhost:5432 (postgres/postgres_password)"
echo "TimescaleDB:     localhost:5433 (postgres/timescale_password)"
echo "Redis:           localhost:6379 (password: redis_password)"
echo "Kafka:           localhost:9092"
echo "MinIO Console:   http://localhost:9001 (minioadmin/minioadmin123)"
echo "Prometheus:      http://localhost:9090"
echo "Grafana:         http://localhost:3000 (admin/grafana_password)"
echo ""
echo -e "${GREEN}✓ All infrastructure services are running!${NC}"
echo ""
echo "To stop all services, run:"
echo "  docker stop \$(docker ps -q --filter name=its-)"
echo ""
echo "To remove all services, run:"
echo "  docker rm -f \$(docker ps -aq --filter name=its-)"
echo ""
