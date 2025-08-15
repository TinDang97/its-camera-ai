#!/bin/bash

# Deploy Citus Database Cluster for ITS Camera AI Production
# This script deploys PostgreSQL with Citus sharding, TimescaleDB, and PgBouncer

set -euo pipefail

# Configuration
CLUSTER_NAME="its-camera-ai-prod"
REGION="us-west-2"
NAMESPACE="postgresql-cluster"

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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi
    
    # Check cluster access
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot access Kubernetes cluster. Please check kubeconfig."
        exit 1
    fi
    
    # Check if cluster has required node groups
    if ! kubectl get nodes -l its-camera-ai/node-type=memory-workload &> /dev/null; then
        log_error "Memory-optimized nodes not found. Please ensure the cluster has memory-workload nodes."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Deploy namespace
deploy_namespace() {
    log_info "Creating PostgreSQL namespace..."
    kubectl apply -f k8s/data/postgresql/namespace.yaml
    log_success "Namespace created"
}

# Deploy Citus cluster
deploy_citus_cluster() {
    log_info "Deploying Citus PostgreSQL cluster..."
    
    # Apply Citus cluster configuration
    kubectl apply -f k8s/data/postgresql/citus-cluster.yaml
    
    # Wait for coordinator to be ready
    log_info "Waiting for Citus coordinator to be ready..."
    kubectl wait --for=condition=ready pod -l app=citus-coordinator -n $NAMESPACE --timeout=600s
    
    # Wait for workers to be ready
    log_info "Waiting for Citus workers to be ready..."
    kubectl wait --for=condition=ready pod -l app=citus-worker -n $NAMESPACE --timeout=600s
    
    log_success "Citus cluster deployed successfully"
}

# Initialize Citus cluster
initialize_citus() {
    log_info "Initializing Citus cluster with schema and data..."
    
    # Run initialization job
    kubectl apply -f k8s/data/postgresql/citus-init-job.yaml
    
    # Wait for initialization to complete
    log_info "Waiting for cluster initialization to complete..."
    kubectl wait --for=condition=complete job/citus-cluster-init -n $NAMESPACE --timeout=1200s
    
    # Show initialization logs
    log_info "Initialization job logs:"
    kubectl logs job/citus-cluster-init -n $NAMESPACE
    
    log_success "Citus cluster initialized successfully"
}

# Deploy PgBouncer
deploy_pgbouncer() {
    log_info "Deploying PgBouncer connection pooling..."
    
    kubectl apply -f k8s/data/postgresql/pgbouncer.yaml
    
    # Wait for PgBouncer to be ready
    log_info "Waiting for PgBouncer to be ready..."
    kubectl wait --for=condition=available deployment/pgbouncer -n $NAMESPACE --timeout=300s
    
    log_success "PgBouncer deployed successfully"
}

# Deploy monitoring
deploy_monitoring() {
    log_info "Deploying PostgreSQL monitoring..."
    
    kubectl apply -f k8s/data/postgresql/monitoring.yaml
    
    # Wait for exporters to be ready
    log_info "Waiting for monitoring components to be ready..."
    kubectl wait --for=condition=available deployment/postgres-exporter-coordinator -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=available deployment/postgres-exporter-workers -n $NAMESPACE --timeout=300s
    
    log_success "Monitoring deployed successfully"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying database deployment..."
    
    echo
    log_info "=== Cluster Status ==="
    
    # Check namespace
    if kubectl get namespace $NAMESPACE &> /dev/null; then
        log_success "Namespace: $NAMESPACE exists"
    else
        log_error "Namespace: $NAMESPACE not found"
    fi
    
    # Check pods
    log_info "Pod status:"
    kubectl get pods -n $NAMESPACE -o wide
    
    # Check services
    log_info "Service status:"
    kubectl get svc -n $NAMESPACE
    
    # Check PVCs
    log_info "Storage status:"
    kubectl get pvc -n $NAMESPACE
    
    # Check Citus cluster status
    log_info "Testing Citus cluster connectivity..."
    
    # Port forward to coordinator (in background)
    kubectl port-forward svc/citus-coordinator 5433:5432 -n $NAMESPACE &
    PORT_FORWARD_PID=$!
    
    # Wait for port forward to establish
    sleep 5
    
    # Test connection
    if command -v psql &> /dev/null; then
        log_info "Testing database connection..."
        PGPASSWORD="its-camera-ai-super-secret-db-password" psql -h localhost -p 5433 -U postgres -d its_camera_ai -c "SELECT * FROM citus_get_active_worker_nodes();" || true
        PGPASSWORD="its-camera-ai-super-secret-db-password" psql -h localhost -p 5433 -U postgres -d its_camera_ai -c "SELECT schemaname, tablename FROM timescaledb_information.hypertables;" || true
    else
        log_warning "psql not found. Skipping connection test."
    fi
    
    # Kill port forward
    kill $PORT_FORWARD_PID 2>/dev/null || true
    
    echo
    log_info "=== Connection Information ==="
    echo "Coordinator (via PgBouncer): pgbouncer.postgresql-cluster.svc.cluster.local:5432"
    echo "Direct Coordinator: citus-coordinator.postgresql-cluster.svc.cluster.local:5432"
    echo "Database: its_camera_ai"
    echo "Username: postgres"
    echo "Password: its-camera-ai-super-secret-db-password"
    echo
    
    log_info "=== Monitoring Endpoints ==="
    echo "PostgreSQL Metrics (Coordinator): postgres-exporter-coordinator.postgresql-cluster.svc.cluster.local:9187/metrics"
    echo "PostgreSQL Metrics (Workers): postgres-exporter-workers.postgresql-cluster.svc.cluster.local:9187/metrics"
    echo "PgBouncer Metrics: pgbouncer.postgresql-cluster.svc.cluster.local:9127/metrics"
    echo
}

# Performance test
performance_test() {
    log_info "Running basic performance test..."
    
    # Port forward to PgBouncer (in background)
    kubectl port-forward svc/pgbouncer 5434:5432 -n $NAMESPACE &
    PORT_FORWARD_PID=$!
    
    # Wait for port forward to establish
    sleep 5
    
    if command -v psql &> /dev/null; then
        log_info "Inserting test data..."
        PGPASSWORD="its-camera-ai-super-secret-db-password" psql -h localhost -p 5434 -U postgres -d its_camera_ai << 'EOF'
        
        -- Insert test detection events
        INSERT INTO detection_events (camera_id, detections, confidence_scores, bounding_boxes, tenant_id)
        SELECT 
            (SELECT camera_id FROM cameras LIMIT 1),
            '{"objects": ["car", "person"]}',
            ARRAY[0.95, 0.87],
            '{"car": [100, 200, 300, 400], "person": [150, 250, 200, 350]}',
            gen_random_uuid()
        FROM generate_series(1, 1000);
        
        -- Insert test metrics
        INSERT INTO camera_metrics (camera_id, fps, bandwidth_mbps, cpu_usage, memory_usage, gpu_usage, inference_latency_ms, tenant_id)
        SELECT 
            (SELECT camera_id FROM cameras LIMIT 1),
            25.0 + random() * 5,
            8.5 + random() * 2,
            45.0 + random() * 30,
            60.0 + random() * 20,
            75.0 + random() * 20,
            45.0 + random() * 30,
            gen_random_uuid()
        FROM generate_series(1, 1000);
        
        -- Test query performance
        \timing on
        SELECT COUNT(*) FROM detection_events WHERE timestamp > NOW() - INTERVAL '1 hour';
        SELECT AVG(inference_latency_ms) FROM camera_metrics WHERE timestamp > NOW() - INTERVAL '1 hour';
        \timing off
        
EOF
        
        log_success "Performance test completed"
    else
        log_warning "psql not found. Skipping performance test."
    fi
    
    # Kill port forward
    kill $PORT_FORWARD_PID 2>/dev/null || true
}

# Show next steps
show_next_steps() {
    echo
    log_success "Database deployment completed successfully!"
    echo
    log_info "Next steps:"
    echo "  1. Configure application connection strings to use pgbouncer service"
    echo "  2. Setup Prometheus to scrape database metrics"
    echo "  3. Configure Grafana dashboards for database monitoring"
    echo "  4. Setup database backup strategy"
    echo "  5. Configure read replicas for analytics workloads"
    echo
    log_info "Application connection examples:"
    echo "  Go: postgres://postgres:password@pgbouncer.postgresql-cluster.svc.cluster.local:5432/its_camera_ai"
    echo "  Python: postgresql://postgres:password@pgbouncer.postgresql-cluster.svc.cluster.local:5432/its_camera_ai"
    echo "  Docker: -e DATABASE_URL=postgres://postgres:password@pgbouncer.postgresql-cluster.svc.cluster.local:5432/its_camera_ai"
    echo
}

# Main deployment function
main() {
    log_info "Starting Citus database cluster deployment..."
    echo "Cluster: $CLUSTER_NAME"
    echo "Region: $REGION"
    echo "Namespace: $NAMESPACE"
    echo
    
    check_prerequisites
    deploy_namespace
    deploy_citus_cluster
    initialize_citus
    deploy_pgbouncer
    deploy_monitoring
    verify_deployment
    performance_test
    show_next_steps
}

# Handle script interruption
trap 'log_error "Script interrupted"; exit 1' INT TERM

# Run main function
main "$@"