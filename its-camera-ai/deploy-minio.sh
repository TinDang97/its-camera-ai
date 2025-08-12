#!/bin/bash

# MinIO Deployment Script for ITS Camera AI
# Automated deployment with validation and monitoring setup

set -euo pipefail

# Default values
ENVIRONMENT="${ENVIRONMENT:-development}"
NAMESPACE="${NAMESPACE:-its-camera-ai}"
DEPLOY_METHOD="${DEPLOY_METHOD:-helm}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_TESTS="${SKIP_TESTS:-false}"
MONITORING_ENABLED="${MONITORING_ENABLED:-true}"

# Colors for output
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

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy MinIO infrastructure for ITS Camera AI system.

OPTIONS:
    -e, --environment    Environment (development|staging|production) [default: development]
    -n, --namespace      Kubernetes namespace [default: its-camera-ai]
    -m, --method         Deployment method (helm|terraform|kubectl) [default: helm]
    -d, --dry-run        Perform dry run without actual deployment [default: false]
    -s, --skip-tests     Skip integration tests [default: false]
    --no-monitoring      Disable monitoring setup [default: false]
    -h, --help           Show this help message

EXAMPLES:
    # Deploy to development environment
    $0 -e development

    # Deploy to production with Terraform
    $0 -e production -m terraform

    # Dry run deployment
    $0 -e staging -d

    # Deploy without monitoring
    $0 -e development --no-monitoring

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -m|--method)
            DEPLOY_METHOD="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN="true"
            shift
            ;;
        -s|--skip-tests)
            SKIP_TESTS="true"
            shift
            ;;
        --no-monitoring)
            MONITORING_ENABLED="false"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
    log_error "Invalid environment: $ENVIRONMENT"
    log_error "Valid environments: development, staging, production"
    exit 1
fi

# Validate deployment method
if [[ ! "$DEPLOY_METHOD" =~ ^(helm|terraform|kubectl)$ ]]; then
    log_error "Invalid deployment method: $DEPLOY_METHOD"
    log_error "Valid methods: helm, terraform, kubectl"
    exit 1
fi

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check required tools
    local tools=("kubectl")
    
    case $DEPLOY_METHOD in
        helm)
            tools+=("helm")
            ;;
        terraform)
            tools+=("terraform")
            ;;
    esac
    
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is required but not installed"
            exit 1
        fi
    done
    
    # Check Kubernetes connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check namespace exists or create it
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Namespace $NAMESPACE does not exist"
        if [[ "$DRY_RUN" == "false" ]]; then
            log_info "Creating namespace $NAMESPACE..."
            kubectl create namespace "$NAMESPACE"
            kubectl label namespace "$NAMESPACE" app.kubernetes.io/name=its-camera-ai
        fi
    fi
    
    log_success "Prerequisites check passed"
}

# Deploy with Helm
deploy_with_helm() {
    log_info "Deploying MinIO with Helm..."
    
    cd infrastructure/helm/its-camera-ai
    
    # Update dependencies
    log_info "Updating Helm dependencies..."
    helm dependency update
    
    # Prepare values for environment
    local helm_args=(
        "--namespace" "$NAMESPACE"
        "--create-namespace"
        "--values" "values.yaml"
        "--set" "minio.enabled=true"
        "--set" "global.environment=$ENVIRONMENT"
    )
    
    # Environment-specific configurations
    case $ENVIRONMENT in
        development)
            helm_args+=(
                "--set" "minio.statefulset.replicaCount=4"
                "--set" "minio.persistence.size=100Gi"
                "--set" "minio.persistence.storageClass=standard"
            )
            ;;
        staging)
            helm_args+=(
                "--set" "minio.statefulset.replicaCount=6"
                "--set" "minio.persistence.size=250Gi"
                "--set" "minio.persistence.storageClass=fast-ssd"
            )
            ;;
        production)
            helm_args+=(
                "--set" "minio.statefulset.replicaCount=8"
                "--set" "minio.persistence.size=500Gi"
                "--set" "minio.persistence.storageClass=fast-ssd"
                "--set" "minio.autoscaling.enabled=true"
                "--set" "minio.metrics.enabled=true"
            )
            ;;
    esac
    
    if [[ "$DRY_RUN" == "true" ]]; then
        helm_args+=("--dry-run")
        log_info "Performing dry run..."
    fi
    
    # Deploy or upgrade
    if helm list -n "$NAMESPACE" | grep -q "its-camera-ai"; then
        log_info "Upgrading existing Helm release..."
        helm upgrade its-camera-ai . "${helm_args[@]}"
    else
        log_info "Installing new Helm release..."
        helm install its-camera-ai . "${helm_args[@]}"
    fi
    
    cd - > /dev/null
}

# Deploy with Terraform
deploy_with_terraform() {
    log_info "Deploying MinIO with Terraform..."
    
    cd infrastructure/terraform
    
    # Initialize Terraform
    log_info "Initializing Terraform..."
    terraform init
    
    # Select or create workspace
    if terraform workspace list | grep -q "$ENVIRONMENT"; then
        terraform workspace select "$ENVIRONMENT"
    else
        terraform workspace new "$ENVIRONMENT"
    fi
    
    local tf_vars_file="environments/${ENVIRONMENT}.tfvars"
    
    if [[ ! -f "$tf_vars_file" ]]; then
        log_error "Terraform variables file not found: $tf_vars_file"
        exit 1
    fi
    
    # Plan deployment
    log_info "Planning Terraform deployment..."
    terraform plan -var-file="$tf_vars_file" -out="$ENVIRONMENT.tfplan"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        # Apply deployment
        log_info "Applying Terraform deployment..."
        terraform apply "$ENVIRONMENT.tfplan"
    else
        log_info "Dry run completed. Terraform plan saved to $ENVIRONMENT.tfplan"
    fi
    
    cd - > /dev/null
}

# Deploy with kubectl
deploy_with_kubectl() {
    log_info "Deploying MinIO with kubectl..."
    
    local manifests=(
        "infrastructure/kubernetes/minio-statefulset.yaml"
        "infrastructure/kubernetes/minio-hpa.yaml"
    )
    
    for manifest in "${manifests[@]}"; do
        if [[ -f "$manifest" ]]; then
            log_info "Applying $manifest..."
            if [[ "$DRY_RUN" == "true" ]]; then
                kubectl apply -f "$manifest" --dry-run=client
            else
                kubectl apply -f "$manifest"
            fi
        else
            log_warning "Manifest not found: $manifest"
        fi
    done
}

# Wait for deployment to be ready
wait_for_deployment() {
    if [[ "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    
    log_info "Waiting for MinIO deployment to be ready..."
    
    # Wait for StatefulSet to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=minio -n "$NAMESPACE" --timeout=600s
    
    # Check service endpoints
    kubectl get svc -n "$NAMESPACE" -l app.kubernetes.io/name=minio
    
    log_success "MinIO deployment is ready"
}

# Run integration tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]] || [[ "$DRY_RUN" == "true" ]]; then
        log_info "Skipping integration tests"
        return 0
    fi
    
    log_info "Running MinIO integration tests..."
    
    # Apply test job
    kubectl apply -f infrastructure/testing/minio-integration-tests.yaml
    
    # Wait for test completion
    kubectl wait --for=condition=complete job/minio-integration-test -n "$NAMESPACE" --timeout=600s
    
    # Show test results
    kubectl logs job/minio-integration-test -n "$NAMESPACE"
    
    # Check if tests passed
    if kubectl get job minio-integration-test -n "$NAMESPACE" -o jsonpath='{.status.succeeded}' | grep -q "1"; then
        log_success "Integration tests passed"
        
        # Cleanup test job
        kubectl delete job minio-integration-test -n "$NAMESPACE" --ignore-not-found=true
    else
        log_error "Integration tests failed"
        kubectl logs job/minio-integration-test -n "$NAMESPACE" --tail=50
        exit 1
    fi
}

# Setup monitoring
setup_monitoring() {
    if [[ "$MONITORING_ENABLED" == "false" ]] || [[ "$DRY_RUN" == "true" ]]; then
        log_info "Skipping monitoring setup"
        return 0
    fi
    
    log_info "Setting up MinIO monitoring..."
    
    # Apply monitoring configuration
    kubectl apply -f infrastructure/monitoring/minio-production-alerts.yaml
    
    log_success "Monitoring setup completed"
}

# Validate deployment
validate_deployment() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Skipping deployment validation (dry run)"
        return 0
    fi
    
    log_info "Validating MinIO deployment..."
    
    # Check pod status
    local ready_pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=minio --no-headers | grep "Running" | wc -l)
    local total_pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=minio --no-headers | wc -l)
    
    if [[ "$ready_pods" -eq "$total_pods" ]] && [[ "$total_pods" -gt 0 ]]; then
        log_success "All MinIO pods are running ($ready_pods/$total_pods)"
    else
        log_error "Some MinIO pods are not ready ($ready_pods/$total_pods)"
        kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=minio
        exit 1
    fi
    
    # Check service connectivity
    log_info "Testing service connectivity..."
    kubectl run test-connectivity --image=curlimages/curl --rm -i --restart=Never -- \
        curl -f "http://minio-service.$NAMESPACE.svc.cluster.local:9000/minio/health/live" || {
        log_error "Service connectivity test failed"
        exit 1
    }
    
    log_success "MinIO deployment validation passed"
}

# Main execution
main() {
    log_info "Starting MinIO deployment for ITS Camera AI"
    log_info "Environment: $ENVIRONMENT"
    log_info "Namespace: $NAMESPACE"
    log_info "Method: $DEPLOY_METHOD"
    log_info "Dry run: $DRY_RUN"
    
    check_prerequisites
    
    case $DEPLOY_METHOD in
        helm)
            deploy_with_helm
            ;;
        terraform)
            deploy_with_terraform
            ;;
        kubectl)
            deploy_with_kubectl
            ;;
    esac
    
    wait_for_deployment
    validate_deployment
    setup_monitoring
    run_tests
    
    log_success "MinIO deployment completed successfully!"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        log_info "MinIO Console URL: http://minio-service.$NAMESPACE.svc.cluster.local:9001"
        log_info "MinIO API URL: http://minio-service.$NAMESPACE.svc.cluster.local:9000"
        
        case $ENVIRONMENT in
            development)
                log_info "Access MinIO Console locally: kubectl port-forward -n $NAMESPACE svc/minio-service 9001:9001"
                ;;
            production)
                log_info "MinIO Console: https://minio-console.its-camera-ai.com"
                log_info "MinIO API: https://minio.its-camera-ai.com"
                ;;
        esac
    fi
}

# Execute main function
main "$@"