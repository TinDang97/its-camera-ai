#!/bin/bash

# Deploy GitOps Infrastructure for ITS Camera AI Production
# This script deploys ArgoCD and Harbor container registry

set -euo pipefail

# Configuration
CLUSTER_NAME="its-camera-ai-prod"
REGION="us-west-2"
NAMESPACE_ARGOCD="argocd"
NAMESPACE_HARBOR="harbor"

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
    
    # Check helm
    if ! command -v helm &> /dev/null; then
        log_error "helm not found. Please install Helm."
        exit 1
    fi
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "aws CLI not found. Please install AWS CLI."
        exit 1
    fi
    
    # Check cluster access
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot access Kubernetes cluster. Please check kubeconfig."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Update kubeconfig for EKS cluster
update_kubeconfig() {
    log_info "Updating kubeconfig for EKS cluster..."
    aws eks update-kubeconfig --region $REGION --name $CLUSTER_NAME
    log_success "Kubeconfig updated"
}

# Add Helm repositories
add_helm_repos() {
    log_info "Adding Helm repositories..."
    
    # ArgoCD
    helm repo add argo https://argoproj.github.io/argo-helm
    
    # Harbor
    helm repo add harbor https://helm.goharbor.io
    
    # Update repositories
    helm repo update
    
    log_success "Helm repositories added and updated"
}

# Deploy ArgoCD
deploy_argocd() {
    log_info "Deploying ArgoCD..."
    
    # Create namespace
    kubectl create namespace $NAMESPACE_ARGOCD --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply ArgoCD using Kustomize
    kubectl apply -k k8s/gitops/argocd/
    
    # Wait for ArgoCD to be ready
    log_info "Waiting for ArgoCD to be ready..."
    kubectl wait --for=condition=available --timeout=600s deployment/argocd-server -n $NAMESPACE_ARGOCD
    kubectl wait --for=condition=available --timeout=600s deployment/argocd-application-controller -n $NAMESPACE_ARGOCD
    kubectl wait --for=condition=available --timeout=600s deployment/argocd-repo-server -n $NAMESPACE_ARGOCD
    
    log_success "ArgoCD deployed successfully"
}

# Deploy Harbor
deploy_harbor() {
    log_info "Deploying Harbor container registry..."
    
    # Create namespace
    kubectl create namespace $NAMESPACE_HARBOR --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy Harbor using Helm
    helm upgrade --install harbor harbor/harbor \
        --namespace $NAMESPACE_HARBOR \
        --values k8s/registry/harbor/values.yaml \
        --wait \
        --timeout=15m
    
    log_success "Harbor deployed successfully"
}

# Create ArgoCD project and applications
deploy_argocd_apps() {
    log_info "Creating ArgoCD project and applications..."
    
    # Wait a bit for ArgoCD to be fully ready
    sleep 30
    
    # Apply project
    kubectl apply -f k8s/gitops/projects/its-camera-ai-project.yaml
    
    # Apply applications
    kubectl apply -f k8s/gitops/applications/
    
    log_success "ArgoCD project and applications created"
}

# Get ArgoCD admin password
get_argocd_password() {
    log_info "Retrieving ArgoCD admin password..."
    
    local password
    password=$(kubectl -n $NAMESPACE_ARGOCD get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d)
    
    echo
    log_success "ArgoCD admin credentials:"
    echo "  URL: https://argocd.its-camera-ai.com"
    echo "  Username: admin"
    echo "  Password: $password"
    echo
}

# Get Harbor admin password
get_harbor_info() {
    log_info "Retrieving Harbor information..."
    
    echo
    log_success "Harbor admin credentials:"
    echo "  URL: https://harbor.its-camera-ai.com"
    echo "  Username: admin"
    echo "  Password: Harbor12345"
    echo
}

# Setup ArgoCD CLI
setup_argocd_cli() {
    log_info "Setting up ArgoCD CLI..."
    
    # Download ArgoCD CLI if not present
    if ! command -v argocd &> /dev/null; then
        log_info "Downloading ArgoCD CLI..."
        curl -sSL -o argocd-linux-amd64 https://github.com/argoproj/argo-cd/releases/latest/download/argocd-linux-amd64
        sudo install -m 555 argocd-linux-amd64 /usr/local/bin/argocd
        rm argocd-linux-amd64
    fi
    
    log_success "ArgoCD CLI is ready"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check ArgoCD
    if kubectl get pods -n $NAMESPACE_ARGOCD | grep -q "Running"; then
        log_success "ArgoCD pods are running"
    else
        log_error "ArgoCD pods are not running properly"
        kubectl get pods -n $NAMESPACE_ARGOCD
    fi
    
    # Check Harbor
    if kubectl get pods -n $NAMESPACE_HARBOR | grep -q "Running"; then
        log_success "Harbor pods are running"
    else
        log_error "Harbor pods are not running properly"
        kubectl get pods -n $NAMESPACE_HARBOR
    fi
    
    # Check services
    log_info "Service status:"
    kubectl get svc -n $NAMESPACE_ARGOCD
    kubectl get svc -n $NAMESPACE_HARBOR
}

# Main deployment function
main() {
    log_info "Starting GitOps infrastructure deployment..."
    echo "Cluster: $CLUSTER_NAME"
    echo "Region: $REGION"
    echo
    
    check_prerequisites
    update_kubeconfig
    add_helm_repos
    deploy_argocd
    deploy_harbor
    deploy_argocd_apps
    setup_argocd_cli
    verify_deployment
    
    echo
    log_success "GitOps infrastructure deployment completed!"
    echo
    
    get_argocd_password
    get_harbor_info
    
    echo
    log_info "Next steps:"
    echo "  1. Configure DNS for argocd.its-camera-ai.com and harbor.its-camera-ai.com"
    echo "  2. Setup SSL certificates in AWS Certificate Manager"
    echo "  3. Configure OIDC authentication for ArgoCD"
    echo "  4. Setup Harbor project and configure robot accounts"
    echo "  5. Configure image pull secrets in application namespaces"
    echo
}

# Run main function
main "$@"