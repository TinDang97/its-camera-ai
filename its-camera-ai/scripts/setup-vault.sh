#!/bin/bash
# Setup script for HashiCorp Vault deployment in ITS Camera AI
# This script deploys Vault, initializes it, and configures secrets management

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="its-camera-ai"
VAULT_RELEASE_NAME="vault"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}🔐 Setting up HashiCorp Vault for ITS Camera AI${NC}"
echo "============================================="

# Function to print status
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Verify prerequisites
echo -e "${BLUE}📋 Checking prerequisites...${NC}"

if ! command_exists kubectl; then
    print_error "kubectl is required but not installed"
    exit 1
fi

if ! command_exists helm; then
    print_error "helm is required but not installed"
    exit 1
fi

# Check if namespace exists
if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
    print_warning "Namespace $NAMESPACE does not exist, creating..."
    kubectl create namespace "$NAMESPACE"
    kubectl label namespace "$NAMESPACE" vault-injection=enabled
    print_status "Created namespace $NAMESPACE"
fi

# Create PostgreSQL database for Vault backend
echo -e "${BLUE}🗄️ Setting up Vault database backend...${NC}"

kubectl apply -f - <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: vault-postgres-secret
  namespace: $NAMESPACE
type: Opaque
data:
  password: $(echo -n "vault_secure_password_$(date +%s)" | base64)
---
apiVersion: batch/v1
kind: Job
metadata:
  name: vault-db-setup
  namespace: $NAMESPACE
spec:
  template:
    spec:
      restartPolicy: OnFailure
      containers:
      - name: postgres-client
        image: postgres:15-alpine
        env:
        - name: PGPASSWORD
          valueFrom:
            secretKeyRef:
              name: postgresql-secrets
              key: postgres-password
        command:
        - /bin/sh
        - -c
        - |
          # Wait for PostgreSQL to be ready
          until pg_isready -h postgresql-cluster -p 5432; do
            echo "Waiting for PostgreSQL..."
            sleep 5
          done
          
          # Create Vault database and user
          psql -h postgresql-cluster -U postgres -c "CREATE DATABASE vault_db;"
          psql -h postgresql-cluster -U postgres -c "CREATE USER vault_user WITH PASSWORD '\$(kubectl get secret vault-postgres-secret -n $NAMESPACE -o jsonpath='{.data.password}' | base64 -d)';"
          psql -h postgresql-cluster -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE vault_db TO vault_user;"
          
          echo "Vault database setup completed"
EOF

# Wait for database setup to complete
kubectl wait --for=condition=complete job/vault-db-setup -n "$NAMESPACE" --timeout=300s

print_status "Vault database backend configured"

# Deploy Vault using our custom manifests
echo -e "${BLUE}🚀 Deploying Vault...${NC}"

# Apply Vault deployment manifests
kubectl apply -f "$PROJECT_ROOT/infrastructure/kubernetes/security/vault-deployment.yaml"

# Wait for Vault pods to be ready
echo "Waiting for Vault pods to be ready..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=vault -n "$NAMESPACE" --timeout=300s

print_status "Vault deployed successfully"

# Initialize Vault
echo -e "${BLUE}🔑 Initializing Vault...${NC}"

kubectl apply -f "$PROJECT_ROOT/infrastructure/kubernetes/security/vault-init-job.yaml"

# Wait for initialization to complete
kubectl wait --for=condition=complete job/vault-init -n "$NAMESPACE" --timeout=600s

print_status "Vault initialized successfully"

# Deploy Vault Agent Injector
echo -e "${BLUE}🤖 Deploying Vault Agent Injector...${NC}"

kubectl apply -f "$PROJECT_ROOT/infrastructure/kubernetes/security/vault-injector.yaml"

# Wait for injector to be ready
kubectl wait --for=condition=available deployment/vault-agent-injector -n "$NAMESPACE" --timeout=300s

print_status "Vault Agent Injector deployed successfully"

# Verify Vault is working
echo -e "${BLUE}✅ Verifying Vault installation...${NC}"

# Port forward to access Vault UI temporarily
kubectl port-forward svc/vault 8200:8200 -n "$NAMESPACE" &
PORT_FORWARD_PID=$!

sleep 5

# Check Vault status
if curl -k -s https://localhost:8200/v1/sys/health | grep -q '"initialized":true'; then
    print_status "Vault is initialized and healthy"
else
    print_error "Vault health check failed"
    kill $PORT_FORWARD_PID 2>/dev/null || true
    exit 1
fi

# Kill port forward
kill $PORT_FORWARD_PID 2>/dev/null || true

# Display important information
echo -e "${BLUE}📝 Vault Setup Complete!${NC}"
echo "=========================="
echo ""
echo "🔐 Vault UI: https://vault.its-camera-ai.svc.cluster.local:8200"
echo "🗄️ Secret Path: its-camera-ai/"
echo "🔑 Unseal Keys: Stored in 'vault-unseal-keys' secret"
echo "🎟️ Root Token: Stored in 'vault-unseal-keys' secret"
echo ""
echo -e "${YELLOW}Important Security Notes:${NC}"
echo "• Unseal keys and root token are stored as Kubernetes secrets"
echo "• Consider using auto-unseal with cloud KMS for production"
echo "• Root token should be revoked after setting up auth methods"
echo "• Regular backup of Vault data is recommended"
echo ""

# Show how to access Vault
echo -e "${BLUE}🔧 Accessing Vault:${NC}"
echo ""
echo "1. Port forward to Vault (for debugging only):"
echo "   kubectl port-forward svc/vault 8200:8200 -n $NAMESPACE"
echo ""
echo "2. Get root token:"
echo "   kubectl get secret vault-unseal-keys -n $NAMESPACE -o jsonpath='{.data.root-token}' | base64 -d"
echo ""
echo "3. Access Vault UI at: https://localhost:8200"
echo ""

# Show how to test secret retrieval
echo -e "${BLUE}🧪 Testing Secret Retrieval:${NC}"
echo ""
echo "Test the Python Vault client:"
echo "python -c \"
import asyncio
from src.its_camera_ai.core.secrets import get_vault_client

async def test():
    client = await get_vault_client()
    config = await client.get_api_config()
    print('Successfully retrieved API config keys:', list(config.keys()))

asyncio.run(test())
\""
echo ""

print_status "Vault setup completed successfully! 🎉"

# Optional: Run tests
if [[ "${1:-}" == "--test" ]]; then
    echo -e "${BLUE}🧪 Running Vault integration tests...${NC}"
    
    # Run basic connectivity test
    kubectl run vault-test --rm -i --restart=Never --image=alpine/curl -- \
        curl -k -s https://vault.its-camera-ai.svc.cluster.local:8200/v1/sys/health
    
    print_status "Vault integration tests passed"
fi

echo ""
echo -e "${GREEN}🎉 Vault secrets management is now ready for ITS Camera AI!${NC}"