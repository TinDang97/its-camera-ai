# Vault Secrets Management for ITS Camera AI

This document describes the HashiCorp Vault secrets management implementation for the ITS Camera AI system, providing secure storage and retrieval of sensitive configuration data.

## Overview

The ITS Camera AI system uses HashiCorp Vault as the centralized secrets management solution, replacing static Kubernetes secrets with dynamic, secure secret retrieval. This implementation provides:

- **Secure Secret Storage**: AES-256 encryption for all secrets at rest
- **Dynamic Secret Injection**: Automatic secret injection into pods via Vault Agent
- **Authentication Integration**: Kubernetes service account-based authentication
- **Audit Logging**: Comprehensive audit trail for all secret access
- **High Availability**: Multi-replica Vault deployment with PostgreSQL backend
- **Token Management**: Automatic token renewal and lifecycle management

## Architecture

### Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │    │  Vault Agent    │    │   Vault Server  │
│      Pods       │◄──►│   Injector      │◄──►│    Cluster      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                               ┌─────────────────┐
                                               │   PostgreSQL    │
                                               │    Backend      │
                                               └─────────────────┘
```

### Secret Organization

Secrets are organized hierarchically in Vault under the `its-camera-ai/` path:

```
its-camera-ai/
├── api/
│   └── jwt                  # JWT signing keys
├── database/
│   ├── postgres            # PostgreSQL credentials
│   └── redis               # Redis credentials
├── encryption/
│   └── master              # Master encryption keys
├── ml/
│   └── models              # ML model secrets
├── analytics/
│   └── timescale           # Analytics database credentials
└── external/
    └── monitoring          # External API keys
```

## Deployment

### Prerequisites

- Kubernetes cluster with RBAC enabled
- PostgreSQL cluster for Vault backend storage
- Helm 3.x installed
- kubectl configured for cluster access

### Quick Setup

Run the automated setup script:

```bash
./scripts/setup-vault.sh
```

This script will:
1. Create the `its-camera-ai` namespace
2. Set up PostgreSQL backend for Vault
3. Deploy Vault StatefulSet with 3 replicas
4. Initialize and unseal Vault
5. Configure authentication methods and policies
6. Deploy Vault Agent Injector
7. Populate initial secrets

### Manual Deployment

1. **Deploy Vault Server:**
   ```bash
   kubectl apply -f infrastructure/kubernetes/security/vault-deployment.yaml
   ```

2. **Initialize Vault:**
   ```bash
   kubectl apply -f infrastructure/kubernetes/security/vault-init-job.yaml
   ```

3. **Deploy Agent Injector:**
   ```bash
   kubectl apply -f infrastructure/kubernetes/security/vault-injector.yaml
   ```

## Configuration

### Vault Server Configuration

The Vault server is configured with:

- **Storage Backend**: PostgreSQL with connection pooling
- **Listener**: TLS-enabled on port 8200
- **High Availability**: 3-node cluster with automatic leader election
- **Audit Logging**: File-based audit logs in `/vault/audit/`
- **Telemetry**: Prometheus metrics enabled

### Authentication Methods

#### Kubernetes Auth

Service accounts authenticate using the Kubernetes auth method:

```bash
vault auth enable kubernetes
vault write auth/kubernetes/config \
    kubernetes_host="https://kubernetes.default.svc" \
    kubernetes_ca_cert="@/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
```

#### Service Account Roles

Three main roles are configured:

- **its-camera-ai-api**: Access to API and database secrets
- **its-camera-ai-ml**: Access to ML model secrets
- **its-camera-ai-analytics**: Access to analytics database secrets

### Policies

Fine-grained policies control access to secrets:

```hcl
# API Service Policy
path "its-camera-ai/data/api/*" {
  capabilities = ["read"]
}

path "its-camera-ai/data/database/*" {
  capabilities = ["read"]
}

path "its-camera-ai/data/encryption/*" {
  capabilities = ["read"]
}
```

## Usage

### Python Vault Client

Use the provided Python client for secret retrieval:

```python
from its_camera_ai.core.secrets import get_vault_client

# Get Vault client instance
vault_client = await get_vault_client()

# Retrieve database configuration
db_config = await vault_client.get_database_config()

# Retrieve API configuration
api_config = await vault_client.get_api_config()

# Retrieve specific secret
secret = await vault_client.get_secret("its-camera-ai/data/api/jwt")
jwt_key = secret.data["secret_key"]
```

### Kubernetes Pod Integration

#### Method 1: Vault Agent Injection (Recommended)

Add annotations to your pod spec:

```yaml
apiVersion: v1
kind: Pod
metadata:
  annotations:
    vault.hashicorp.com/agent-inject: "true"
    vault.hashicorp.com/role: "its-camera-ai-api"
    vault.hashicorp.com/agent-inject-secret-database: "its-camera-ai/data/database/postgres"
    vault.hashicorp.com/agent-inject-template-database: |
      {{- with secret "its-camera-ai/data/database/postgres" -}}
      export DB_USERNAME="{{ .Data.data.username }}"
      export DB_PASSWORD="{{ .Data.data.password }}"
      export DB_HOST="{{ .Data.data.host }}"
      {{- end }}
spec:
  serviceAccountName: api-service
  containers:
  - name: app
    image: its-camera-ai:latest
    command: ["/bin/sh"]
    args: ["-c", "source /vault/secrets/database && exec my-app"]
```

#### Method 2: Direct API Access

```python
import os
from its_camera_ai.core.secrets import load_secrets_from_vault

# Load all secrets and set as environment variables
secrets = await load_secrets_from_vault()
for key, value in secrets.items():
    os.environ[key] = value
```

### CLI Commands

The ITS Camera AI CLI includes Vault management commands:

```bash
# Check Vault status
its-camera-ai security vault-status

# Rotate secrets
its-camera-ai security rotate-secrets

# Backup secrets
its-camera-ai security backup-secrets --output-file secrets-backup.json

# Restore secrets
its-camera-ai security restore-secrets --input-file secrets-backup.json
```

## Secret Types

### Database Credentials

**Path**: `its-camera-ai/data/database/postgres`

```json
{
  "username": "its_user",
  "password": "secure_generated_password",
  "host": "postgresql-cluster",
  "port": "5432",
  "database": "its_camera_ai",
  "ssl_mode": "require"
}
```

### API Secrets

**Path**: `its-camera-ai/data/api/jwt`

```json
{
  "secret_key": "base64_encoded_jwt_secret",
  "refresh_secret_key": "base64_encoded_refresh_secret"
}
```

### Encryption Keys

**Path**: `its-camera-ai/data/encryption/master`

```json
{
  "key": "base64_encoded_encryption_key",
  "algorithm": "AES-256-GCM"
}
```

### ML Model Secrets

**Path**: `its-camera-ai/data/ml/models`

```json
{
  "model_registry_key": "model_registry_api_key",
  "model_encryption_key": "model_specific_encryption_key",
  "tensorrt_license_key": "tensorrt_license_key"
}
```

## Security Features

### Encryption

- **Transit Encryption**: All communication with Vault uses TLS 1.2+
- **Storage Encryption**: AES-256 encryption for data at rest
- **Key Rotation**: Automatic encryption key rotation every 90 days

### Access Control

- **Authentication**: Kubernetes service account-based authentication
- **Authorization**: Fine-grained policies per service
- **Audit Logging**: Complete audit trail for all operations
- **Token Lifecycle**: Automatic token renewal and expiration

### Network Security

- **Network Policies**: Restricted network access to Vault
- **TLS Termination**: TLS certificates managed by cert-manager
- **Service Mesh**: Integration with Istio for mTLS

## Monitoring and Alerts

### Vault Metrics

Vault exposes Prometheus metrics on `/v1/sys/metrics`:

- `vault_core_unsealed`: Vault seal status
- `vault_runtime_alloc_bytes`: Memory usage
- `vault_runtime_num_goroutines`: Active goroutines
- `vault_barrier_put`: Write operations
- `vault_barrier_get`: Read operations

### Key Alerts

```yaml
# Vault Down Alert
- alert: VaultDown
  expr: up{job="vault"} == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Vault server is down"

# Vault Sealed Alert  
- alert: VaultSealed
  expr: vault_core_unsealed == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Vault is sealed"

# High Memory Usage
- alert: VaultHighMemoryUsage
  expr: vault_runtime_alloc_bytes > 1073741824  # 1GB
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Vault memory usage is high"
```

### Grafana Dashboard

A pre-configured Grafana dashboard is available at:
- **Dashboard ID**: vault-overview
- **Panels**: Health status, performance metrics, audit log analysis
- **Alerts**: Integration with PagerDuty for critical issues

## Backup and Recovery

### Automated Backups

Vault data is automatically backed up every 24 hours:

```bash
# Manual backup
kubectl exec vault-0 -n its-camera-ai -- vault operator raft snapshot save /tmp/backup.snap

# Copy backup locally
kubectl cp its-camera-ai/vault-0:/tmp/backup.snap ./vault-backup-$(date +%Y%m%d).snap
```

### Disaster Recovery

1. **Restore from Snapshot:**
   ```bash
   kubectl exec vault-0 -n its-camera-ai -- vault operator raft snapshot restore /tmp/backup.snap
   ```

2. **Re-initialize if needed:**
   ```bash
   kubectl delete job vault-init -n its-camera-ai
   kubectl apply -f infrastructure/kubernetes/security/vault-init-job.yaml
   ```

### Unseal Key Management

Unseal keys are stored in the `vault-unseal-keys` Kubernetes secret:

```bash
# Retrieve unseal keys
kubectl get secret vault-unseal-keys -n its-camera-ai -o json | \
  jq -r '.data | to_entries[] | "\(.key): \(.value | @base64d)"'

# Manual unseal (if needed)
kubectl exec vault-0 -n its-camera-ai -- vault operator unseal $UNSEAL_KEY_1
kubectl exec vault-0 -n its-camera-ai -- vault operator unseal $UNSEAL_KEY_2
kubectl exec vault-0 -n its-camera-ai -- vault operator unseal $UNSEAL_KEY_3
```

## Troubleshooting

### Common Issues

#### 1. Vault Pods Not Starting

**Symptoms**: Vault pods stuck in `Pending` or `CrashLoopBackOff`

**Solutions**:
```bash
# Check pod events
kubectl describe pod vault-0 -n its-camera-ai

# Check persistent volume
kubectl get pv,pvc -n its-camera-ai

# Check TLS certificates
kubectl get secret vault-tls -n its-camera-ai -o yaml
```

#### 2. Authentication Failures

**Symptoms**: Applications cannot authenticate with Vault

**Solutions**:
```bash
# Check service account token
kubectl get serviceaccount api-service -n its-camera-ai -o yaml

# Verify Kubernetes auth configuration
kubectl exec vault-0 -n its-camera-ai -- vault auth list

# Test authentication manually
kubectl exec vault-0 -n its-camera-ai -- vault write auth/kubernetes/login \
  role=its-camera-ai-api jwt=$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)
```

#### 3. Secret Access Denied

**Symptoms**: `403 Forbidden` errors when accessing secrets

**Solutions**:
```bash
# Check policies
kubectl exec vault-0 -n its-camera-ai -- vault policy list
kubectl exec vault-0 -n its-camera-ai -- vault policy read its-camera-ai-api

# Verify role binding
kubectl exec vault-0 -n its-camera-ai -- vault read auth/kubernetes/role/its-camera-ai-api
```

#### 4. High Memory Usage

**Symptoms**: Vault consuming excessive memory

**Solutions**:
```bash
# Check memory usage
kubectl top pod vault-0 -n its-camera-ai

# Adjust memory limits
kubectl patch statefulset vault -n its-camera-ai -p '{"spec":{"template":{"spec":{"containers":[{"name":"vault","resources":{"limits":{"memory":"4Gi"}}}]}}}}'

# Clear cache if applicable
kubectl exec vault-0 -n its-camera-ai -- vault operator seal
kubectl exec vault-0 -n its-camera-ai -- vault operator unseal $KEY1 $KEY2 $KEY3
```

### Debugging Tools

#### Vault Logs
```bash
# View Vault server logs
kubectl logs vault-0 -n its-camera-ai -f

# View Vault Agent Injector logs
kubectl logs deployment/vault-agent-injector -n its-camera-ai -f
```

#### Health Checks
```bash
# Check Vault health
kubectl exec vault-0 -n its-camera-ai -- vault status

# Check cluster status
kubectl exec vault-0 -n its-camera-ai -- vault operator raft list-peers
```

#### Secret Verification
```bash
# List secret engines
kubectl exec vault-0 -n its-camera-ai -- vault secrets list

# Test secret retrieval
kubectl exec vault-0 -n its-camera-ai -- vault kv get its-camera-ai/api/jwt
```

## Performance Tuning

### Connection Pooling

Adjust PostgreSQL backend connection settings:

```hcl
storage "postgresql" {
  connection_url = "postgres://vault_user:password@postgresql:5432/vault_db"
  max_parallel   = 128
  max_idle_connections = 20
  max_connections      = 40
}
```

### Memory Optimization

Configure Vault memory limits based on usage:

```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

### Cache Settings

Optimize secret caching in the Python client:

```python
vault_config = VaultConfig(
    cache_ttl=1800,  # 30 minutes
    enable_cache=True,
    max_retries=5
)
```

## Migration from Kubernetes Secrets

### Migration Steps

1. **Backup existing secrets:**
   ```bash
   kubectl get secrets -n its-camera-ai -o yaml > secrets-backup.yaml
   ```

2. **Deploy Vault and populate secrets:**
   ```bash
   ./scripts/setup-vault.sh
   ```

3. **Update application deployments:**
   - Add Vault Agent annotations
   - Update environment variable sources
   - Modify startup scripts to source Vault secrets

4. **Validate functionality:**
   - Test secret retrieval
   - Verify application startup
   - Check audit logs

5. **Remove old secrets:**
   ```bash
   kubectl delete secret api-secrets postgresql-secrets redis-secrets -n its-camera-ai
   ```

### Rollback Plan

If issues occur, rollback to Kubernetes secrets:

1. **Restore original secrets:**
   ```bash
   kubectl apply -f secrets-backup.yaml
   ```

2. **Revert application deployments:**
   - Remove Vault Agent annotations
   - Restore original environment variable sources

3. **Scale down Vault:**
   ```bash
   kubectl scale statefulset vault --replicas=0 -n its-camera-ai
   ```

## Best Practices

### Security
- Use separate roles for different service types
- Implement least-privilege access policies
- Regularly rotate secrets and tokens
- Monitor audit logs for suspicious activity
- Use auto-unseal with cloud KMS in production

### Operations
- Automate secret rotation
- Implement backup and disaster recovery procedures
- Monitor Vault health and performance
- Use infrastructure as code for Vault configuration
- Document secret management procedures

### Development
- Use the Python Vault client for secret access
- Implement proper error handling for secret retrieval
- Cache secrets appropriately to reduce Vault load
- Use environment-specific Vault instances
- Test secret rotation scenarios

---

**Last Updated**: January 16, 2025  
**Version**: 1.0  
**Maintainer**: ITS Camera AI Infrastructure Team