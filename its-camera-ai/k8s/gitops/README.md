# ITS Camera AI GitOps Infrastructure

This directory contains the GitOps configuration for the ITS Camera AI production deployment using ArgoCD and Harbor container registry.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Git Repos     │    │     ArgoCD      │    │   Kubernetes    │
│                 │    │                 │    │     Cluster     │
│ Infrastructure  │───▶│  Application    │───▶│                 │
│ Applications    │    │  Controller     │    │ ITS Camera AI   │
│ ML Models       │    │                 │    │   Services      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │     Harbor      │
                       │ Container Reg.  │
                       │                 │
                       │ Security Scan   │
                       │ Image Signing   │
                       └─────────────────┘
```

## Components

### ArgoCD
- **Purpose**: GitOps continuous delivery tool
- **Features**: 
  - Multi-cluster deployment
  - RBAC with team-based access
  - Automated sync with git repositories
  - Web UI and CLI interface
  - OIDC integration
- **High Availability**: 3 replicas for server, controller, and repo-server

### Harbor
- **Purpose**: Enterprise container registry
- **Features**:
  - Vulnerability scanning with Trivy
  - Image signing with Notary
  - Helm chart repository
  - Replication across regions
  - RBAC and project isolation
- **Storage**: AWS EBS with 1TB for image storage

## Directory Structure

```
k8s/gitops/
├── argocd/                     # ArgoCD deployment configuration
│   ├── namespace.yaml          # ArgoCD namespace
│   ├── values.yaml            # Helm values for ArgoCD
│   ├── kustomization.yaml     # Kustomize configuration
│   └── patches/               # Kustomize patches for customization
├── applications/              # ArgoCD Application definitions
│   ├── its-camera-ai-core.yaml
│   ├── its-camera-ai-ml.yaml
│   └── its-camera-ai-streaming.yaml
├── applicationsets/           # ArgoCD ApplicationSet for automated deployment
│   └── its-camera-ai-appset.yaml
├── projects/                  # ArgoCD Project definitions
│   └── its-camera-ai-project.yaml
└── registry/                  # Harbor container registry
    └── harbor/
        ├── namespace.yaml
        └── values.yaml
```

## Deployment

### Prerequisites

1. **EKS Cluster**: Production Kubernetes cluster with GPU nodes
2. **AWS CLI**: Configured with appropriate permissions
3. **kubectl**: Configured to access the cluster
4. **Helm**: Version 3.x installed

### Step 1: Deploy GitOps Infrastructure

```bash
# Run the deployment script
./scripts/deploy-gitops.sh
```

This script will:
1. Check prerequisites
2. Update kubeconfig for EKS access
3. Add required Helm repositories
4. Deploy ArgoCD with high-availability configuration
5. Deploy Harbor container registry
6. Create ArgoCD projects and applications
7. Verify the deployment

### Step 2: Access ArgoCD

```bash
# Get ArgoCD admin password
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d

# Access ArgoCD UI
# URL: https://argocd.its-camera-ai.com
# Username: admin
# Password: <from above command>
```

### Step 3: Access Harbor

```bash
# Harbor is available at: https://harbor.its-camera-ai.com
# Username: admin
# Password: Harbor12345 (change this in production)
```

## Configuration

### ArgoCD Configuration

Key configuration files:
- `argocd/values.yaml`: Main ArgoCD configuration
- `projects/its-camera-ai-project.yaml`: Project-level RBAC and policies
- `applications/`: Individual application definitions

### RBAC Configuration

The setup includes role-based access control:

1. **Platform Admin**: Full access to all applications and clusters
2. **ML Engineers**: Access to ML-specific applications
3. **DevOps Engineers**: Deployment access to all applications
4. **Readonly**: Read-only access for monitoring

### Harbor Configuration

Key features configured:
- **Trivy Security Scanning**: Automatic vulnerability scanning
- **Notary Image Signing**: Digital signing for image integrity
- **Project Isolation**: Separate projects for different teams
- **Replication**: Cross-region replication for disaster recovery

## Application Deployment

### Using ArgoCD Applications

Applications are defined in the `applications/` directory:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: its-camera-ai-core
spec:
  project: its-camera-ai
  source:
    repoURL: https://github.com/its-camera-ai/infrastructure
    path: k8s/applications/core
  destination:
    server: https://kubernetes.default.svc
    namespace: its-camera-ai-core
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

### Using ApplicationSets

For multi-environment deployments, use ApplicationSets:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: its-camera-ai-services
spec:
  generators:
  - git:
      repoURL: https://github.com/its-camera-ai/infrastructure
      directories:
      - path: k8s/applications/*
```

## Monitoring and Observability

### ArgoCD Metrics

ArgoCD exposes Prometheus metrics for monitoring:
- Application sync status
- Repository health
- Controller performance metrics

### Harbor Metrics

Harbor provides metrics for:
- Image pull/push statistics
- Vulnerability scan results
- Storage usage

## Security

### Image Security

1. **Vulnerability Scanning**: All images are scanned with Trivy
2. **Image Signing**: Images are signed with Notary
3. **Admission Control**: Only signed images from Harbor are allowed

### Access Control

1. **OIDC Integration**: Single sign-on with corporate identity provider
2. **RBAC**: Role-based access control for teams
3. **Network Policies**: Kubernetes network isolation

### Secrets Management

1. **Sealed Secrets**: Encrypted secrets in Git repositories
2. **External Secrets**: Integration with AWS Secrets Manager
3. **Service Accounts**: IRSA for AWS service access

## Backup and Disaster Recovery

### ArgoCD Backup

- Application definitions stored in Git (GitOps)
- Cluster state can be recreated from Git
- ArgoCD configuration backed up to S3

### Harbor Backup

- Image data replicated across regions
- Database backups to AWS RDS
- Configuration stored in Git

## Troubleshooting

### Common Issues

1. **Application Sync Failures**
   ```bash
   # Check application status
   argocd app get <app-name>
   
   # Check sync logs
   argocd app logs <app-name>
   ```

2. **Image Pull Failures**
   ```bash
   # Check Harbor connectivity
   kubectl get pods -n harbor
   
   # Verify image pull secrets
   kubectl get secrets -n <namespace>
   ```

3. **ArgoCD Controller Issues**
   ```bash
   # Check controller logs
   kubectl logs -n argocd deployment/argocd-application-controller
   
   # Restart controller
   kubectl rollout restart deployment/argocd-application-controller -n argocd
   ```

### Debugging Commands

```bash
# ArgoCD status
kubectl get applications -n argocd
kubectl describe application <app-name> -n argocd

# Harbor status
kubectl get pods -n harbor
kubectl logs <harbor-pod> -n harbor

# Check ingress
kubectl get ingress -A
kubectl describe ingress <ingress-name> -n <namespace>
```

## Maintenance

### Updating ArgoCD

```bash
# Update ArgoCD using Kustomize
kubectl apply -k k8s/gitops/argocd/

# Or update using Helm
helm upgrade argocd argo/argo-cd -n argocd -f argocd/values.yaml
```

### Updating Harbor

```bash
# Update Harbor using Helm
helm upgrade harbor harbor/harbor -n harbor -f registry/harbor/values.yaml
```

### Certificate Rotation

- SSL certificates are managed by AWS Certificate Manager
- ArgoCD and Harbor automatically pick up certificate updates
- Monitor certificate expiration dates

## Support

For issues and support:
1. Check ArgoCD documentation: https://argo-cd.readthedocs.io/
2. Check Harbor documentation: https://goharbor.io/docs/
3. Review application logs and ArgoCD events
4. Contact the platform team for assistance