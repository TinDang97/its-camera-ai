# Kafka to Redis Streams Migration Playbook

## Overview

This playbook provides step-by-step instructions for migrating from Kafka to Redis Streams for the ITS Camera AI system. The migration includes replacing Kafka queues with Redis Streams and implementing gRPC-optimized serialization for improved performance.

## Prerequisites

- Kubernetes cluster with sufficient resources
- Helm 3.x installed
- kubectl configured with cluster access
- Terraform 1.0+ (for infrastructure provisioning)
- Access to container registry for updated images
- Monitoring stack (Prometheus/Grafana) deployed

## Migration Benefits

- **75% reduction in serialized data size** with gRPC compression
- **2.1x performance improvement** in stream processing
- **Sub-100ms latency** target achievement
- **Simplified infrastructure** with Redis replacing Kafka
- **Better resource utilization** and cost optimization

## Phase 1: Pre-Migration Preparation

### 1.1 Backup Current Configuration

```bash
# Backup current Kafka configurations
kubectl get configmaps -n its-camera-ai -o yaml > kafka-configs-backup.yaml
kubectl get secrets -n its-camera-ai -o yaml > kafka-secrets-backup.yaml

# Backup current deployments
kubectl get deployments -n its-camera-ai -o yaml > deployments-backup.yaml
```

### 1.2 Validate Current System Health

```bash
# Check current system metrics
kubectl top nodes
kubectl top pods -n its-camera-ai

# Verify Kafka health
kubectl logs -n its-camera-ai deployment/kafka-deployment --tail=100

# Check current processing metrics
curl http://api-service.its-camera-ai:8000/metrics | grep frames_processed
```

### 1.3 Scale Down Non-Critical Services

```bash
# Reduce replicas for non-critical services during migration
kubectl scale deployment camera-stream-processor --replicas=5 -n its-camera-ai
kubectl scale deployment event-processor --replicas=2 -n its-camera-ai
```

## Phase 2: Redis Infrastructure Deployment

### 2.1 Deploy Redis Streams Infrastructure

```bash
# Apply Redis Streams StatefulSet
kubectl apply -f infrastructure/kubernetes/redis-streams.yaml

# Verify Redis deployment
kubectl get statefulsets -n its-camera-ai
kubectl get pods -l app.kubernetes.io/name=redis-streams -n its-camera-ai

# Check Redis connectivity
kubectl exec -it redis-streams-0 -n its-camera-ai -- redis-cli ping
```

### 2.2 Configure Redis Streams

```bash
# Create consumer groups for each service
kubectl exec -it redis-streams-0 -n its-camera-ai -- redis-cli XGROUP CREATE camera_frames stream_processor 0 MKSTREAM
kubectl exec -it redis-streams-0 -n its-camera-ai -- redis-cli XGROUP CREATE camera_frames ml_inference 0 MKSTREAM
kubectl exec -it redis-streams-0 -n its-camera-ai -- redis-cli XGROUP CREATE processed_frames output_consumers 0 MKSTREAM

# Verify stream configuration
kubectl exec -it redis-streams-0 -n its-camera-ai -- redis-cli XINFO GROUPS camera_frames
```

### 2.3 Deploy Monitoring for Redis

```bash
# Apply Redis monitoring configuration
kubectl apply -f infrastructure/monitoring/redis-streams-monitoring.yaml

# Verify ServiceMonitor is created
kubectl get servicemonitor redis-streams-monitor -n its-camera-ai

# Check Prometheus targets
curl http://prometheus.monitoring:9090/api/v1/targets | jq '.data.activeTargets[] | select(.job=="redis-streams")'
```

## Phase 3: Application Updates

### 3.1 Deploy Updated Stream Processor

```bash
# Build and push updated container images with Redis integration
docker build -t its-camera-ai/stream-processor:redis-v1.0.0 .
docker push its-camera-ai/stream-processor:redis-v1.0.0

# Update deployment with new image and Redis configuration
kubectl apply -f infrastructure/kubernetes/deployments.yaml

# Verify new pods are using Redis
kubectl logs -f deployment/camera-stream-processor -n its-camera-ai
```

### 3.2 Deploy gRPC Services

```bash
# Apply gRPC service configurations
kubectl apply -f infrastructure/kubernetes/grpc-services.yaml

# Verify gRPC services
kubectl get services -l app.kubernetes.io/component=grpc-service -n its-camera-ai

# Test gRPC connectivity
grpcurl -plaintext grpc-stream-processor.its-camera-ai:50051 list
```

### 3.3 Update HPA with Redis Metrics

```bash
# Apply updated HPA configurations
kubectl apply -f infrastructure/kubernetes/hpa.yaml

# Verify HPA is using Redis metrics
kubectl describe hpa ml-inference-redis-scaler -n its-camera-ai
```

## Phase 4: Traffic Migration

### 4.1 Gradual Traffic Shift

```bash
# Start with 10% traffic to Redis-based processors
kubectl patch deployment camera-stream-processor -p '{"spec":{"replicas":2}}' -n its-camera-ai

# Monitor metrics during traffic shift
kubectl exec -it redis-streams-0 -n its-camera-ai -- redis-cli XLEN camera_frames
kubectl exec -it redis-streams-0 -n its-camera-ai -- redis-cli XLEN processed_frames
```

### 4.2 Performance Validation

```bash
# Check processing latency
curl http://api-service.its-camera-ai:8000/metrics | grep processing_time

# Verify queue lengths are manageable
kubectl exec -it redis-streams-0 -n its-camera-ai -- redis-cli XINFO STREAM camera_frames

# Monitor gRPC performance
curl http://grpc-stream-processor.its-camera-ai:9090/metrics | grep grpc_
```

### 4.3 Increase Traffic Gradually

```bash
# Scale to 50% traffic
kubectl patch deployment camera-stream-processor -p '{"spec":{"replicas":5}}' -n its-camera-ai

# Scale to 100% traffic
kubectl patch deployment camera-stream-processor -p '{"spec":{"replicas":10}}' -n its-camera-ai

# Verify all traffic is processed successfully
kubectl logs deployment/camera-stream-processor -n its-camera-ai --tail=100
```

## Phase 5: Kafka Decommission

### 5.1 Stop Kafka Producers

```bash
# Scale down Kafka-dependent services
kubectl scale deployment kafka-producer --replicas=0 -n its-camera-ai

# Verify no new messages in Kafka topics
kubectl exec -it kafka-0 -n its-camera-ai -- kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic camera_frames --timeout-ms 10000
```

### 5.2 Drain Kafka Queues

```bash
# Process remaining Kafka messages
kubectl logs deployment/kafka-consumer -n its-camera-ai --follow

# Verify Kafka topics are empty
kubectl exec -it kafka-0 -n its-camera-ai -- kafka-run-class.sh kafka.tools.GetOffsetShell --broker-list localhost:9092 --topic camera_frames
```

### 5.3 Remove Kafka Infrastructure

```bash
# Remove Kafka deployments
kubectl delete -f infrastructure/kubernetes/kafka-deployment.yaml

# Remove Kafka StatefulSets
kubectl delete statefulset kafka -n its-camera-ai

# Remove Kafka services
kubectl delete service kafka-service -n its-camera-ai

# Clean up Kafka persistent volumes
kubectl delete pvc -l app=kafka -n its-camera-ai
```

## Phase 6: Optimization and Validation

### 6.1 Performance Tuning

```bash
# Optimize Redis configuration based on observed metrics
kubectl exec -it redis-streams-0 -n its-camera-ai -- redis-cli CONFIG SET stream-node-max-entries 200

# Tune HPA thresholds based on actual performance
kubectl patch hpa camera-stream-hpa -p '{"spec":{"metrics":[{"type":"Resource","resource":{"name":"cpu","target":{"type":"Utilization","averageUtilization":60}}}]}}' -n its-camera-ai
```

### 6.2 Validate SLA Compliance

```bash
# Check processing latency targets
kubectl exec -it redis-streams-0 -n its-camera-ai -- redis-cli --latency-history -i 1

# Verify throughput targets
curl http://grpc-stream-processor.its-camera-ai:9090/metrics | grep frames_per_second

# Validate error rates
kubectl logs deployment/camera-stream-processor -n its-camera-ai | grep -c ERROR
```

### 6.3 Load Testing

```bash
# Run load test to validate performance under stress
kubectl apply -f test/load-test-job.yaml

# Monitor system during load test
kubectl top pods -n its-camera-ai
kubectl exec -it redis-streams-0 -n its-camera-ai -- redis-cli INFO stats
```

## Phase 7: Monitoring and Alerting

### 7.1 Configure Production Monitoring

```bash
# Import Redis Streams dashboard to Grafana
curl -X POST http://grafana.monitoring:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @infrastructure/monitoring/redis-streams-dashboard.json

# Verify alerts are firing correctly
curl http://prometheus.monitoring:9090/api/v1/alerts | jq '.data.alerts[] | select(.labels.service=="redis-streams")'
```

### 7.2 Set Up Backup and Recovery

```bash
# Apply backup CronJob
kubectl apply -f infrastructure/terraform/redis-infrastructure.tf

# Test backup process
kubectl create job --from=cronjob/redis-backup redis-backup-test -n its-camera-ai
kubectl logs job/redis-backup-test -n its-camera-ai
```

## Rollback Procedures

### Emergency Rollback to Kafka

If issues are encountered during migration:

```bash
# 1. Scale down Redis-based processors
kubectl scale deployment camera-stream-processor --replicas=0 -n its-camera-ai

# 2. Restore Kafka infrastructure
kubectl apply -f kafka-configs-backup.yaml
kubectl apply -f kafka-secrets-backup.yaml
kubectl apply -f deployments-backup.yaml

# 3. Wait for Kafka to be ready
kubectl wait --for=condition=ready pod -l app=kafka -n its-camera-ai --timeout=300s

# 4. Scale up Kafka-based processors
kubectl scale deployment camera-stream-processor --replicas=10 -n its-camera-ai

# 5. Verify traffic flow
kubectl logs deployment/camera-stream-processor -n its-camera-ai --tail=100
```

## Post-Migration Checklist

- [ ] All services are using Redis Streams for queuing
- [ ] gRPC serialization is active and performing well
- [ ] Processing latency is under 100ms (P95)
- [ ] Throughput targets are met
- [ ] Monitoring and alerting are configured
- [ ] Backup procedures are working
- [ ] Load testing passed
- [ ] Documentation updated
- [ ] Team training completed
- [ ] Kafka infrastructure removed
- [ ] Cost optimization achieved

## Troubleshooting

### Common Issues

1. **High Redis Memory Usage**
   ```bash
   kubectl exec -it redis-streams-0 -n its-camera-ai -- redis-cli CONFIG SET maxmemory-policy volatile-lru
   kubectl exec -it redis-streams-0 -n its-camera-ai -- redis-cli MEMORY PURGE
   ```

2. **Consumer Lag Issues**
   ```bash
   kubectl exec -it redis-streams-0 -n its-camera-ai -- redis-cli XPENDING camera_frames stream_processor
   kubectl scale deployment camera-stream-processor --replicas=15 -n its-camera-ai
   ```

3. **gRPC Connection Issues**
   ```bash
   kubectl logs deployment/camera-stream-processor -n its-camera-ai | grep gRPC
   kubectl describe service grpc-stream-processor -n its-camera-ai
   ```

4. **Performance Degradation**
   ```bash
   kubectl top pods -n its-camera-ai
   kubectl exec -it redis-streams-0 -n its-camera-ai -- redis-cli INFO stats
   ```

## Contacts

- **Platform Engineering Team**: platform@its-camera-ai.com
- **SRE Team**: sre@its-camera-ai.com
- **On-call**: +1-555-CAMERA-AI

## References

- [Redis Streams Documentation](https://redis.io/topics/streams-intro)
- [gRPC Performance Best Practices](https://grpc.io/docs/guides/performance/)
- [Kubernetes HPA Documentation](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [ITS Camera AI Architecture Guide](https://docs.its-camera-ai.com/architecture)