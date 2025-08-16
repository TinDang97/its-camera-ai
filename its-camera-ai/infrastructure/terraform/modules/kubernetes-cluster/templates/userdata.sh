#!/bin/bash
# User data script for EKS nodes
# This script configures the node to join the EKS cluster

set -o xtrace

# Update system packages
yum update -y

# Install additional packages for ITS Camera AI
yum install -y \
    docker \
    awscli \
    jq \
    htop \
    iotop \
    sysstat \
    nvidia-docker2

# Configure Docker daemon
cat > /etc/docker/daemon.json <<EOF
{
    "exec-opts": ["native.cgroupdriver=systemd"],
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "100m",
        "max-file": "3"
    },
    "storage-driver": "overlay2",
    "storage-opts": [
        "overlay2.override_kernel_check=true"
    ],
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF

# Configure kubelet
mkdir -p /etc/kubernetes/kubelet
cat > /etc/kubernetes/kubelet/kubelet-config.json <<EOF
{
    "kind": "KubeletConfiguration",
    "apiVersion": "kubelet.config.k8s.io/v1beta1",
    "address": "0.0.0.0",
    "port": 10250,
    "readOnlyPort": 0,
    "cgroupDriver": "systemd",
    "hairpinMode": "promiscuous-bridge",
    "serializeImagePulls": false,
    "featureGates": {
        "RotateKubeletServerCertificate": true
    },
    "serverTLSBootstrap": true,
    "authentication": {
        "x509": {
            "clientCAFile": "/etc/kubernetes/pki/ca.crt"
        },
        "webhook": {
            "enabled": true,
            "cacheTTL": "0s"
        },
        "anonymous": {
            "enabled": false
        }
    },
    "authorization": {
        "mode": "Webhook",
        "webhook": {
            "cacheAuthorizedTTL": "0s",
            "cacheUnauthorizedTTL": "0s"
        }
    },
    "registryPullQPS": 5,
    "registryBurst": 10,
    "eventRecordQPS": 0,
    "eventBurst": 10,
    "enableDebuggingHandlers": true,
    "enableContentionProfiling": true,
    "healthzPort": 10248,
    "healthzBindAddress": "127.0.0.1",
    "clusterDNS": ["172.20.0.10"],
    "clusterDomain": "cluster.local",
    "streamingConnectionIdleTimeout": "4h0m0s",
    "nodeStatusUpdateFrequency": "10s",
    "nodeStatusReportFrequency": "1m0s",
    "imageMinimumGCAge": "2m0s",
    "imageGCHighThresholdPercent": 85,
    "imageGCLowThresholdPercent": 80,
    "volumeStatsAggPeriod": "1m0s",
    "kubeletCgroups": "/systemd/system.slice",
    "systemCgroups": "/systemd/system.slice",
    "cgroupRoot": "/",
    "cgroupsPerQOS": true,
    "cgroupDriver": "systemd",
    "runtimeRequestTimeout": "2m0s",
    "hairpinMode": "promiscuous-bridge",
    "maxPods": 110,
    "podCIDR": "10.244.0.0/16",
    "podPidsLimit": -1,
    "resolvConf": "/run/systemd/resolve/resolv.conf",
    "cpuManagerPolicy": "none",
    "cpuManagerPolicyOptions": {},
    "topologyManagerPolicy": "none",
    "topologyManagerScope": "container",
    "runtimeClass": "",
    "systemReserved": {
        "cpu": "100m",
        "memory": "100Mi",
        "ephemeral-storage": "1Gi"
    },
    "kubeReserved": {
        "cpu": "100m",
        "memory": "100Mi",
        "ephemeral-storage": "1Gi"
    },
    "enforceNodeAllocatable": ["pods"],
    "allowedUnsafeSysctls": [],
    "volumePluginDir": "/usr/libexec/kubernetes/kubelet-plugins/volume/exec/",
    "logging": {
        "format": "json"
    }
}
EOF

# Configure system parameters for performance
cat > /etc/sysctl.d/99-its-camera-ai.conf <<EOF
# Network performance tuning
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.ipv4.tcp_window_scaling = 1
net.ipv4.tcp_timestamps = 1
net.ipv4.tcp_sack = 1
net.core.netdev_max_backlog = 5000

# Memory management
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5

# File system
fs.file-max = 1000000
fs.inotify.max_user_watches = 524288
fs.inotify.max_user_instances = 256

# Process limits
kernel.pid_max = 4194304
kernel.threads-max = 1000000
EOF

# Apply system parameters
sysctl -p /etc/sysctl.d/99-its-camera-ai.conf

# Set up logging
mkdir -p /var/log/its-camera-ai
chown -R ec2-user:ec2-user /var/log/its-camera-ai

# Configure log rotation
cat > /etc/logrotate.d/its-camera-ai <<EOF
/var/log/its-camera-ai/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    sharedscripts
    postrotate
        /bin/systemctl reload rsyslog > /dev/null 2>&1 || true
    endscript
}
EOF

# Install Node Exporter for monitoring
NODE_EXPORTER_VERSION="1.6.1"
cd /tmp
wget https://github.com/prometheus/node_exporter/releases/download/v$${NODE_EXPORTER_VERSION}/node_exporter-$${NODE_EXPORTER_VERSION}.linux-amd64.tar.gz
tar xvf node_exporter-$${NODE_EXPORTER_VERSION}.linux-amd64.tar.gz
cp node_exporter-$${NODE_EXPORTER_VERSION}.linux-amd64/node_exporter /usr/local/bin/
rm -rf node_exporter-*

# Create node_exporter service
cat > /etc/systemd/system/node_exporter.service <<EOF
[Unit]
Description=Node Exporter
After=network.target

[Service]
Type=simple
User=nobody
ExecStart=/usr/local/bin/node_exporter --web.listen-address=:9100
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Enable and start node_exporter
systemctl daemon-reload
systemctl enable node_exporter
systemctl start node_exporter

# Install GPU monitoring (if GPU instance)
if lspci | grep -i nvidia > /dev/null 2>&1; then
    echo "GPU detected, installing NVIDIA monitoring tools"
    
    # Install NVIDIA Docker runtime
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    
    # Install DCGM exporter for GPU metrics
    DCGM_EXPORTER_VERSION="2.4.0"
    docker run -d --restart=unless-stopped \
        --name dcgm-exporter \
        --gpus all \
        -p 9400:9400 \
        nvcr.io/nvidia/k8s/dcgm-exporter:$${DCGM_EXPORTER_VERSION}-2.6.0-ubuntu20.04
    
    # Configure GPU performance
    nvidia-smi -pm 1  # Enable persistent mode
    nvidia-smi -ac 2505,875  # Set memory and graphics clocks (adjust based on GPU)
fi

# Bootstrap the node to join the EKS cluster
/etc/eks/bootstrap.sh ${cluster_name} \
    --apiserver-endpoint ${cluster_endpoint} \
    --b64-cluster-ca ${cluster_ca} \
    --container-runtime containerd \
    ${bootstrap_arguments}

# Set up custom metrics collection
cat > /opt/its-camera-ai-metrics.sh <<'EOF'
#!/bin/bash
# Custom metrics collection for ITS Camera AI

METRICS_DIR="/var/lib/node_exporter/textfile_collector"
mkdir -p $METRICS_DIR

while true; do
    # Collect custom application metrics
    echo "# HELP its_camera_ai_node_info Node information for ITS Camera AI" > $METRICS_DIR/its_camera_ai.prom
    echo "# TYPE its_camera_ai_node_info gauge" >> $METRICS_DIR/its_camera_ai.prom
    echo "its_camera_ai_node_info{instance_type=\"$(curl -s http://169.254.169.254/latest/meta-data/instance-type)\",availability_zone=\"$(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone)\"} 1" >> $METRICS_DIR/its_camera_ai.prom
    
    # Collect disk usage for critical paths
    df -h /var/lib/docker /var/lib/kubelet /var/log | tail -n +2 | while read line; do
        filesystem=$(echo $line | awk '{print $1}')
        mount=$(echo $line | awk '{print $6}')
        used_percent=$(echo $line | awk '{print $5}' | sed 's/%//')
        echo "its_camera_ai_disk_usage_percent{filesystem=\"$filesystem\",mount=\"$mount\"} $used_percent" >> $METRICS_DIR/its_camera_ai.prom
    done
    
    sleep 60
done
EOF

chmod +x /opt/its-camera-ai-metrics.sh

# Create systemd service for custom metrics
cat > /etc/systemd/system/its-camera-ai-metrics.service <<EOF
[Unit]
Description=ITS Camera AI Custom Metrics
After=node_exporter.service

[Service]
Type=simple
User=nobody
ExecStart=/opt/its-camera-ai-metrics.sh
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Enable and start custom metrics service
systemctl daemon-reload
systemctl enable its-camera-ai-metrics.service
systemctl start its-camera-ai-metrics.service

# Create cleanup script
cat > /opt/cleanup-docker.sh <<'EOF'
#!/bin/bash
# Cleanup script for Docker resources

# Remove stopped containers
docker container prune -f

# Remove unused images (keep images from last 24 hours)
docker image prune -a --filter "until=24h" -f

# Remove unused volumes
docker volume prune -f

# Remove unused networks
docker network prune -f

# Clean up logs
find /var/log/containers/ -name "*.log" -type f -mtime +7 -delete
find /var/log/pods/ -name "*.log" -type f -mtime +7 -delete

# Clean up kubelet cache
rm -rf /var/lib/kubelet/cpu_manager_state
rm -rf /var/lib/kubelet/memory_manager_state

echo "Docker cleanup completed at $(date)"
EOF

chmod +x /opt/cleanup-docker.sh

# Schedule cleanup script
echo "0 2 * * * root /opt/cleanup-docker.sh >> /var/log/docker-cleanup.log 2>&1" >> /etc/crontab

# Set up health checks
cat > /opt/node-health-check.sh <<'EOF'
#!/bin/bash
# Node health check script

HEALTH_FILE="/var/lib/kubelet/health"
mkdir -p /var/lib/kubelet

# Check critical services
services_healthy=true

# Check Docker
if ! systemctl is-active --quiet docker; then
    echo "Docker is not running" >&2
    services_healthy=false
fi

# Check kubelet
if ! systemctl is-active --quiet kubelet; then
    echo "Kubelet is not running" >&2
    services_healthy=false
fi

# Check node exporter
if ! systemctl is-active --quiet node_exporter; then
    echo "Node exporter is not running" >&2
    services_healthy=false
fi

# Check disk space
if df / | tail -1 | awk '{print $5}' | sed 's/%//' | awk '$1 > 90'; then
    echo "Root disk usage is above 90%" >&2
    services_healthy=false
fi

if df /var/lib/docker | tail -1 | awk '{print $5}' | sed 's/%//' | awk '$1 > 85'; then
    echo "Docker disk usage is above 85%" >&2
    services_healthy=false
fi

# Update health status
if [ "$services_healthy" = true ]; then
    echo "healthy" > $HEALTH_FILE
    exit 0
else
    echo "unhealthy" > $HEALTH_FILE
    exit 1
fi
EOF

chmod +x /opt/node-health-check.sh

# Schedule health checks
echo "*/5 * * * * root /opt/node-health-check.sh" >> /etc/crontab

# Configure automatic updates for security patches
yum install -y yum-cron
systemctl enable yum-cron
systemctl start yum-cron

# Configure yum-cron for security updates only
sed -i 's/apply_updates = no/apply_updates = yes/' /etc/yum/yum-cron.conf
sed -i 's/update_cmd = default/update_cmd = security/' /etc/yum/yum-cron.conf

# Final system configuration
echo "Node configuration completed at $(date)" >> /var/log/user-data.log

# Signal completion
/opt/aws/bin/cfn-signal -e $? --stack $${AWS::StackName} --resource NodeGroup --region $${AWS::Region} || true