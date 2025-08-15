# ITS Camera AI - Production Kubernetes Cluster Module
# This module creates a production-ready Kubernetes cluster optimized for AI workloads

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
}

# Data sources for availability zones and VPC
data "aws_availability_zones" "available" {
  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

data "aws_caller_identity" "current" {}

# VPC for the cluster
resource "aws_vpc" "cluster_vpc" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = merge(var.common_tags, {
    Name                                        = "${var.cluster_name}-vpc"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  })
}

# Internet Gateway
resource "aws_internet_gateway" "cluster_igw" {
  vpc_id = aws_vpc.cluster_vpc.id

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-igw"
  })
}

# Public subnets for load balancers
resource "aws_subnet" "public_subnets" {
  count = length(var.public_subnet_cidrs)

  vpc_id                  = aws_vpc.cluster_vpc.id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = merge(var.common_tags, {
    Name                                        = "${var.cluster_name}-public-${count.index + 1}"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/elb"                    = "1"
  })
}

# Private subnets for worker nodes
resource "aws_subnet" "private_subnets" {
  count = length(var.private_subnet_cidrs)

  vpc_id            = aws_vpc.cluster_vpc.id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = merge(var.common_tags, {
    Name                                        = "${var.cluster_name}-private-${count.index + 1}"
    "kubernetes.io/cluster/${var.cluster_name}" = "owned"
    "kubernetes.io/role/internal-elb"           = "1"
  })
}

# NAT Gateways for private subnets
resource "aws_eip" "nat_eips" {
  count = length(var.public_subnet_cidrs)

  domain = "vpc"
  depends_on = [aws_internet_gateway.cluster_igw]

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-nat-eip-${count.index + 1}"
  })
}

resource "aws_nat_gateway" "nat_gateways" {
  count = length(var.public_subnet_cidrs)

  allocation_id = aws_eip.nat_eips[count.index].id
  subnet_id     = aws_subnet.public_subnets[count.index].id
  depends_on    = [aws_internet_gateway.cluster_igw]

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-nat-gateway-${count.index + 1}"
  })
}

# Route table for public subnets
resource "aws_route_table" "public_rt" {
  vpc_id = aws_vpc.cluster_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.cluster_igw.id
  }

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-public-rt"
  })
}

resource "aws_route_table_association" "public_rta" {
  count = length(var.public_subnet_cidrs)

  subnet_id      = aws_subnet.public_subnets[count.index].id
  route_table_id = aws_route_table.public_rt.id
}

# Route tables for private subnets
resource "aws_route_table" "private_rt" {
  count = length(var.private_subnet_cidrs)

  vpc_id = aws_vpc.cluster_vpc.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat_gateways[count.index].id
  }

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-private-rt-${count.index + 1}"
  })
}

resource "aws_route_table_association" "private_rta" {
  count = length(var.private_subnet_cidrs)

  subnet_id      = aws_subnet.private_subnets[count.index].id
  route_table_id = aws_route_table.private_rt[count.index].id
}

# Security group for EKS cluster
resource "aws_security_group" "cluster_sg" {
  name_prefix = "${var.cluster_name}-cluster-sg"
  vpc_id      = aws_vpc.cluster_vpc.id

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-cluster-sg"
  })
}

# Security group for worker nodes
resource "aws_security_group" "node_sg" {
  name_prefix = "${var.cluster_name}-node-sg"
  vpc_id      = aws_vpc.cluster_vpc.id

  ingress {
    description = "Node to node communication"
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    self        = true
  }

  ingress {
    description     = "Cluster API to node groups"
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    security_groups = [aws_security_group.cluster_sg.id]
  }

  ingress {
    description     = "Cluster API to node kubelets"
    from_port       = 10250
    to_port         = 10250
    protocol        = "tcp"
    security_groups = [aws_security_group.cluster_sg.id]
  }

  ingress {
    description     = "Node groups to cluster API"
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    security_groups = [aws_security_group.cluster_sg.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-node-sg"
  })
}

# IAM role for EKS cluster
resource "aws_iam_role" "cluster_role" {
  name = "${var.cluster_name}-cluster-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
      }
    ]
  })

  tags = var.common_tags
}

resource "aws_iam_role_policy_attachment" "cluster_AmazonEKSClusterPolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.cluster_role.name
}

resource "aws_iam_role_policy_attachment" "cluster_AmazonEKSVPCResourceController" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSVPCResourceController"
  role       = aws_iam_role.cluster_role.name
}

# CloudWatch Log Group for EKS cluster
resource "aws_cloudwatch_log_group" "cluster_logs" {
  name              = "/aws/eks/${var.cluster_name}/cluster"
  retention_in_days = var.log_retention_days

  tags = var.common_tags
}

# EKS Cluster
resource "aws_eks_cluster" "cluster" {
  name     = var.cluster_name
  role_arn = aws_iam_role.cluster_role.arn
  version  = var.kubernetes_version

  vpc_config {
    security_group_ids      = [aws_security_group.cluster_sg.id]
    subnet_ids              = concat(aws_subnet.private_subnets[*].id, aws_subnet.public_subnets[*].id)
    endpoint_private_access = true
    endpoint_public_access  = var.endpoint_public_access
    public_access_cidrs     = var.endpoint_public_access_cidrs
  }

  enabled_cluster_log_types = var.cluster_log_types

  encryption_config {
    provider {
      key_arn = aws_kms_key.cluster_encryption.arn
    }
    resources = ["secrets"]
  }

  depends_on = [
    aws_iam_role_policy_attachment.cluster_AmazonEKSClusterPolicy,
    aws_iam_role_policy_attachment.cluster_AmazonEKSVPCResourceController,
    aws_cloudwatch_log_group.cluster_logs,
  ]

  tags = var.common_tags
}

# KMS key for cluster encryption
resource "aws_kms_key" "cluster_encryption" {
  description             = "EKS cluster ${var.cluster_name} encryption key"
  deletion_window_in_days = 7

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-encryption-key"
  })
}

resource "aws_kms_alias" "cluster_encryption" {
  name          = "alias/${var.cluster_name}-encryption"
  target_key_id = aws_kms_key.cluster_encryption.key_id
}

# IAM role for worker nodes
resource "aws_iam_role" "node_role" {
  name = "${var.cluster_name}-node-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = var.common_tags
}

resource "aws_iam_role_policy_attachment" "node_AmazonEKSWorkerNodePolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.node_role.name
}

resource "aws_iam_role_policy_attachment" "node_AmazonEKS_CNI_Policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.node_role.name
}

resource "aws_iam_role_policy_attachment" "node_AmazonEC2ContainerRegistryReadOnly" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.node_role.name
}

# Additional policies for GPU and ML workloads
resource "aws_iam_role_policy" "node_additional_policy" {
  name = "${var.cluster_name}-node-additional-policy"
  role = aws_iam_role.node_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
          "cloudwatch:PutMetricData",
          "ec2:DescribeVolumes",
          "ec2:DescribeSnapshots",
          "ec2:CreateTags",
          "ec2:DescribeInstances",
          "autoscaling:DescribeAutoScalingGroups",
          "autoscaling:DescribeAutoScalingInstances",
          "autoscaling:DescribeLaunchConfigurations",
          "autoscaling:DescribeTags",
          "autoscaling:SetDesiredCapacity",
          "autoscaling:TerminateInstanceInAutoScalingGroup"
        ]
        Resource = "*"
      }
    ]
  })
}

# GPU Node Group for inference workloads
resource "aws_eks_node_group" "gpu_nodes" {
  count = var.enable_gpu_nodes ? 1 : 0

  cluster_name    = aws_eks_cluster.cluster.name
  node_group_name = "${var.cluster_name}-gpu-nodes"
  node_role_arn   = aws_iam_role.node_role.arn
  subnet_ids      = aws_subnet.private_subnets[*].id

  ami_type       = "AL2_x86_64_GPU"
  capacity_type  = "ON_DEMAND"
  instance_types = var.gpu_instance_types

  scaling_config {
    desired_size = var.gpu_nodes_desired_size
    max_size     = var.gpu_nodes_max_size
    min_size     = var.gpu_nodes_min_size
  }

  update_config {
    max_unavailable = 1
  }

  remote_access {
    ec2_ssh_key = var.node_ssh_key
    source_security_group_ids = [aws_security_group.node_sg.id]
  }

  # Taint GPU nodes so only GPU workloads are scheduled
  taint {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  labels = {
    "node.kubernetes.io/instance-type" = "gpu"
    "its-camera-ai/node-type"          = "gpu-inference"
  }

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-gpu-nodes"
  })

  depends_on = [
    aws_iam_role_policy_attachment.node_AmazonEKSWorkerNodePolicy,
    aws_iam_role_policy_attachment.node_AmazonEKS_CNI_Policy,
    aws_iam_role_policy_attachment.node_AmazonEC2ContainerRegistryReadOnly,
  ]
}

# General CPU Node Group
resource "aws_eks_node_group" "general_nodes" {
  cluster_name    = aws_eks_cluster.cluster.name
  node_group_name = "${var.cluster_name}-general-nodes"
  node_role_arn   = aws_iam_role.node_role.arn
  subnet_ids      = aws_subnet.private_subnets[*].id

  ami_type       = "AL2_x86_64"
  capacity_type  = "ON_DEMAND"
  instance_types = var.general_instance_types

  scaling_config {
    desired_size = var.general_nodes_desired_size
    max_size     = var.general_nodes_max_size
    min_size     = var.general_nodes_min_size
  }

  update_config {
    max_unavailable = 2
  }

  remote_access {
    ec2_ssh_key = var.node_ssh_key
    source_security_group_ids = [aws_security_group.node_sg.id]
  }

  labels = {
    "node.kubernetes.io/instance-type" = "general"
    "its-camera-ai/node-type"          = "general-workload"
  }

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-general-nodes"
  })

  depends_on = [
    aws_iam_role_policy_attachment.node_AmazonEKSWorkerNodePolicy,
    aws_iam_role_policy_attachment.node_AmazonEKS_CNI_Policy,
    aws_iam_role_policy_attachment.node_AmazonEC2ContainerRegistryReadOnly,
  ]
}

# Memory-optimized Node Group for Redis, Kafka
resource "aws_eks_node_group" "memory_nodes" {
  count = var.enable_memory_nodes ? 1 : 0

  cluster_name    = aws_eks_cluster.cluster.name
  node_group_name = "${var.cluster_name}-memory-nodes"
  node_role_arn   = aws_iam_role.node_role.arn
  subnet_ids      = aws_subnet.private_subnets[*].id

  ami_type       = "AL2_x86_64"
  capacity_type  = "ON_DEMAND"
  instance_types = var.memory_instance_types

  scaling_config {
    desired_size = var.memory_nodes_desired_size
    max_size     = var.memory_nodes_max_size
    min_size     = var.memory_nodes_min_size
  }

  update_config {
    max_unavailable = 1
  }

  remote_access {
    ec2_ssh_key = var.node_ssh_key
    source_security_group_ids = [aws_security_group.node_sg.id]
  }

  labels = {
    "node.kubernetes.io/instance-type" = "memory-optimized"
    "its-camera-ai/node-type"          = "memory-workload"
  }

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-memory-nodes"
  })

  depends_on = [
    aws_iam_role_policy_attachment.node_AmazonEKSWorkerNodePolicy,
    aws_iam_role_policy_attachment.node_AmazonEKS_CNI_Policy,
    aws_iam_role_policy_attachment.node_AmazonEC2ContainerRegistryReadOnly,
  ]
}

# EKS addons
resource "aws_eks_addon" "vpc_cni" {
  cluster_name = aws_eks_cluster.cluster.name
  addon_name   = "vpc-cni"
  addon_version = var.vpc_cni_version
  resolve_conflicts = "OVERWRITE"

  tags = var.common_tags
}

resource "aws_eks_addon" "coredns" {
  cluster_name = aws_eks_cluster.cluster.name
  addon_name   = "coredns"
  addon_version = var.coredns_version
  resolve_conflicts = "OVERWRITE"

  depends_on = [aws_eks_node_group.general_nodes]

  tags = var.common_tags
}

resource "aws_eks_addon" "kube_proxy" {
  cluster_name = aws_eks_cluster.cluster.name
  addon_name   = "kube-proxy"
  addon_version = var.kube_proxy_version
  resolve_conflicts = "OVERWRITE"

  tags = var.common_tags
}

resource "aws_eks_addon" "ebs_csi" {
  cluster_name = aws_eks_cluster.cluster.name
  addon_name   = "aws-ebs-csi-driver"
  addon_version = var.ebs_csi_version
  resolve_conflicts = "OVERWRITE"

  tags = var.common_tags
}

# OIDC provider for service accounts
data "tls_certificate" "cluster_cert" {
  url = aws_eks_cluster.cluster.identity[0].oidc[0].issuer
}

resource "aws_iam_openid_connect_provider" "cluster_oidc" {
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.cluster_cert.certificates[0].sha1_fingerprint]
  url             = aws_eks_cluster.cluster.identity[0].oidc[0].issuer

  tags = var.common_tags
}