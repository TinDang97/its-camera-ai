# Bootstrap Variables

variable "aws_region" {
  description = "AWS region for the Terraform state backend"
  type        = string
  default     = "us-west-2"
}

variable "state_bucket_name" {
  description = "Name of the S3 bucket for Terraform state"
  type        = string
  default     = "its-camera-ai-terraform-state"
}

variable "lock_table_name" {
  description = "Name of the DynamoDB table for state locking"
  type        = string
  default     = "its-camera-ai-terraform-locks"
}