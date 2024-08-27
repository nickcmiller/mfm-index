#!/bin/bash
set -euo pipefail

# Source environment variables
source .env

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Print environment variables
log "Environment variables:"
log "PROJECT_ID: ${PROJECT_ID}"
log "SQL_INSTANCE: ${SQL_INSTANCE}"

# Verify the active account
log "Active account:"
gcloud config get-value account

# Initialize Terraform
log "Initializing Terraform..."
cd ./deployment_scripts/terraform
terraform init

# Show current Terraform state
log "Current Terraform state:"
terraform show

# List Terraform-managed resources
log "Terraform-managed resources:"
terraform state list

# Apply all Terraform-managed resources
log "Applying all Terraform-managed resources..."
terraform apply \
    -var "ADMIN_PASSWORD=${ADMIN_PASSWORD}" \
    -var "DEFAULT_PROJECT=${PROJECT_ID}" \
    -var "DEFAULT_REGION=${DEFAULT_REGION}" \
    -var "DEFAULT_ZONE=${DEFAULT_ZONE}" \
    -var "SQL_INSTANCE=${SQL_INSTANCE}" \
    -var "DATABASE_NAME=${SQL_DATABASE}" \
    -var "SQL_DATABASE=${SQL_DATABASE}" \
    -var "DATABASE_VERSION=${DATABASE_VERSION}" \
    -var "DELETION_PROTECTION=${DELETION_PROTECTION}" \
    -var "TIER=${TIER}" \
    -var "ADMIN_USER=${ADMIN_USER}" \
    -var "GROQ_API_KEY=${GROQ_API_KEY}" \
    -var "OPENAI_API_KEY=${OPENAI_API_KEY}" \
    -var "SQL_HOST=${SQL_HOST}" \
    -var "TABLE_NAME=${TABLE_NAME}" \
    -auto-approve

log "Terraform apply completed successfully"
log "Script completed"