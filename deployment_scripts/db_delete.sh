#!/bin/bash
set -euo pipefail

# Source environment variables
source ../.env

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Print environment variables
log "Environment variables:"
log "PROJECT_ID: ${PROJECT_ID}"

# Verify the active account
log "Active account:"
gcloud config get-value account

# Initialize Terraform
log "Initializing Terraform..."
terraform init

# Prompt user to confirm backup before deletion
read -p "Do you want to perform a backup before deletion? (y/N): " BACKUP_CONFIRM
if [[ ${BACKUP_CONFIRM,,} == "y" ]]; then
    log "Performing backup..."
    ./db_backup.sh
    log "Backup completed."
fi

# Prompt user to confirm deletion
log "Current Terraform state:"
terraform state list

read -p "Are you sure you want to delete the SQL instance and database? This action cannot be undone. (y/N): " CONFIRM
if [[ ${CONFIRM,,} != "y" ]]; then
    log "Deletion cancelled."
    exit 0
fi

# Destroy Terraform-managed resources
log "Destroying Terraform-managed resources..."
terraform destroy -var "ADMIN_PASSWORD=${SQL_PASSWORD}" \
                  -var "DEFAULT_PROJECT=${PROJECT_ID}" \
                  -var "DEFAULT_REGION=${DEFAULT_REGION}" \
                  -var "DEFAULT_ZONE=${DEFAULT_ZONE}" \
                  -var "SQL_INSTANCE=${SQL_INSTANCE}" \
                  -var "DATABASE_NAME=${SQL_DATABASE}" \
                  -auto-approve

log "Resources deleted successfully"
log "Script completed"