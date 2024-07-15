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

# Verify the active account
log "Active account:"
gcloud config get-value account


# Prompt user to confirm backup before deletion
read -p "Do you want to perform a backup before deletion? (y/N): " BACKUP_CONFIRM
if [[ ${BACKUP_CONFIRM,,} == "y" ]]; then
    log "Performing backup..."
    ./deployment_scripts/db_backup.sh
    log "Backup completed."
fi

# Initialize Terraform
cd deployment_scripts/terraform
log "Initializing Terraform..."
terraform init

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
terraform destroy -target=module.postgres_pgvector_db \
                  -var "ADMIN_PASSWORD=${ADMIN_PASSWORD}" \
                  -var "DEFAULT_PROJECT=${PROJECT_ID}" \
                  -var "DEFAULT_REGION=${DEFAULT_REGION}" \
                  -var "DEFAULT_ZONE=${DEFAULT_ZONE}" \
                  -var "SQL_INSTANCE=${SQL_INSTANCE}" \
                  -var "DATABASE_NAME=${SQL_DATABASE}" \
                  -var "DATABASE_VERSION=${DATABASE_VERSION}" \
                  -var "DELETION_PROTECTION=${DELETION_PROTECTION}" \
                  -var "TIER=${TIER}" \
                  -var "ADMIN_USER=${ADMIN_USER}" \
                  -auto-approve

log "Resources deleted successfully"
log "Script completed"