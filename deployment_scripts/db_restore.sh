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
log "SQL_INSTANCE: ${SQL_INSTANCE}"

# Verify the active account
log "Active account:"
gcloud config get-value account

# Initialize Terraform
log "Initializing Terraform..."
terraform init

# Show current Terraform state
log "Current Terraform state:"
terraform show

# List Terraform-managed resources
log "Terraform-managed resources:"
terraform state list

# List available backups with creation dates
log "Available backups:"
PROCESSED_OUTPUT=$(gsutil ls -l gs://${SQL_INSTANCE}_backup | awk '{
    if ($1 ~ /^[0-9]+$/) {
        size = $1
        date = $2
        file = $3
        for (i=4; i<=NF; i++) file = file " " $i
        printf "\nDate: %s\nSize: %.2f MB\n%s\n", date, size/1024/1024, file
    }
}')

if [ -z "$PROCESSED_OUTPUT" ]; then
    log "No backups found or failed to process backups."
else
    echo -e "$PROCESSED_OUTPUT \n"
fi

# Prompt user to select a backup file
read -p "Enter the full path of the backup file to restore: " BACKUP_FILE

# Apply Terraform configuration
log "Applying Terraform configuration..."
terraform apply -var "ADMIN_PASSWORD=${SQL_PASSWORD}" \
                -var "DEFAULT_PROJECT=${PROJECT_ID}" \
                -var "DEFAULT_REGION=${DEFAULT_REGION}" \
                -var "DEFAULT_ZONE=${DEFAULT_ZONE}" \
                -var "SQL_INSTANCE=${SQL_INSTANCE}" \
                -var "DATABASE_NAME=${SQL_DATABASE}" \
                -auto-approve


# Get the SQL service account
SQL_SERVICE_ACCOUNT=$(gcloud sql instances describe ${SQL_INSTANCE} --format="value(serviceAccountEmailAddress)")
log "SQL Service Account: ${SQL_SERVICE_ACCOUNT}"

# Set IAM permissions for the service account on the bucket
log "Setting IAM permissions for ${SQL_SERVICE_ACCOUNT} on gs://${SQL_INSTANCE}_backup"
gsutil iam ch serviceAccount:${SQL_SERVICE_ACCOUNT}:objectAdmin gs://${SQL_INSTANCE}_backup

# Verify service account permissions
log "Verifying service account permissions"
gcloud projects get-iam-policy ${PROJECT_ID} \
    --flatten="bindings[].members" \
    --format='table(bindings.role)' \
    --filter="bindings.members:${SQL_SERVICE_ACCOUNT}"

# Verify the backup file exists
if ! gsutil -q stat "${BACKUP_FILE}"; then
    log "Error: Backup file does not exist"
    exit 1
fi

# Perform the SQL import
log "Starting SQL import..."
if ! gcloud sql import sql ${SQL_INSTANCE} ${BACKUP_FILE} \
    --database=${SQL_DATABASE} \
    --quiet; then
    log "SQL import failed. Checking SQL instance details..."
    gcloud sql instances describe ${SQL_INSTANCE} \
        --format="yaml(name,state)"
    exit 1
fi

log "SQL import completed successfully"
log "Script completed"