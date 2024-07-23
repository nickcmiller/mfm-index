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

# Check if the SQL instance already exists
if gcloud sql instances describe ${SQL_INSTANCE} &>/dev/null; then
    log "SQL instance ${SQL_INSTANCE} already exists."
    
    # Check if the resource is already in Terraform state
    if terraform state list | grep -q "module.postgres_pgvector_db.google_sql_database_instance.postgres_pgvector_instance"; then
        log "SQL instance is already in Terraform state. Skipping import."
    else
        log "Importing SQL instance into Terraform state..."
        terraform import \
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
            -var "SQL_DATABASE=${SQL_DATABASE}" \
            -var "GROQ_API_KEY=${GROQ_API_KEY}" \
            -var "OPENAI_API_KEY=${OPENAI_API_KEY}" \
            -var "SQL_HOST=${SQL_HOST}" \
            -var "TABLE_NAME=${TABLE_NAME}" \
            module.postgres_pgvector_db.google_sql_database_instance.postgres_pgvector_instance ${PROJECT_ID}/${DEFAULT_REGION}/${SQL_INSTANCE}
    fi
else
    log "SQL instance ${SQL_INSTANCE} does not exist. It will be created by Terraform."
fi

# # Show current Terraform state
# log "Current Terraform state:"
# terraform show

# # List Terraform-managed resources
# log "Terraform-managed resources:"
# terraform state list

# Prompt user to choose whether to restore a backup
read -p "Do you want to restore a backup? (y/N): " RESTORE_BACKUP
BACKUP_FILE=""
if [[ $RESTORE_BACKUP =~ ^[Yy]$ ]]; then
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
    }' )

    if [ -z "$PROCESSED_OUTPUT" ]; then
        log "No backups found or failed to process backups."
    else
        echo -e "$PROCESSED_OUTPUT \n"
    fi

    # Prompt user to select a backup file
    read -p "Enter the full path of the backup file to restore: " BACKUP_FILE
fi

# Restore Terraform-managed resources
log "Restoring Terraform-managed resources..."
terraform apply -target=module.postgres_pgvector_db \
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
                -var "SQL_DATABASE=${SQL_DATABASE}" \
                -var "GROQ_API_KEY=${GROQ_API_KEY}" \
                -var "OPENAI_API_KEY=${OPENAI_API_KEY}" \
                -var "SQL_HOST=${SQL_HOST}" \
                -var "TABLE_NAME=${TABLE_NAME}" \
                -auto-approve

if [[ $RESTORE_BACKUP =~ ^[Yy]$ && -n "$BACKUP_FILE" ]]; then
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
else
    log "Skipping backup restoration"
fi

log "Script completed"