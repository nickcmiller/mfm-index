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

# Get the correct service account for the SQL instance
SQL_SERVICE_ACCOUNT=$(gcloud sql instances describe ${SQL_INSTANCE} --format="value(serviceAccountEmailAddress)")
log "SQL Service Account: ${SQL_SERVICE_ACCOUNT}"

# Verify the active account
log "Active account:"
gcloud config get-value account

# Check if the bucket exists, create if it doesn't
if ! gsutil ls -b gs://${SQL_INSTANCE}_backup >/dev/null 2>&1; then
    log "Creating bucket gs://${SQL_INSTANCE}_backup"
    gsutil mb -p ${PROJECT_ID} -l us-west1 gs://${SQL_INSTANCE}_backup
else
    log "Bucket gs://${SQL_INSTANCE}_backup already exists"
fi

# Set IAM permissions for the service account on the bucket
log "Setting IAM permissions for ${SQL_SERVICE_ACCOUNT} on gs://${SQL_INSTANCE}_backup"
gsutil iam ch serviceAccount:${SQL_SERVICE_ACCOUNT}:objectAdmin gs://${SQL_INSTANCE}_backup

# Verify service account permissions
log "Verifying service account permissions"
gcloud projects get-iam-policy ${PROJECT_ID} \
    --flatten="bindings[].members" \
    --format='table(bindings.role)' \
    --filter="bindings.members:${SQL_SERVICE_ACCOUNT}"

# List databases in the instance
log "Listing databases in the SQL instance:"
DATABASES=$(gcloud sql databases list --instance=${SQL_INSTANCE} --format="value(name)")
log "${DATABASES}"

# Prompt user to select a database
read -p "Enter the name of the database to backup: " DATABASE_NAME

# Prompt user for backup name suffix
read -p "Enter a suffix for the backup name (default: '_backup'): " BACKUP_SUFFIX
BACKUP_SUFFIX=${BACKUP_SUFFIX:-"_backup"}

# Perform the SQL export
log "Starting SQL export..."
if ! gcloud sql export sql ${SQL_INSTANCE} \
    gs://${SQL_INSTANCE}_backup/${DATABASE_NAME}${BACKUP_SUFFIX}_$(date +'%Y%m%d_%H%M%S').gz \
    --database=${DATABASE_NAME} \
    --offload; then
    log "SQL export failed. Checking SQL instance details..."
    gcloud sql instances describe ${SQL_INSTANCE} \
        --format="yaml(name,serviceAccountEmailAddress,state)"
    exit 1
fi

log "SQL export completed successfully"
log "Script completed"