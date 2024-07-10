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

# Check if SQL instance exists
if ! gcloud sql instances describe ${SQL_INSTANCE} &>/dev/null; then
    log "SQL instance ${SQL_INSTANCE} does not exist."
    read -p "Do you want to create the SQL instance? (y/n): " create_instance
    if [[ ${create_instance,,} == "y" ]]; then
        log "Creating SQL instance ${SQL_INSTANCE}..."
        gcloud sql instances create ${SQL_INSTANCE} --region=us-central1
        log "SQL instance created successfully."
    else
        log "SQL instance creation skipped. Exiting script."
        exit 1
    fi
fi

# List available backups
log "Available backups:"
gsutil ls gs://${SQL_INSTANCE}_backup

# Prompt user to select a backup file
read -p "Enter the full path of the backup file to restore: " BACKUP_FILE

# Verify the backup file exists
if ! gsutil -q stat "${BACKUP_FILE}"; then
    log "Error: Backup file does not exist"
    exit 1
fi

# Perform the SQL import
log "Starting SQL import..."
if ! gcloud sql import sql ${SQL_INSTANCE} ${BACKUP_FILE} \
    --database=${SQL_INSTANCE} \
    --quiet; then
    log "SQL import failed. Checking SQL instance details..."
    gcloud sql instances describe ${SQL_INSTANCE} \
        --format="yaml(name,state)"
    exit 1
fi

log "SQL import completed successfully"
log "Script completed"