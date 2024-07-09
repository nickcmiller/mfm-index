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