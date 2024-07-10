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

# Function to select SQL instance
select_sql_instance() {
    log "Listing SQL instances in the project:"
    INSTANCES=$(gcloud sql instances list --project=${PROJECT_ID} --format="value(name)")
    if [ -z "$INSTANCES" ]; then
        log "No SQL instances found in the project."
        return 1
    fi
    
    PS3="Select a SQL instance: "
    select SQL_INSTANCE in $INSTANCES; do
        if [ -n "$SQL_INSTANCE" ]; then
            log "Selected SQL instance: ${SQL_INSTANCE}"
            return 0
        else
            log "Invalid selection. Please try again."
        fi
    done
}

# Function to delete database
delete_database() {
    local DATABASE_NAME=$1
    log "Deleting database '${DATABASE_NAME}' from SQL instance '${SQL_INSTANCE}'..."
    if ! gcloud sql databases delete ${DATABASE_NAME} \
        --instance=${SQL_INSTANCE} \
        --project=${PROJECT_ID} \
        --quiet; then
        log "Database deletion failed. Checking SQL instance details..."
        gcloud sql instances describe ${SQL_INSTANCE} \
            --format="yaml(name,state)"
        return 1
    fi
    log "Database '${DATABASE_NAME}' deleted successfully"
}

# Function to delete SQL instance
delete_sql_instance() {
    log "Deleting SQL instance '${SQL_INSTANCE}'..."
    if ! gcloud sql instances delete ${SQL_INSTANCE} \
        --project=${PROJECT_ID} \
        --quiet; then
        log "SQL instance deletion failed. Checking instance details..."
        gcloud sql instances describe ${SQL_INSTANCE} \
            --format="yaml(name,state)"
        return 1
    fi
    log "SQL instance '${SQL_INSTANCE}' deleted successfully"
}

# Main menu
while true; do
    if ! select_sql_instance; then
        log "No SQL instances available. Exiting."
        exit 1
    fi

    echo "Select an option:"
    echo "1. Delete a database"
    echo "2. Delete the SQL instance"
    echo "3. Select a different SQL instance"
    echo "4. Exit"
    read -p "Enter your choice (1-4): " choice

    case $choice in
        1)
            # List databases in the instance
            log "Listing databases in the SQL instance:"
            DATABASES=$(gcloud sql databases list --instance=${SQL_INSTANCE} --format="value(name)")
            if [ -z "$DATABASES" ]; then
                log "No databases found in the instance."
                continue
            fi
            log "${DATABASES}"

            # Prompt user to select a database to delete
            PS3="Select a database to delete: "
            select DATABASE_NAME in $DATABASES; do
                if [ -n "$DATABASE_NAME" ]; then
                    break
                else
                    log "Invalid selection. Please try again."
                fi
            done

            # Confirm deletion
            read -p "Are you sure you want to delete the database '${DATABASE_NAME}'? (y/N): " CONFIRM
            if [[ ${CONFIRM,,} == "y" ]]; then
                delete_database "${DATABASE_NAME}"
            else
                log "Database deletion cancelled."
            fi
            ;;
        2)
            # Confirm deletion of SQL instance
            read -p "Are you sure you want to delete the SQL instance '${SQL_INSTANCE}'? This will delete all databases in the instance. (y/N): " CONFIRM
            if [[ ${CONFIRM,,} == "y" ]]; then
                delete_sql_instance
            else
                log "SQL instance deletion cancelled."
            fi
            ;;
        3)
            log "Selecting a different SQL instance..."
            continue
            ;;
        4)
            log "Exiting script."
            exit 0
            ;;
        *)
            log "Invalid choice. Please try again."
            ;;
    esac
done

log "Script completed"