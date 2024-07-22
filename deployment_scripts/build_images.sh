source .env

PROJECT_ID="${PROJECT_ID}"
REGION="${DEFAULT_REGION}"

read -p "Enter one of the following options:

1 for 'backend'
2 for 'frontend'
3 for 'both': " option

if [ "$option" = "1" ]; then
    # Build and push backend image
    gcloud builds submit --tag gcr.io/$PROJECT_ID/backend-api ./backend
elif [ "$option" = "2" ]; then
    # Build and push frontend image
    gcloud builds submit --tag gcr.io/$PROJECT_ID/streamlit-app ./frontend
elif [ "$option" = "3" ]; then
    # Build and push backend image
    gcloud builds submit --tag gcr.io/$PROJECT_ID/backend-api ./backend
    # Build and push frontend image
    gcloud builds submit --tag gcr.io/$PROJECT_ID/streamlit-app ./frontend
else
    echo "Invalid option. Please specify '1' for 'backend', '2' for 'frontend', or '3' for 'both'."
fi