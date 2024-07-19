source .env

PROJECT_ID="${PROJECT_ID}"
REGION="${DEFAULT_REGION}"

# # Build and push backend image
gcloud builds submit --tag gcr.io/$PROJECT_ID/backend-api ./backend

# Build and push frontend image
# gcloud builds submit --tag gcr.io/$PROJECT_ID/streamlit-app ./frontend