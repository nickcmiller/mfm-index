source .env

PROJECT_ID="${PROJECT_ID}"
REGION="${DEFAULT_REGION}"

# # Build and push backend image
# docker build -t gcr.io/$PROJECT_ID/backend-api ./backend
# docker push gcr.io/$PROJECT_ID/backend-api

# Build and push frontend image
gcloud builds submit --tag gcr.io/$PROJECT_ID/streamlit-app ./frontend