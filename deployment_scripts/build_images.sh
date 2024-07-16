source .env

PROJECT_ID="${PROJECT_ID}"
REGION="${DEFAULT_REGION}"

# # Build and push backend image
# docker build -t gcr.io/$PROJECT_ID/backend-api ./backend
# docker push gcr.io/$PROJECT_ID/backend-api

# Build and push frontend image
docker build -t gcr.io/$PROJECT_ID/streamlit-app ./streamlit_app
docker push gcr.io/$PROJECT_ID/streamlit-app