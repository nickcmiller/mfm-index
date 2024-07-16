FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN apt update && apt install ffmpeg -y
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "backend_api:app", "--host", "0.0.0.0", "--port", "8000"]