FROM python:3.9.5-slim-buster

WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install ultralytics fastapi bootstrap-py python-ffmpeg imagehash python-multipart uvicorn[standard]
COPY . .