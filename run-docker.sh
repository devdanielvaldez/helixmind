#!/usr/bin/env bash
set -e

echo "=========================================================="
echo " Starting HelixMind Enterprise Platform with Docker"
echo "=========================================================="

# Check if Docker daemon is accessible
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker daemon is not running."
    echo "Please launch Docker Desktop or start your Docker service and retry."
    exit 1
fi

echo "Building and starting services via Docker Compose..."
docker-compose up -d --build

echo ""
echo "=========================================================="
echo " HelixMind is up and running!"
echo " Web Application:  http://localhost:8000"
echo " Interactive Docs: http://localhost:8000/docs"
echo " Ollama Server:    http://localhost:11434"
echo " Real-Time Stream: ws://localhost:8000/ws/live"
echo "=========================================================="
echo "=========================================================="
echo "To run CLI inside Docker:       docker exec -it helixmind-app helixmind"
echo "To run investigation in Docker: docker exec -it helixmind-app helixmind investigate \"<topic>\""
echo "To view live streaming logs:    docker-compose logs -f"
echo "To stop all services:           docker-compose down"
echo "=========================================================="
