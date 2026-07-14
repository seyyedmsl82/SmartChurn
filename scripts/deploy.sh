#!/bin/bash
# Deployment script for production

set -e

echo "🚀 Deploying SmartChurn API..."

# Load environment variables
if [ -f .env.production ]; then
    export $(cat .env.production | xargs)
fi

# Run with docker-compose
echo "Starting services..."
docker-compose up -d

# Wait for API to be ready
echo "Waiting for API to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/api/v1/health > /dev/null; then
        echo "API is ready!"
        break
    fi
    sleep 2
done

echo "Deployment complete!"
echo "API available at: http://localhost:8000"
echo "Swagger docs: http://localhost:8000/docs"