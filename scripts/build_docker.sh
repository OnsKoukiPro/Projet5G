#!/bin/bash

# Build Docker image
docker-compose -f docker/docker-compose.yml build

echo "Docker image built successfully!"