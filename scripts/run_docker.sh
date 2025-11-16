#!/bin/bash

# Run Docker container
docker-compose -f docker/docker-compose.yml up -d

# Attach to container
docker exec -it coop-perception bash