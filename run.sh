#!/bin/bash
docker-compose down && docker-compose up --build -d backend && docker-compose up --build -d frontend
