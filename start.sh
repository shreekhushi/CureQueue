#!/bin/bash

# Install backend packages
pip install -r backend/requirements.txt

# Build frontend
cd frontend
npm install
npm run build
cd ..

# Copy frontend build into backend
cp -r frontend/build backend/build

# Start FastAPI server
uvicorn backend.app:app --host 0.0.0.0 --port $PORT
