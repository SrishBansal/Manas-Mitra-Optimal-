#!/bin/bash

# Start the backend server in the background
echo "Starting backend server..."
cd "$(dirname "$0")/api"
source ../venv/bin/activate
uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!

# Give the backend a moment to start
sleep 3

# Start the frontend in the background
echo "Starting frontend..."
cd "$(dirname "$0")/frontend"
npm install
npm run dev &
FRONTEND_PID=$!

# Function to clean up background processes on script exit
cleanup() {
    echo "Shutting down..."
    kill $BACKEND_PID $FRONTEND_PID 2> /dev/null
    exit 0
}

# Set up trap to catch script termination
trap cleanup SIGINT SIGTERM

# Keep the script running
wait $BACKEND_PID $FRONTEND_PID
