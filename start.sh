#!/bin/bash

# Start the application
cd "$(dirname "$0")/frontend"
echo "Starting Manas Mitra..."
echo "Access the application at: http://localhost:3000"

# Start the application in development mode
npm run dev
