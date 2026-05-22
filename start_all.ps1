# PowerShell script to start the Manas Mitra application on Windows (Option A: Gemini API Backend)

Write-Host "=========================================" -ForegroundColor Green
Write-Host "      Starting Manas Mitra Chatbot       " -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host "Mode: Option A - Gemini API Backend" -ForegroundColor Cyan
Write-Host ""

# Start Backend
Write-Host "Starting backend in a new PowerShell window..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "$Host.UI.RawUI.WindowTitle = 'Manas Mitra Backend'; cd api; .\venv\Scripts\Activate.ps1; uvicorn main:app --reload --port 8000"

# Start Frontend
Write-Host "Starting frontend in a new PowerShell window..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "$Host.UI.RawUI.WindowTitle = 'Manas Mitra Frontend'; cd frontend; npm run dev"

Write-Host ""
Write-Host ">>> Application is starting!" -ForegroundColor Green
Write-Host "Access the frontend at:  http://localhost:3000" -ForegroundColor Green
Write-Host "Backend API docs at:    http://localhost:8000/docs" -ForegroundColor Green
Write-Host ""
Write-Host "You can close this main launcher window. The backend and frontend servers will continue running in their respective windows." -ForegroundColor Gray
