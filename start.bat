@echo off
REM start.bat — launches the backend (FastAPI) and frontend (Vite) dev
REM servers together, each in its own window, from wherever this file
REM lives, regardless of the working directory it's run from.

setlocal
set "ROOT=%~dp0"

REM Make sure Node.js is found even if it isn't on the system PATH yet
REM (e.g. right after installation, before a PATH refresh).
set "PATH=C:\Program Files\nodejs;%PATH%"

echo Starting backend (FastAPI) on http://localhost:8000 ...
start "Vet Matching - Backend" cmd /k "cd /d "%ROOT%" && py -3 -m uvicorn main:app --reload --port 8000 --app-dir backend"

echo Starting frontend (Vite) on http://localhost:5173 ...
start "Vet Matching - Frontend" cmd /k "cd /d "%ROOT%frontend" && npm run dev"

echo.
echo Both servers are starting in their own windows:
echo   Backend:  http://localhost:8000  (docs at /docs)
echo   Frontend: http://localhost:5173
echo.
echo Close a window (or press Ctrl+C inside it) to stop that server.
echo.
pause
