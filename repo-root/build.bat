@echo off
REM PDF Compare - Docker Build and Deployment Script (Windows)

setlocal enabledelayedexpansion

set MODE=%1
set ACTION=%2

if "%MODE%"=="" set MODE=standalone
if "%ACTION%"=="" set ACTION=up

if "%MODE%"=="standalone" goto standalone
if "%MODE%"=="multi-user" goto multiuser
if "%MODE%"=="worker" goto worker
if "%MODE%"=="clean" goto clean
if "%MODE%"=="logs" goto logs
if "%MODE%"=="shell" goto shell
goto help

:standalone
echo [PDF-COMPARE] Building standalone container (all-in-one)
docker-compose build pdf-compare-standalone
if "%ACTION%"=="up" (
    echo [PDF-COMPARE] Starting standalone container...
    docker-compose up -d pdf-compare-standalone
    echo [INFO] Access UI at: http://localhost:8501
)
goto end

:multiuser
echo [PDF-COMPARE] Building multi-user containers (UI + worker)
docker-compose --profile multi-user build
if "%ACTION%"=="up" (
    echo [PDF-COMPARE] Starting multi-user deployment...
    docker-compose --profile multi-user up -d
    echo [INFO] Access UI at: http://localhost:8501
)
goto end

:worker
echo [PDF-COMPARE] Building worker container only
docker-compose --profile multi-user build pdf-compare-worker
if "%ACTION%"=="up" (
    set WORKERS=%3
    if "!WORKERS!"=="" set WORKERS=2
    echo [PDF-COMPARE] Starting !WORKERS! worker container(s)...
    docker-compose --profile multi-user up -d --scale pdf-compare-worker=!WORKERS! pdf-compare-worker
)
goto end

:clean
echo [WARNING] Stopping all containers and removing volumes...
docker-compose --profile multi-user down -v
docker-compose down -v
echo [PDF-COMPARE] Cleanup complete
goto end

:logs
set SERVICE=%3
if "%SERVICE%"=="" set SERVICE=pdf-compare-standalone
docker-compose logs -f %SERVICE%
goto end

:shell
set SERVICE=%3
if "%SERVICE%"=="" set SERVICE=pdf-compare-standalone
docker exec -it %SERVICE% /bin/bash
goto end

:help
echo PDF Compare - Build and Deployment Script
echo.
echo Usage: %0 ^<mode^> ^<action^> [options]
echo.
echo Modes:
echo   standalone     Single container (UI + processing)
echo   multi-user     Separate UI and worker containers
echo   worker         Worker containers only
echo   clean          Stop and remove all containers
echo   logs           View container logs
echo   shell          Open shell in container
echo.
echo Actions:
echo   build          Build images only
echo   up             Build and start containers
echo   down           Stop containers
echo.
echo Examples:
echo   %0 standalone up              # Start all-in-one container
echo   %0 multi-user up              # Start UI + workers
echo   %0 worker up 4                # Start 4 worker containers
echo   %0 logs pdf-compare-worker    # View worker logs
echo   %0 shell pdf-compare-ui       # Open shell in UI container
echo   %0 clean                      # Clean up everything
goto end

:end
echo [PDF-COMPARE] Done!