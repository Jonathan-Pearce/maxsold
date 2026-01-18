@echo off
REM MaxSold Docker Helper Script for Windows
REM This script provides convenient commands for working with the MaxSold Docker container

setlocal EnableDelayedExpansion

set IMAGE_NAME=maxsold:latest
set CONTAINER_NAME=maxsold-container

if "%1"=="" goto :usage
if "%1"=="build" goto :build
if "%1"=="shell" goto :shell
if "%1"=="scraper" goto :scraper
if "%1"=="ml-minimal" goto :ml_minimal
if "%1"=="ml-fast" goto :ml_fast
if "%1"=="ml-quick" goto :ml_quick
if "%1"=="ml-complete" goto :ml_complete
if "%1"=="verify" goto :verify
if "%1"=="test" goto :test
if "%1"=="clean" goto :clean
if "%1"=="logs" goto :logs
goto :usage

:usage
echo MaxSold Docker Helper Script
echo.
echo Usage: docker-run.bat [command]
echo.
echo Commands:
echo   build                  - Build the Docker image
echo   shell                  - Start an interactive shell in the container
echo   scraper [script]       - Run a specific scraper script
echo   ml-minimal             - Train minimal ML model (fastest)
echo   ml-fast                - Train ML model with visualizations
echo   ml-quick               - Train ML model with full dataset
echo   ml-complete            - Run complete ML pipeline
echo   verify                 - Verify model setup
echo   test                   - Run test modules
echo   clean                  - Remove containers and clean up
echo   logs                   - View container logs
echo.
echo Examples:
echo   docker-run.bat build
echo   docker-run.bat shell
echo   docker-run.bat scraper monthly_scraping_pipeline.py
echo   docker-run.bat ml-minimal
goto :eof

:build
echo Building Docker image...
docker build -t %IMAGE_NAME% .
echo Image built successfully!
goto :eof

:shell
echo Starting interactive shell...
docker run -it --rm -v "%cd%\data:/app/data" --name %CONTAINER_NAME% %IMAGE_NAME% bash
goto :eof

:scraper
if "%2"=="" (
    echo Please specify a scraper script
    echo Example: docker-run.bat scraper 01_extract_auction_search.py
    goto :eof
)
echo Running scraper: %2
docker run -it --rm -v "%cd%\data:/app/data" --name %CONTAINER_NAME% %IMAGE_NAME% python scrapers/%2
goto :eof

:ml_minimal
echo Training minimal ML model (30-60 seconds)...
docker run -it --rm -v "%cd%\data:/app/data" --name %CONTAINER_NAME% %IMAGE_NAME% python ml_pipeline/scripts/train_model_minimal.py
goto :eof

:ml_fast
echo Training ML model with visualizations (1-2 minutes)...
docker run -it --rm -v "%cd%\data:/app/data" --name %CONTAINER_NAME% %IMAGE_NAME% python ml_pipeline/scripts/model_pipeline_fast.py
goto :eof

:ml_quick
echo Training ML model with full dataset (2-3 minutes)...
docker run -it --rm -v "%cd%\data:/app/data" --name %CONTAINER_NAME% %IMAGE_NAME% python ml_pipeline/scripts/model_pipeline_quick.py
goto :eof

:ml_complete
echo Running complete ML pipeline (5-10 minutes)...
docker run -it --rm -v "%cd%\data:/app/data" --name %CONTAINER_NAME% %IMAGE_NAME% python ml_pipeline/scripts/model_pipeline.py
goto :eof

:verify
echo Verifying model setup...
docker run -it --rm -v "%cd%\data:/app/data" --name %CONTAINER_NAME% %IMAGE_NAME% python ml_pipeline/utils/verify_model_setup.py
goto :eof

:test
echo Running test modules...
docker run -it --rm --name %CONTAINER_NAME% %IMAGE_NAME% python utils/test_modules.py
goto :eof

:clean
echo Cleaning up containers...
REM Use docker filter to get containers with names containing maxsold
set "found=0"
for /f "tokens=*" %%i in ('docker ps -a -q --filter "name=maxsold" 2^>nul') do (
    set "found=1"
    docker rm -f %%i 2>nul
)
if "%found%"=="1" (
    echo Cleanup complete!
) else (
    echo No maxsold containers to clean up
)
goto :eof

:logs
echo Viewing logs...
docker logs %CONTAINER_NAME%
goto :eof
