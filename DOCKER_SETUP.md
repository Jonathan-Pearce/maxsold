# Docker Setup Guide for MaxSold Data Project

This guide provides detailed instructions for setting up and running the MaxSold Data Project using Docker.

## Quick Start

**For the impatient:** The repository includes helper scripts for easy Docker management:

```bash
# Linux/Mac
git clone https://github.com/Jonathan-Pearce/maxsold.git
cd maxsold
chmod +x docker-run.sh
./docker-run.sh build
./docker-run.sh shell
```

```cmd
REM Windows
git clone https://github.com/Jonathan-Pearce/maxsold.git
cd maxsold
docker-run.bat build
docker-run.bat shell
```

Continue reading for detailed instructions and all available options.

## Table of Contents
- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Building the Docker Image](#building-the-docker-image)
- [Running the Container](#running-the-container)
- [Helper Scripts Reference](#helper-scripts-reference)
- [Common Use Cases](#common-use-cases)
- [Data Persistence](#data-persistence)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

---

## Prerequisites

Before you begin, ensure you have the following installed on your local computer:

1. **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)
   - Download from: https://www.docker.com/products/docker-desktop
   - Minimum version: Docker 20.10 or higher
   - Verify installation: `docker --version`

2. **Git** (optional, for cloning the repository)
   - Download from: https://git-scm.com/downloads
   - Verify installation: `git --version`

3. **System Requirements**
   - At least 4GB of free disk space
   - At least 2GB of RAM available for Docker
   - Internet connection (for downloading dependencies)

---

## Building the Docker Image

### Step 1: Clone or Download the Repository

If you have Git installed:
```bash
git clone https://github.com/Jonathan-Pearce/maxsold.git
cd maxsold
```

Or download the repository as a ZIP file from GitHub and extract it.

### Step 2: Build the Docker Image

### Quick Start with Helper Scripts

For the easiest experience, use the provided helper scripts:

**Linux/Mac:**
```bash
chmod +x docker-run.sh
./docker-run.sh build
```

**Windows:**
```cmd
docker-run.bat build
```

### Manual Build (Alternative)

Navigate to the project root directory (where the `Dockerfile` is located) and run:

```bash
docker build -t maxsold:latest .
```

**Explanation:**
- `docker build` - Docker command to build an image
- `-t maxsold:latest` - Tags the image with name "maxsold" and version "latest"
- `.` - Specifies the current directory as the build context

**Expected output:**
- The build process will take 5-10 minutes depending on your internet speed
- You'll see multiple "Step X/Y" messages as Docker builds each layer
- Final message should be: "Successfully tagged maxsold:latest"

### Step 3: Verify the Image

Check that the image was created successfully:

```bash
docker images | grep maxsold
```

You should see output similar to:
```
maxsold      latest      abc123def456   2 minutes ago   1.2GB
```

---

## Running the Container

### Using Helper Scripts (Easiest Method)

The repository includes helper scripts that simplify common Docker operations:

**Linux/Mac - `docker-run.sh`:**
```bash
# Interactive shell
./docker-run.sh shell

# Run scrapers
./docker-run.sh scraper monthly_scraping_pipeline.py

# Train ML models
./docker-run.sh ml-minimal      # Fastest (30-60 seconds)
./docker-run.sh ml-fast         # With visualizations (1-2 minutes)
./docker-run.sh ml-quick        # Full dataset (2-3 minutes)
./docker-run.sh ml-complete     # Complete pipeline (5-10 minutes)

# Other commands
./docker-run.sh verify          # Verify setup
./docker-run.sh test            # Run tests
./docker-run.sh clean           # Clean up containers
```

**Windows - `docker-run.bat`:**
```cmd
REM Interactive shell
docker-run.bat shell

REM Run scrapers
docker-run.bat scraper monthly_scraping_pipeline.py

REM Train ML models
docker-run.bat ml-minimal      REM Fastest (30-60 seconds)
docker-run.bat ml-fast         REM With visualizations (1-2 minutes)
docker-run.bat ml-quick        REM Full dataset (2-3 minutes)
docker-run.bat ml-complete     REM Complete pipeline (5-10 minutes)

REM Other commands
docker-run.bat verify          REM Verify setup
docker-run.bat test            REM Run tests
docker-run.bat clean           REM Clean up containers
```

### Using Docker Compose (Alternative)

The repository includes a `docker-compose.yml` file for managing multiple services:

```bash
# Start an interactive container
docker-compose run --rm maxsold bash

# Run a specific scraper in the background
docker-compose up -d scraper

# Train ML model
docker-compose run --rm ml-training

# View logs
docker-compose logs -f scraper

# Stop all services
docker-compose down
```

### Manual Docker Commands

#### Basic Container Run

To start a container with an interactive bash shell:

```bash
docker run -it --rm maxsold bash
```

**Explanation:**
- `docker run` - Creates and starts a new container
- `-it` - Interactive mode with terminal
- `--rm` - Automatically removes the container when it exits
- `maxsold` - The image name
- `bash` - Command to run inside the container

#### With Data Persistence (Recommended)

To mount a local directory for data persistence:

```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  maxsold bash
```

**For Windows PowerShell:**
```powershell
docker run -it --rm `
  -v ${PWD}/data:/app/data `
  maxsold bash
```

**For Windows Command Prompt:**
```cmd
docker run -it --rm -v %cd%/data:/app/data maxsold bash
```

---

## Helper Scripts Reference

The repository includes convenience scripts to simplify Docker usage. These scripts handle volume mounting and common tasks automatically.

### Available Commands

Both `docker-run.sh` (Linux/Mac) and `docker-run.bat` (Windows) support the following commands:

| Command | Description | Typical Duration |
|---------|-------------|-----------------|
| `build` | Build the Docker image | 5-10 minutes |
| `shell` | Start interactive bash shell | Instant |
| `scraper <script>` | Run a specific scraper | Varies |
| `ml-minimal` | Train minimal ML model | 30-60 seconds |
| `ml-fast` | Train with visualizations | 1-2 minutes |
| `ml-quick` | Train with full dataset | 2-3 minutes |
| `ml-complete` | Complete ML pipeline | 5-10 minutes |
| `verify` | Verify model setup | <10 seconds |
| `test` | Run test modules | <30 seconds |
| `clean` | Remove all containers | <5 seconds |
| `logs` | View container logs | Instant |

### Usage Examples

**Linux/Mac:**
```bash
# Make script executable (first time only)
chmod +x docker-run.sh

# Use any command
./docker-run.sh <command> [args]
```

**Windows:**
```cmd
docker-run.bat <command> [args]
```

### Features

- ✅ Automatically mounts the `data` directory for persistence
- ✅ Uses `--rm` flag to clean up containers after use
- ✅ Provides clear, colored output (Linux/Mac)
- ✅ Handles all common use cases with simple commands
- ✅ Works across Windows, Mac, and Linux

---

## Common Use Cases

### 1. Running Web Scrapers

#### Using Helper Scripts (Recommended)

**Linux/Mac:**
```bash
./docker-run.sh scraper 01_extract_auction_search.py
./docker-run.sh scraper 02_extract_auction_details.py
./docker-run.sh scraper monthly_scraping_pipeline.py
```

**Windows:**
```cmd
docker-run.bat scraper 01_extract_auction_search.py
docker-run.bat scraper 02_extract_auction_details.py
docker-run.bat scraper monthly_scraping_pipeline.py
```

#### Manual Commands

##### Extract Auction Search Data
```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  maxsold \
  python scrapers/01_extract_auction_search.py
```

#### Extract Auction Details
```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  maxsold \
  python scrapers/02_extract_auction_details.py
```

#### Extract Item Details
```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  maxsold \
  python scrapers/03_extract_items_details.py
```

#### Extract Bid History
```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  maxsold \
  python scrapers/04_extract_bid_history.py
```

#### Run Complete Monthly Scraping Pipeline
```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -e KAGGLE_USERNAME=your_username \
  -e KAGGLE_KEY=your_api_key \
  maxsold \
  python scrapers/monthly_scraping_pipeline.py
```

**Note:** For Kaggle integration, you need to set environment variables for your Kaggle credentials.

### 2. Running Machine Learning Pipeline

#### Using Helper Scripts (Recommended)

**Linux/Mac:**
```bash
./docker-run.sh ml-minimal      # Fastest - 30-60 seconds
./docker-run.sh ml-fast         # With visualizations - 1-2 minutes
./docker-run.sh ml-quick        # Full dataset - 2-3 minutes
./docker-run.sh ml-complete     # Complete pipeline - 5-10 minutes
```

**Windows:**
```cmd
docker-run.bat ml-minimal      REM Fastest - 30-60 seconds
docker-run.bat ml-fast         REM With visualizations - 1-2 minutes
docker-run.bat ml-quick        REM Full dataset - 2-3 minutes
docker-run.bat ml-complete     REM Complete pipeline - 5-10 minutes
```

#### Manual Commands

##### Minimal Model Training (Fastest - 30-60 seconds)
```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  maxsold \
  python ml_pipeline/scripts/train_model_minimal.py
```

#### Quick Pipeline with Visualizations (1-2 minutes)
```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  maxsold \
  python ml_pipeline/scripts/model_pipeline_fast.py
```

#### Full Dataset Training (2-3 minutes)
```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  maxsold \
  python ml_pipeline/scripts/model_pipeline_quick.py
```

#### Complete Pipeline (5-10 minutes)
```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  maxsold \
  python ml_pipeline/scripts/model_pipeline.py
```

### 3. Verify Setup

**Using Helper Script:**
```bash
# Linux/Mac
./docker-run.sh verify

# Windows
docker-run.bat verify
```

**Manual Command:**
```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  maxsold \
  python ml_pipeline/utils/verify_model_setup.py
```

### 4. Testing Modules

**Using Helper Script:**
```bash
# Linux/Mac
./docker-run.sh test

# Windows
docker-run.bat test
```

**Manual Command:**
```bash
docker run -it --rm maxsold python utils/test_modules.py
```

### 5. Interactive Development

**Using Helper Script:**
```bash
# Linux/Mac
./docker-run.sh shell

# Windows
docker-run.bat shell
```

**Manual Command for exploratory work or debugging:**

```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/scrapers:/app/scrapers \
  -v $(pwd)/ml_pipeline:/app/ml_pipeline \
  maxsold bash
```

Once inside, you can run any Python scripts:
```bash
python scrapers/01_extract_auction_search.py
python ml_pipeline/scripts/train_model_minimal.py
python utils/test_modules.py
```

---

## Data Persistence

### Understanding Volume Mounts

By default, any data created inside a Docker container is lost when the container stops. To persist data, you need to mount volumes.

### Mounting the Data Directory

The most common approach is to mount the local `data` directory:

```bash
docker run -it --rm -v $(pwd)/data:/app/data maxsold bash
```

This ensures:
- Scraped data is saved to your local machine
- Trained models persist after the container stops
- Data can be reused across multiple container runs

### Mounting Specific Subdirectories

You can mount specific subdirectories for more granular control:

```bash
docker run -it --rm \
  -v $(pwd)/data/raw:/app/data/raw \
  -v $(pwd)/data/models:/app/data/models \
  maxsold bash
```

### Mounting Source Code for Development

To modify source code and test changes immediately:

```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/scrapers:/app/scrapers \
  -v $(pwd)/ml_pipeline:/app/ml_pipeline \
  -v $(pwd)/feature_engineering:/app/feature_engineering \
  -v $(pwd)/utils:/app/utils \
  maxsold bash
```

---

## Troubleshooting

### Issue: "docker: command not found"

**Solution:** Docker is not installed or not in your PATH.
- Install Docker Desktop from https://www.docker.com/products/docker-desktop
- Restart your terminal after installation
- Verify with: `docker --version`

### Issue: "Cannot connect to the Docker daemon"

**Solution:** Docker daemon is not running.
- **Windows/Mac:** Start Docker Desktop application
- **Linux:** Run `sudo systemctl start docker`
- Verify with: `docker ps`

### Issue: "permission denied while trying to connect to the Docker daemon socket"

**Solution (Linux):** Add your user to the docker group:
```bash
sudo usermod -aG docker $USER
newgrp docker
```

### Issue: Build fails with "no space left on device"

**Solution:** Clean up Docker resources:
```bash
docker system prune -a
docker volume prune
```

### Issue: Volume mount not working on Windows

**Solution:** 
1. Ensure the drive is shared in Docker Desktop settings
2. Use full paths instead of relative paths
3. Check Docker Desktop → Settings → Resources → File Sharing

### Issue: "Module not found" errors

**Solution:** Rebuild the image to ensure all dependencies are installed:
```bash
docker build --no-cache -t maxsold:latest .
```

### Issue: Kaggle credentials not working

**Solution:** Pass credentials as environment variables:
```bash
docker run -it --rm \
  -e KAGGLE_USERNAME=your_username \
  -e KAGGLE_KEY=your_api_key \
  -v $(pwd)/data:/app/data \
  maxsold bash
```

Or mount your Kaggle configuration:
```bash
docker run -it --rm \
  -v ~/.kaggle:/root/.kaggle \
  -v $(pwd)/data:/app/data \
  maxsold bash
```

---

## Advanced Usage

### Running as a Background Service

To run a long-running task in the background:

```bash
docker run -d \
  --name maxsold-scraper \
  -v $(pwd)/data:/app/data \
  maxsold \
  python scrapers/monthly_scraping_pipeline.py
```

Monitor logs:
```bash
docker logs -f maxsold-scraper
```

Stop the container:
```bash
docker stop maxsold-scraper
docker rm maxsold-scraper
```

### Using Docker Compose (Optional)

Create a `docker-compose.yml` file in the project root:

```yaml
version: '3.8'

services:
  maxsold:
    build: .
    image: maxsold:latest
    volumes:
      - ./data:/app/data
    environment:
      - KAGGLE_USERNAME=${KAGGLE_USERNAME}
      - KAGGLE_KEY=${KAGGLE_KEY}
    command: bash
    stdin_open: true
    tty: true
```

Run with:
```bash
docker-compose up
```

### Customizing the Image

You can create your own Dockerfile based on the provided one:

```dockerfile
FROM maxsold:latest

# Add custom packages
RUN pip install jupyter notebook

# Set custom working directory
WORKDIR /app/scrapers

CMD ["bash"]
```

Build it:
```bash
docker build -f Dockerfile.custom -t maxsold:custom .
```

### Resource Limits

Limit CPU and memory usage:

```bash
docker run -it --rm \
  --cpus="2.0" \
  --memory="4g" \
  -v $(pwd)/data:/app/data \
  maxsold bash
```

---

## Environment Variables

The following environment variables can be set when running the container:

| Variable | Description | Example |
|----------|-------------|---------|
| `KAGGLE_USERNAME` | Your Kaggle username | `-e KAGGLE_USERNAME=myusername` |
| `KAGGLE_KEY` | Your Kaggle API key | `-e KAGGLE_KEY=abc123...` |
| `PYTHONUNBUFFERED` | Force Python to run in unbuffered mode | Already set to `1` |

---

## Quick Reference Commands

### Build
```bash
docker build -t maxsold:latest .
```

### Run Interactive Shell
```bash
docker run -it --rm -v $(pwd)/data:/app/data maxsold bash
```

### Run Specific Script
```bash
docker run -it --rm -v $(pwd)/data:/app/data maxsold python scrapers/<script>.py
```

### List Running Containers
```bash
docker ps
```

### Stop a Container
```bash
docker stop <container_id>
```

### Remove All Stopped Containers
```bash
docker container prune
```

### Remove the Image
```bash
docker rmi maxsold:latest
```

### View Container Logs
```bash
docker logs <container_id>
```

### Copy Files from Container
```bash
docker cp <container_id>:/app/data/output.txt ./output.txt
```

---

## Best Practices

1. **Always use volume mounts** for data persistence
2. **Use `--rm` flag** for temporary containers to save disk space
3. **Keep the image updated** by rebuilding when dependencies change
4. **Use specific tags** instead of `latest` for production use
5. **Monitor resource usage** with `docker stats`
6. **Clean up regularly** with `docker system prune`

---

## Getting Help

If you encounter issues not covered in this guide:

1. Check the main [README.md](README.md) for project-specific information
2. Review the [ML Pipeline README](ml_pipeline/README.md) for ML-specific guidance
3. Consult Docker documentation: https://docs.docker.com/
4. Check Docker Desktop troubleshooting: https://docs.docker.com/desktop/troubleshoot/overview/
5. Open an issue on GitHub: https://github.com/Jonathan-Pearce/maxsold/issues

---

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Python Docker Best Practices](https://pythonspeed.com/docker/)
- [MaxSold Project README](README.md)
- [ML Pipeline Documentation](ml_pipeline/README.md)

---

**Last Updated:** January 18, 2026  
**Docker Version Tested:** 24.0+  
**Python Version:** 3.12
