# Docker Setup Guide for MaxSold Data Project

This guide provides detailed instructions for setting up and running the MaxSold Data Project using Docker.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Building the Docker Image](#building-the-docker-image)
- [Running the Container](#running-the-container)
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

### Basic Container Run

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

### With Data Persistence (Recommended)

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

## Common Use Cases

### 1. Running Web Scrapers

#### Extract Auction Search Data
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

#### Minimal Model Training (Fastest - 30-60 seconds)
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

```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  maxsold \
  python ml_pipeline/utils/verify_model_setup.py
```

### 4. Testing Modules

```bash
docker run -it --rm maxsold python utils/test_modules.py
```

### 5. Interactive Development

For exploratory work or debugging:

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

### Issue: Container runs but scripts fail with "No such file or directory"

**Solution:** Ensure you're running commands from the project root directory where the Dockerfile is located.

### Issue: Selenium/Chrome issues in container

**Solution:** The Dockerfile includes Chrome installation. If you still encounter issues:
```bash
docker run -it --rm maxsold bash
# Inside container:
google-chrome --version  # Verify Chrome is installed
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
