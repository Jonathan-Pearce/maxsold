# Docker Quick Reference Card

## One-Time Setup

```bash
# Clone repository
git clone https://github.com/Jonathan-Pearce/maxsold.git
cd maxsold

# Build image (Linux/Mac)
chmod +x docker-run.sh
./docker-run.sh build

# Build image (Windows)
docker-run.bat build
```

## Daily Usage

### Interactive Shell

```bash
# Linux/Mac
./docker-run.sh shell

# Windows
docker-run.bat shell
```

### Run Scrapers

```bash
# Linux/Mac
./docker-run.sh scraper monthly_scraping_pipeline.py

# Windows
docker-run.bat scraper monthly_scraping_pipeline.py
```

### Train ML Models

```bash
# Linux/Mac
./docker-run.sh ml-minimal     # 30-60 seconds
./docker-run.sh ml-fast        # 1-2 minutes
./docker-run.sh ml-quick       # 2-3 minutes
./docker-run.sh ml-complete    # 5-10 minutes

# Windows
docker-run.bat ml-minimal      REM 30-60 seconds
docker-run.bat ml-fast         REM 1-2 minutes
docker-run.bat ml-quick        REM 2-3 minutes
docker-run.bat ml-complete     REM 5-10 minutes
```

## All Available Commands

| Command | Description |
|---------|-------------|
| `build` | Build Docker image |
| `shell` | Interactive bash shell |
| `scraper <script>` | Run scraper script |
| `ml-minimal` | Train minimal ML model |
| `ml-fast` | Train with visualizations |
| `ml-quick` | Train with full dataset |
| `ml-complete` | Complete ML pipeline |
| `verify` | Verify model setup |
| `test` | Run test modules |
| `clean` | Remove all containers |
| `logs` | View container logs |

## Data Persistence

All commands automatically mount `./data` directory to `/app/data` in the container, ensuring your data persists after the container exits.

## Troubleshooting

### Docker not found
- Install Docker Desktop from https://www.docker.com/products/docker-desktop
- Restart terminal after installation

### Permission denied (Linux)
```bash
sudo usermod -aG docker $USER
newgrp docker
```

### Build fails
```bash
# Clean up and rebuild
docker system prune -a
./docker-run.sh build  # or docker-run.bat build
```

## Full Documentation

See [DOCKER_SETUP.md](DOCKER_SETUP.md) for complete documentation including:
- Detailed setup instructions
- Manual Docker commands
- Docker Compose usage
- Advanced configurations
- Common issues and solutions

## Quick Commands Cheat Sheet

```bash
# Build
./docker-run.sh build

# Shell
./docker-run.sh shell

# Scraper
./docker-run.sh scraper <script_name>.py

# ML Training
./docker-run.sh ml-minimal

# Verify
./docker-run.sh verify

# Clean Up
./docker-run.sh clean
```
