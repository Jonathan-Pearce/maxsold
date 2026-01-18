#!/bin/bash
# MaxSold Docker Helper Script
# This script provides convenient commands for working with the MaxSold Docker container

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

IMAGE_NAME="maxsold:latest"
CONTAINER_NAME="maxsold-container"

print_usage() {
    echo -e "${BLUE}MaxSold Docker Helper Script${NC}"
    echo ""
    echo "Usage: ./docker-run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  build                  - Build the Docker image"
    echo "  shell                  - Start an interactive shell in the container"
    echo "  scraper [script]       - Run a specific scraper script"
    echo "  ml-minimal             - Train minimal ML model (fastest)"
    echo "  ml-fast                - Train ML model with visualizations"
    echo "  ml-quick               - Train ML model with full dataset"
    echo "  ml-complete            - Run complete ML pipeline"
    echo "  verify                 - Verify model setup"
    echo "  test                   - Run test modules"
    echo "  clean                  - Remove containers and clean up"
    echo "  logs                   - View container logs"
    echo ""
    echo "Examples:"
    echo "  ./docker-run.sh build"
    echo "  ./docker-run.sh shell"
    echo "  ./docker-run.sh scraper monthly_scraping_pipeline.py"
    echo "  ./docker-run.sh ml-minimal"
}

build_image() {
    echo -e "${GREEN}Building Docker image...${NC}"
    docker build -t ${IMAGE_NAME} .
    echo -e "${GREEN}✓ Image built successfully${NC}"
}

run_shell() {
    echo -e "${GREEN}Starting interactive shell...${NC}"
    docker run -it --rm \
        -v "$(pwd)/data:/app/data" \
        --name ${CONTAINER_NAME} \
        ${IMAGE_NAME} bash
}

run_scraper() {
    local script=$1
    if [ -z "$script" ]; then
        echo -e "${YELLOW}Please specify a scraper script${NC}"
        echo "Example: ./docker-run.sh scraper 01_extract_auction_search.py"
        exit 1
    fi
    echo -e "${GREEN}Running scraper: ${script}${NC}"
    docker run -it --rm \
        -v "$(pwd)/data:/app/data" \
        --name ${CONTAINER_NAME} \
        ${IMAGE_NAME} python "scrapers/${script}"
}

run_ml_minimal() {
    echo -e "${GREEN}Training minimal ML model (30-60 seconds)...${NC}"
    docker run -it --rm \
        -v "$(pwd)/data:/app/data" \
        --name ${CONTAINER_NAME} \
        ${IMAGE_NAME} python ml_pipeline/scripts/train_model_minimal.py
}

run_ml_fast() {
    echo -e "${GREEN}Training ML model with visualizations (1-2 minutes)...${NC}"
    docker run -it --rm \
        -v "$(pwd)/data:/app/data" \
        --name ${CONTAINER_NAME} \
        ${IMAGE_NAME} python ml_pipeline/scripts/model_pipeline_fast.py
}

run_ml_quick() {
    echo -e "${GREEN}Training ML model with full dataset (2-3 minutes)...${NC}"
    docker run -it --rm \
        -v "$(pwd)/data:/app/data" \
        --name ${CONTAINER_NAME} \
        ${IMAGE_NAME} python ml_pipeline/scripts/model_pipeline_quick.py
}

run_ml_complete() {
    echo -e "${GREEN}Running complete ML pipeline (5-10 minutes)...${NC}"
    docker run -it --rm \
        -v "$(pwd)/data:/app/data" \
        --name ${CONTAINER_NAME} \
        ${IMAGE_NAME} python ml_pipeline/scripts/model_pipeline.py
}

verify_setup() {
    echo -e "${GREEN}Verifying model setup...${NC}"
    docker run -it --rm \
        -v "$(pwd)/data:/app/data" \
        --name ${CONTAINER_NAME} \
        ${IMAGE_NAME} python ml_pipeline/utils/verify_model_setup.py
}

run_test() {
    echo -e "${GREEN}Running test modules...${NC}"
    docker run -it --rm \
        --name ${CONTAINER_NAME} \
        ${IMAGE_NAME} python utils/test_modules.py
}

clean_up() {
    echo -e "${GREEN}Cleaning up containers...${NC}"
    # Filter for containers with names containing "maxsold"
    # This matches maxsold-container, maxsold-scraper, maxsold-ml, etc.
    local containers=$(docker ps -a -q --filter "name=maxsold" 2>/dev/null)
    if [ -n "$containers" ]; then
        echo "$containers" | xargs docker rm -f
        echo -e "${GREEN}✓ Cleanup complete${NC}"
    else
        echo -e "${YELLOW}No maxsold containers to clean up${NC}"
    fi
}

view_logs() {
    echo -e "${GREEN}Viewing logs...${NC}"
    docker logs ${CONTAINER_NAME}
}

# Main script logic
case "${1}" in
    build)
        build_image
        ;;
    shell)
        run_shell
        ;;
    scraper)
        run_scraper "${2}"
        ;;
    ml-minimal)
        run_ml_minimal
        ;;
    ml-fast)
        run_ml_fast
        ;;
    ml-quick)
        run_ml_quick
        ;;
    ml-complete)
        run_ml_complete
        ;;
    verify)
        verify_setup
        ;;
    test)
        run_test
        ;;
    clean)
        clean_up
        ;;
    logs)
        view_logs
        ;;
    *)
        print_usage
        exit 1
        ;;
esac
