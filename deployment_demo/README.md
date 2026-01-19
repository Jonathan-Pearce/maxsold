# Model Deployment Demo

This demo showcases the deployment of the MobileNetV2 image feature extraction model used in the MaxSold machine learning pipeline.

## Overview

This deployment demo consists of:

1. **Backend (FastAPI)**: A REST API that extracts image features from MaxSold items
2. **Frontend (GitHub Pages)**: A simple web UI to interact with the API
3. **Docker Support**: Containerized deployment for the backend

## Architecture

```
┌─────────────────┐      HTTP Request      ┌──────────────────┐
│                 │  ───────────────────>   │                  │
│  GitHub Pages   │                         │  FastAPI Backend │
│  (Frontend)     │  <───────────────────   │                  │
│                 │      JSON Response      └──────────────────┘
└─────────────────┘                                  │
                                                     │
                                                     ▼
                                         ┌──────────────────────┐
                                         │  MobileNetV2 Model   │
                                         │  (Feature Extractor) │
                                         └──────────────────────┘
                                                     │
                                                     ▼
                                         ┌──────────────────────┐
                                         │   MaxSold API        │
                                         │   (Image Source)     │
                                         └──────────────────────┘
```

## Features

### Backend API

- **Endpoint**: `POST /extract-features`
- **Input**: MaxSold item URL (e.g., `https://maxsold.com/listing/7433850/`)
- **Processing**:
  1. Extracts item ID from URL
  2. Fetches first image URL from MaxSold API
  3. Downloads and preprocesses image (resize to 256px max, center crop to 224x224)
  4. Extracts 1280-dimensional feature vector using MobileNetV2
  5. Returns features + metadata
- **CORS**: Enabled for cross-origin requests from GitHub Pages

### Frontend UI

- Simple, responsive web interface
- Input field for MaxSold item URLs
- Displays:
  - Item ID
  - Model name
  - Feature dimensions
  - Image preview
  - First 100 feature values

## API Contract

### Request

```json
POST /extract-features
Content-Type: application/json

{
  "item_url": "https://maxsold.com/listing/7433850/"
}
```

### Response

```json
{
  "item_id": "7433850",
  "image_url": "https://...",
  "features": [0.123, 0.456, ...],  // 1280 values
  "feature_dimension": 1280,
  "model_name": "MobileNetV2"
}
```

### Error Response

```json
{
  "detail": "Error message"
}
```

## Local Development

### Prerequisites

- Python 3.12+
- pip

### Backend Setup

1. Navigate to the backend directory:
```bash
cd deployment_demo/backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the FastAPI server:
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Optional - Configure CORS for production:**
```bash
# Set allowed origins via environment variable
export ALLOWED_ORIGINS="https://yourusername.github.io,https://example.com"
python main.py
```

The API will be available at `http://localhost:8000`

4. View API documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Frontend Setup

1. Open `deployment_demo/frontend/index.html` in a web browser
2. Ensure the API URL is set to `http://localhost:8000` (default)
3. Enter a MaxSold item URL and click "Extract Features"

Alternatively, serve it with a local web server:
```bash
cd deployment_demo/frontend
python -m http.server 8080
```

Then open `http://localhost:8080` in your browser.

## Docker Deployment

### Build the Docker image

```bash
cd deployment_demo/backend
docker build -t maxsold-feature-api .
```

### Run the container

```bash
docker run -p 8000:8000 maxsold-feature-api
```

The API will be available at `http://localhost:8000`

### Using Docker Compose (recommended)

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  api:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
```

Then run:
```bash
docker-compose up -d
```

## Deployment Options

### Backend Hosting

The FastAPI backend can be deployed to:

1. **Cloud Run (Google Cloud)**
   - Serverless, auto-scaling
   - Pay per use
   - Easy Docker deployment

2. **Heroku**
   - Simple git-based deployment
   - Free tier available (with limitations)

3. **AWS Lambda + API Gateway**
   - Serverless
   - Requires additional configuration for PyTorch

4. **DigitalOcean App Platform**
   - Docker support
   - Affordable pricing

5. **Self-hosted VPS**
   - Full control
   - Requires server management

### Frontend Hosting

The frontend is designed to be deployed on **GitHub Pages**:

1. Create a `gh-pages` branch or use the `docs/` folder
2. Copy the `frontend/index.html` file
3. Enable GitHub Pages in repository settings
4. Update the API URL in the frontend to point to your deployed backend

## Testing

### Test the API directly

Using curl:
```bash
curl -X POST http://localhost:8000/extract-features \
  -H "Content-Type: application/json" \
  -d '{"item_url": "https://maxsold.com/listing/7433850/"}'
```

Using Python:
```python
import requests

response = requests.post(
    "http://localhost:8000/extract-features",
    json={"item_url": "https://maxsold.com/listing/7433850/"}
)

data = response.json()
print(f"Item ID: {data['item_id']}")
print(f"Feature dimension: {data['feature_dimension']}")
print(f"First 5 features: {data['features'][:5]}")
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Model Details

- **Model**: MobileNetV2 (pre-trained on ImageNet)
- **Architecture**: Convolutional neural network optimized for mobile devices
- **Feature Dimension**: 1280
- **Input Size**: 224×224 pixels (RGB)
- **Preprocessing**:
  - Resize to 256px (max dimension)
  - Center crop to 224×224
  - Normalize with ImageNet mean/std
- **Output**: Flattened feature vector from the last convolutional layer

## Performance Considerations

### Backend

- **Model Loading**: Model is loaded once at startup (~500MB memory)
- **Inference Time**: ~0.5-2 seconds per image (depending on hardware)
- **CPU vs GPU**: 
  - CPU mode works fine for demo purposes
  - GPU can provide 5-10x speedup for production
- **Scaling**: Use multiple worker processes or container replicas for high load

### Frontend

- **CORS**: Ensure backend CORS settings match your frontend origin
- **Caching**: Browser caches images automatically
- **Error Handling**: Frontend includes comprehensive error messages

## Security Considerations

### Current Implementation (Demo)

⚠️ **This is a demo/prototype** - not production-ready

Current limitations:
- No authentication/authorization
- **Open CORS policy (`allow_origins=["*"]`)** - Allows requests from any origin
- No rate limiting
- No input validation beyond basic checks
- No HTTPS enforcement

### Production Recommendations

**CRITICAL:** Before deploying to production, you MUST implement:

1. **CORS Restrictions**: 
   ```bash
   # Method 1: Environment variable (recommended)
   export ALLOWED_ORIGINS="https://yourusername.github.io"
   
   # Method 2: Update code directly in main.py
   # Replace the allow_origins line with:
   allow_origins=["https://yourusername.github.io"]
   ```

2. **Authentication**: API keys, OAuth, or JWT tokens
3. **Rate Limiting**: Prevent abuse (e.g., using `slowapi`)
   ```python
   from slowapi import Limiter
   limiter = Limiter(key_func=get_remote_address)
   @app.post("/extract-features")
   @limiter.limit("10/minute")
   async def extract_features_endpoint(...):
   ```

4. **Input Validation**: Strict URL validation, sanitization
5. **HTTPS**: Always use TLS/SSL in production
6. **Monitoring**: Log requests, track errors, set up alerts
7. **Resource Limits**: Timeout limits, max file sizes
8. **Secrets Management**: Use environment variables, never commit secrets

### Security Checklist for Production

- [ ] Restrict CORS to specific frontend origin(s) via `ALLOWED_ORIGINS` env var
- [ ] Add authentication (API keys minimum)
- [ ] Implement rate limiting
- [ ] Enable HTTPS/TLS
- [ ] Set up logging and monitoring
- [ ] Add input sanitization (✓ basic validation included)
- [ ] Configure timeouts and resource limits
- [ ] Use environment variables for configuration
- [ ] Add request validation with Pydantic (✓ included)
- [ ] Set up WAF (Web Application Firewall) if using cloud services

### Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `ALLOWED_ORIGINS` | Comma-separated list of allowed CORS origins | `*` (demo) | `https://user.github.io,https://example.com` |

## Future Enhancements

Potential improvements for the full application:

### Backend
- [ ] Batch processing support (multiple items at once)
- [ ] Feature caching with Redis
- [ ] Support for multiple models (ResNet, EfficientNet)
- [ ] Asynchronous processing with job queues
- [ ] Model versioning and A/B testing
- [ ] Metrics and monitoring (Prometheus, Grafana)
- [ ] Rate limiting and authentication

### Frontend
- [ ] Batch upload support
- [ ] Feature visualization (PCA, t-SNE)
- [ ] Compare multiple items
- [ ] Download results as CSV/JSON
- [ ] Integration with other MaxSold data
- [ ] Dark mode

### Infrastructure
- [ ] CI/CD pipeline
- [ ] Automated testing
- [ ] Load balancing
- [ ] Auto-scaling configuration
- [ ] Database for feature storage
- [ ] CDN for static assets

## Troubleshooting

### Backend Issues

**Error: Model not loading**
- Ensure PyTorch and torchvision are installed correctly
- Check Python version (3.12+ required)
- Verify sufficient memory (~1GB minimum)

**Error: CORS issues**
- Check CORS configuration in `main.py`
- Ensure frontend URL matches allowed origins
- Use browser dev tools to inspect CORS headers

**Error: MaxSold API timeout**
- MaxSold API may be slow or down
- Increase timeout values in `main.py`
- Try different item IDs

### Frontend Issues

**Error: Cannot connect to API**
- Verify backend is running (`curl http://localhost:8000/health`)
- Check API URL in frontend
- Look for CORS errors in browser console

**Error: Invalid item URL**
- Use complete URL: `https://maxsold.com/listing/{id}/`
- Item must exist on MaxSold
- Try with example: `https://maxsold.com/listing/7433850/`

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [PyTorch Vision Models](https://pytorch.org/vision/stable/models.html)
- [Docker Documentation](https://docs.docker.com/)
- [GitHub Pages Guide](https://pages.github.com/)

## License

Same as the main repository.

## Contact

For questions or issues, please open a GitHub issue.
