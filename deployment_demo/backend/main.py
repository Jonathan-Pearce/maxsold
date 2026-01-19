"""
FastAPI Backend for MaxSold Image Feature Extraction Demo

This API provides a simple endpoint to extract image features from MaxSold items
using the pre-trained MobileNetV2 model.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from contextlib import asynccontextmanager
import requests
import torch
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
import numpy as np
from typing import List

# Global model instance
model = None
transform = None
device = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for model loading"""
    global model, transform, device
    
    print("Loading MobileNetV2 model...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pre-trained MobileNetV2
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    base_model = models.mobilenet_v2(weights=weights)
    
    # Remove classifier, keep only features
    model = base_model.features
    model.eval()
    model.to(device)
    
    # Define image preprocessing (same as in image_features.py)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    print(f"✓ Model loaded on {device}")
    yield
    # Cleanup if needed
    print("Shutting down...")

app = FastAPI(
    title="MaxSold Image Feature Extraction API",
    description="Extract image features from MaxSold items using MobileNetV2",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for GitHub Pages
# WARNING: This configuration allows all origins and is NOT production-ready!
# For production, replace ["*"] with specific allowed origins, e.g.:
# allow_origins=["https://yourusername.github.io"]
# See: https://fastapi.tiangolo.com/tutorial/cors/
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ DEMO ONLY - Restrict in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

class ItemRequest(BaseModel):
    """Request model for item URL"""
    item_url: str
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "item_url": "https://maxsold.com/listing/7433850/"
            }
        }
    )

class FeatureResponse(BaseModel):
    """Response model with extracted features"""
    item_id: str
    image_url: str
    features: List[float]
    feature_dimension: int
    model_name: str

def extract_item_id_from_url(url: str) -> str:
    """Extract item ID from MaxSold URL"""
    # URL format: https://maxsold.com/listing/{item_id}/
    parts = url.rstrip('/').split('/')
    if 'listing' in parts:
        idx = parts.index('listing')
        if idx + 1 < len(parts):
            return parts[idx + 1]
    raise ValueError("Could not extract item ID from URL")

def fetch_first_image_url(item_id: str) -> str:
    """Fetch the first image URL for an item from MaxSold API"""
    api_url = f"https://api.maxsold.com/listings/am/{item_id}/enriched"
    
    try:
        response = requests.get(api_url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Try to find images in the response
        # First try images array directly
        if 'images' in data and isinstance(data['images'], list) and len(data['images']) > 0:
            return data['images'][0]
        
        # Try auction item API as fallback
        auction_id = data.get('amAuctionId')
        if auction_id:
            auction_api_url = f"https://maxsold.maxsold.com/msapi/auctions/items?auctionid={auction_id}&itemid={item_id}"
            response = requests.get(auction_api_url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            auction_data = response.json()
            
            # Navigate to items array
            if 'auction' in auction_data and 'items' in auction_data['auction']:
                items = auction_data['auction']['items']
                if len(items) > 0 and 'images' in items[0]:
                    images = items[0]['images']
                    if len(images) > 0:
                        return images[0]
        
        raise ValueError("No images found for this item")
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
        raise HTTPException(status_code=500, detail=f"Error fetching item data: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching image: {str(e)}")

def download_and_preprocess_image(image_url: str) -> torch.Tensor:
    """Download image from URL and preprocess it"""
    try:
        # Download image
        response = requests.get(image_url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        # Open and convert to RGB
        image = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        return image_tensor
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

def extract_features(image_tensor: torch.Tensor) -> np.ndarray:
    """Extract features using MobileNetV2"""
    with torch.no_grad():
        features = model(image_tensor)
        
        # Global average pooling
        if len(features.shape) == 4:  # [batch, channels, height, width]
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        
        # Flatten
        features = features.view(features.size(0), -1)
    
    return features.cpu().numpy().flatten()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MaxSold Image Feature Extraction API",
        "version": "1.0.0",
        "endpoints": {
            "POST /extract-features": "Extract features from a MaxSold item"
        },
        "example_usage": {
            "url": "/extract-features",
            "method": "POST",
            "body": {
                "item_url": "https://maxsold.com/listing/7433850/"
            }
        }
    }

@app.post("/extract-features", response_model=FeatureResponse)
async def extract_features_endpoint(request: ItemRequest):
    """
    Extract image features from a MaxSold item.
    
    The endpoint:
    1. Extracts the item ID from the URL
    2. Fetches the first image URL from MaxSold API
    3. Downloads and preprocesses the image (resize to 256px max, center crop to 224x224)
    4. Extracts features using MobileNetV2
    5. Returns the feature vector
    """
    try:
        # Extract item ID from URL
        item_id = extract_item_id_from_url(request.item_url)
        
        # Fetch first image URL
        image_url = fetch_first_image_url(item_id)
        
        # Download and preprocess image
        image_tensor = download_and_preprocess_image(image_url)
        
        # Extract features
        features = extract_features(image_tensor)
        
        return FeatureResponse(
            item_id=item_id,
            image_url=image_url,
            features=features.tolist(),
            feature_dimension=len(features),
            model_name="MobileNetV2"
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "not initialized"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
