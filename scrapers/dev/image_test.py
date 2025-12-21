import requests
from PIL import Image
from io import BytesIO

# Download the image
url = "https://i.maxsold.com/_/lg:1x/plain/https://cdn-d12srav5gxm0re.maxsold.com/auctionimages/103293/1762628131/wpampered_chef_heritage_stoneware_rectangular_baker-69-1.jpeg"
response = requests.get(url)
response.raise_for_status()

# Load the image
img = Image.open(BytesIO(response.content))

# Save the original image
img.save('original_image.jpg')

# Get current dimensions
width, height = img.size

# Calculate new dimensions (scale down to 256px on largest side)
if width > height:
    new_width = 256
    new_height = int((256 / width) * height)
else:
    new_height = 256
    new_width = int((256 / height) * width)

# Resize the image with high-quality resampling
img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

# Save as WebP format (more efficient than JPEG) with compression
img_resized.save('scaled_image.webp', 'WEBP', quality=85, method=6)

# Alternative: save as JPEG with compression
img_resized.save('scaled_image.jpg', 'JPEG', quality=85, optimize=True)

print(f"Original size: {width}x{height}")
print(f"Scaled size: {new_width}x{new_height}")
print(f"Original file saved as: original_image.jpg")
print(f"Scaled file saved as: scaled_image.webp and scaled_image.jpg")