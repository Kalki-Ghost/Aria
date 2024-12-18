import torch
from PIL import Image
import requests
from io import BytesIO
from aria.tokenizer import ARIATokenizer
from aria.lora.utils import preprocess_image
from aria.model import ARIAModel

# Load the tokenizer and model
tokenizer = ARIATokenizer.from_pretrained("aria")
model = ARIAModel.from_pretrained("aria")

# Set the model to evaluation mode
model.eval()

# Example input: Multimodal data (text + image)
input_text = "Describe the scene in this image:"
input_image_url = "https://letsenhance.io/static/8f5e523ee6b2479e26ecc91b9c25261e/1015f/MainAfter.jpg"

# Tokenize the text
text_tokens = tokenizer.encode(input_text, return_tensors="pt")

# Load and preprocess the image
try:
    response = requests.get(input_image_url)
    response.raise_for_status()  # Check for HTTP errors
    image = Image.open(BytesIO(response.content)).convert("RGB")
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

# Preprocess the image (ensure preprocess_image is defined correctly)
image_tensor = preprocess_image(image).unsqueeze(0)

# Combine text and image tokens
inputs = {"text": text_tokens, "image": image_tensor}

# Generate output
with torch.no_grad():
    output = model.generate(**inputs)

# Decode the output
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Output:", output_text)
