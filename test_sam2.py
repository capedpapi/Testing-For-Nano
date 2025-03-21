import sys
sys.path.append(r"C:\Users\mailt\OneDrive\Desktop\baitech\sam2Clone\sam2")
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the SAM 2 model
checkpoint = r"C:\Users\mailt\OneDrive\Desktop\baitech\sam2Clone\sam2\checkpointssam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"   # Path to the configuration file
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Load your sample image (change this path to your image)
image_path = 'Pigfarmsample.png'  # Update this to your sample image file path
image = cv2.imread(image_path)

# Preprocess the image
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run segmentation
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image_rgb)
    masks, _, _ = predictor.predict()  # The model automatically detects objects in the image

# Show the result
plt.imshow(masks[0], cmap='jet')  # Display the first mask (adjust if there are multiple)
plt.show()
