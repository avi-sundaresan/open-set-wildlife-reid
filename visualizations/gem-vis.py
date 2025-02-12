import torch
import timm
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from functools import partial

from datasets.datasets import prepare_datasets, split_dataset
from configs.config import get_dataset_root
from models import ModelWithIntermediateLayersMD, GeMPooler  # Import GeM Pooling Layer

# 📌 Load the first image from SeaTurtleIDHeads dataset
def load_first_image(dataset="SeaTurtleIDHeads", image_size=384):
    root = get_dataset_root(dataset)
    df, _, _ = split_dataset(prepare_datasets(root, dataset))
    
    image_path = os.path.join(root, df.iloc[0]['path'])  # First image
    image = Image.open(image_path).convert("RGB")
    
    # Transform the image
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    return transform(image).unsqueeze(0), image_path  # Return tensor and path

# 📌 Load MegaDescriptor model
def load_megadescriptor(device="cuda:0"):
    model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.float)
    return ModelWithIntermediateLayersMD(model, autocast_ctx).to(device)

# 📌 Visualize GeM activations for different p values using GeMPooler
def visualize_gem_p(query_image_path, patch_tokens, p_values=[1, 3, 10], grid_size=12):
    fig, axes = plt.subplots(1, len(p_values), figsize=(12, 4))

    for i, p in enumerate(p_values):
        # Initialize GeM Pooler with p
        gem_pooler = GeMPooler(p=p)
        heatmap = gem_pooler(patch_tokens).view(grid_size, grid_size).cpu().numpy()

        # Normalize for visualization
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        # Resize heatmap to match original image
        heatmap_resized = cv2.resize(heatmap, (384, 384))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

        # Overlay on image
        image = cv2.imread(query_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)

        axes[i].imshow(overlay)
        axes[i].axis("off")
        axes[i].set_title(f"p = {p}")

    plt.tight_layout()
    plt.show()

# 📌 Main execution
device = "cuda:0" if torch.cuda.is_available() else "cpu"
image_tensor, image_path = load_first_image()
model = load_megadescriptor(device)

with torch.no_grad():
    patch_tokens, _ = model(image_tensor.to(device))

visualize_gem_p(image_path, patch_tokens)
