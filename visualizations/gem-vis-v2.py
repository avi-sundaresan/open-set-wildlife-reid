import torch
import timm
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from functools import partial

from data_utils.datasets import prepare_datasets, split_dataset
from configs.config import get_dataset_root
from models import ModelWithIntermediateLayersMD, GeMPooler  # Import GeM Pooling Layer

# Load the first image from SeaTurtleIDHeads dataset
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
    
    return transform(image).unsqueeze(0), image_path  

def load_megadescriptor(device="cuda:0"):
    model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.float)
    return ModelWithIntermediateLayersMD(model, autocast_ctx).to(device)

# Visualize GeM activations for different p values using the spatial GeM
def visualize_gem_p(query_image_path, patch_tokens, device, p_values=[1, 3, 10], grid_size=12, save_dir="visualizations/gem"):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, len(p_values), figsize=(12, 4))

    # Apply spatial GeM pooling (across feature channels, not patches)
    def spatial_gem(x, p):
        return (x.clamp(min=1e-6).pow(p).mean(dim=-1, keepdim=False)).pow(1.0 / p)  # (144,)

    # Compute the min and max across ALL heatmaps, not per heatmap
    global_min = min([spatial_gem(patch_tokens, p).min().item() for p in p_values])
    global_max = max([spatial_gem(patch_tokens, p).max().item() for p in p_values])

    print(f"Global min: {global_min}")
    print(f"Global max: {global_max}")

    for i, p in enumerate(p_values):
        # Apply GeM pooling spatially (over feature channels)
        pooled = spatial_gem(patch_tokens, p)  # (144,)
        heatmap = pooled.view(grid_size, grid_size).detach().cpu().numpy()  # (12, 12)

        print(f"Min for p = {p}: {pooled.min().item()}")
        print(f"Max for p = {p}: {pooled.max().item()}")
        # the *spread* between the two goes up significantly from p = 1 --> p =
        # 10... so what gives?? 

        # Normalize activations across the image (0 to 1 scale)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        # heatmap = heatmap / heatmap.max()

        # Normalize using global min/max instead of per heatmap
        # heatmap = (heatmap - global_min) / (global_max - global_min)

        # Apply log scaling
        # heatmap = np.log1p(heatmap) 
        heatmap_resized = cv2.resize(heatmap, (384, 384), interpolation=cv2.INTER_NEAREST)

        # Red intensity based on activation
        red_colormap = np.zeros((heatmap_resized.shape[0], heatmap_resized.shape[1], 3), dtype=np.uint8)
        red_colormap[:, :, 0] = np.uint8(255 * heatmap_resized) 
        red_colormap[:, :, 1] = 0  
        red_colormap[:, :, 2] = 0 

        # get OG image
        image = cv2.imread(query_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # make sure heatmap matches image size just in case
        if image.shape[:2] != red_colormap.shape[:2]:
            red_colormap = cv2.resize(red_colormap, (image.shape[1], image.shape[0]))

        # overlay the heatmap onto the original image with transparency
        overlay = cv2.addWeighted(image, 0.6, red_colormap, 0.4, 0)
        save_path = os.path.join(save_dir, f"gem_p_{p}.png")
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        axes[i].imshow(overlay)
        axes[i].axis("off")
        axes[i].set_title(f"p = {p}")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gem_p_comparison.png"), bbox_inches="tight")
    print(f"Saved GeM visualizations in: {save_dir}")
    plt.show()

EMBEDDINGS_PATH = "visualizations/saved_patch_tokens.pt"
device = "cpu"
image_tensor, image_path = load_first_image()

if not os.path.exists(EMBEDDINGS_PATH):
    device = "cuda:1" 
    model = load_megadescriptor(device)
    print("Generating embeddings and saving to disk...")
    with torch.no_grad():
        image_tensor = image_tensor.to(device) 
        features = model(image_tensor)
        ((patch_tokens, class_token),) = features
        patch_tokens = patch_tokens.squeeze(0)  

    torch.save(patch_tokens.cpu(), EMBEDDINGS_PATH) 
    print(f"Embeddings saved to {EMBEDDINGS_PATH}")
else:
    print(f"Loading cached embeddings from {EMBEDDINGS_PATH}...")
    patch_tokens = torch.load(EMBEDDINGS_PATH, map_location=torch.device("cpu"))

visualize_gem_p(image_path, patch_tokens, device=device)
