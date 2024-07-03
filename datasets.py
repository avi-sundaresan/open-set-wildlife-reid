import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, image_paths, image_labels, transform):
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = Image.open(image_filepath)
        label = self.image_labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        patch_tokens, class_token = self.embeddings[idx]
        label = self.labels[idx]
        return patch_tokens, class_token, label
    
def compute_full_embeddings(dataloader, feature_model, device):
    embeddings = []
    labels = []
    feature_model.eval()
    with torch.no_grad():
        for images, lbls in tqdm(dataloader):
            images = images.to(device)
            features = feature_model(images)
            ((patch_tokens, class_token),) = features
            embeddings.append((patch_tokens.clone(), class_token.clone()))
            labels.append(lbls.clone())
    return embeddings, labels