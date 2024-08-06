import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms
import PIL
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm 
from torch.utils.data import DataLoader


from models import create_linear_input, AttentiveClassifier
from datasets import EmbeddingsDataset

def get_transformation(model):
    print(model)
    if model == 'dinov2' or model == 'dinov2_reg':
        return transforms.Compose([
            transforms.Resize((224, 224), interpolation=PIL.Image.Resampling.BILINEAR, antialias=True),
            transforms.ToTensor()
        ])
    elif model == "megadescriptor":
        return transforms.Compose([
        transforms.Resize((384, 384), interpolation= PIL.Image.Resampling.BILINEAR, antialias=True),
        transforms.ToTensor(),
    ])
    else:
        raise ValueError(f"Unknown model: {model}")

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

def compute_embeddings(dataloaders, feature_model, device):
    embeddings = []
    labels = []
    for loader in dataloaders:
        e, l = compute_full_embeddings(loader, feature_model, device)
        embeddings.append(e)
        labels.append(l)
    return embeddings, labels

def flatten_embeddings(embeddings, labels, pooling_method, use_class, attentive_classifier=None):
    embeddings_f = []

    if pooling_method == 'attentive':
        for embedding in embeddings:
            device = embedding[0].device
            attentive_classifier = attentive_classifier.to(device)
            attended_output = attentive_classifier.get_pooled_output(embedding)
            embeddings_f.append(attended_output.cpu().detach().numpy())
    else:
        for embedding in embeddings:
            linear_input = create_linear_input(embedding, use_avgpool=(pooling_method == 'linear'), use_class=use_class)
            embeddings_f.extend([np.array(tr.cpu()) for tr in linear_input])

    labels_f = [np.array(l.cpu()) for label in labels for l in label]
    embeddings_f = np.vstack(embeddings_f)
    return np.array(embeddings_f), np.array(labels_f)

def combine_train_val(train_embeddings, train_labels, val_embeddings, val_labels):
    combined_embeddings = train_embeddings + val_embeddings
    combined_labels = train_labels + val_labels
    return combined_embeddings, combined_labels

def train_attentive_classifier(train_embeddings, train_labels, use_class, num_classes=1000, num_epochs=10, learning_rate=1e-5, device='cuda'):
    # Create the embeddings dataset and dataloader
    train_dataset = EmbeddingsDataset(train_embeddings, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=False)

    attentive_classifier = AttentiveClassifier(num_classes=num_classes, use_class=use_class).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(attentive_classifier.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        attentive_classifier.train()
        total_loss = 0.0
        for patch_tokens, class_token, labels in train_loader:
            patch_tokens = patch_tokens.to(device).float()  
            class_token = class_token.to(device).float()    
            labels = labels.to(device).long()              
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = attentive_classifier((patch_tokens, class_token))
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    return attentive_classifier

def evaluate_knn(train_embeddings, test_embeddings, train_labels, test_labels):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine').fit(train_embeddings)
    test_knn_distances, test_knn_indices = neighbors.kneighbors(test_embeddings)
    t1_hits = sum(1 for i, label in enumerate(test_labels) if train_labels[test_knn_indices[i][0]] == label)
    top1_accuracy = t1_hits / len(test_labels)

    min_distances = [elem[0] for elem in test_knn_distances]
    return min_distances, top1_accuracy

def plot_KNN_ROC(closed_min_dist, open_min_dist, plot=False):
    yt = [1 for elem in open_min_dist] + [0 for elem in closed_min_dist]
    ys = open_min_dist + closed_min_dist
    fpr, tpr, _ = roc_curve(yt, ys)

    if plot:
        plt.rcParams["figure.figsize"] = (8,8)
        plt.rcParams.update({'font.size': 22})
        label_str = f'(AUROC: {auc(fpr, tpr):.3f})'
        plt.plot(fpr, tpr, label=label_str)

        plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
        plt.xlim([-0.01, 1.0])
        plt.ylim([-0.01, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc='lower right')
        plt.title('Closed/Open Classification ROC (NN Min Distance)')
        plt.show()

    return auc(fpr, tpr)
