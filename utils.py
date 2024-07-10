import numpy as np
import torch
from torchvision import transforms
import PIL
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm 

from models import create_linear_input

def get_transformation(model):
    if model == "dinov2":
        return transforms.Compose([
            transforms.Resize((224, 224), interpolation=PIL.Image.Resampling.BILINEAR, antialias=True),
            transforms.ToTensor()
        ])
    elif model == "megadescriptor":
        return transforms.Compose([
        transforms.Resize((384, 384), interpolation= PIL.Image.Resampling.BILINEAR, antialias=True),
        transforms.ToTensor(),
    ])

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

def flatten_embeddings(embeddings, labels, pooling_method, use_class, attentive_embedder=None):
    embeddings_f = []

    if pooling_method == 'attentive':
        for embedding in embeddings:
            device = embedding[0].device
            attentive_embedder = attentive_embedder.to(device)
            attended_output = attentive_embedder.embed(embedding, use_class=use_class)
            embeddings_f.append(attended_output.cpu().detach().numpy())
    else:
        for embedding in embeddings:
            linear_input = create_linear_input(embedding, use_avgpool=(pooling_method == 'linear'), use_class=use_class)
            embeddings_f.extend([np.array(tr.cpu()) for tr in linear_input])

    labels_f = [np.array(l.cpu()) for label in labels for l in label]
    embeddings_f = np.vstack(embeddings_f)
    return np.array(embeddings_f), np.array(labels_f)

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
