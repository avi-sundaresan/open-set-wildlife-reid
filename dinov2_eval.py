import pandas as pd
import numpy as np
import torch
from torchvision import transforms
import PIL
from functools import partial
from sklearn.neighbors import NearestNeighbors
import argparse
import logging
from tqdm import tqdm

from wildlife_datasets import datasets, splits
from models import ModelWithIntermediateLayers, create_linear_input
from datasets import CustomDataset, compute_full_embeddings
from config import DATASET, get_dataset_root, BATCH_SIZE
from utils import plot_KNN_ROC

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="KNN Classification Script")
    parser.add_argument('--dataset', type=str, default=DATASET, help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for data loading')
    parser.add_argument('--use_avgpool', type=bool, default=False, help='Use average pooling')
    parser.add_argument('--use_class', type=bool, default=True, help='Use class token')
    parser.add_argument('--use_attentive_pooling', type=bool, default=False, help='Use attentive pooling')
    return parser.parse_args()

def load_model(device):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14', skip_validation=True).to(device)
    n_last_blocks = 1
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.float)
    return ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx).to(device)

def get_transformation():
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=PIL.Image.Resampling.BILINEAR, antialias=True),
        transforms.ToTensor()
    ])

def prepare_datasets(root, dataset_name):
    if dataset_name == 'StripeSpotter':
        datasets.StripeSpotter.get_data(root)
        d = datasets.StripeSpotter(root)
    elif dataset_name == 'FriesianCattle2017':
        datasets.FriesianCattle2017.get_data(root)
        d = datasets.FriesianCattle2017(root)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return d

def split_dataset(d):
    df = d.df
    splitter = splits.OpenSetSplit(0.8, 0.1)
    split = splitter.split(df)
    idx_train, idx_test = split[0]
    splits.analyze_split(df, idx_train, idx_test)
    return df, idx_train, idx_test

def create_dataloaders(root, df, idx_train, idx_test, transformation, batch_size):
    df_train, df_test = df.loc[idx_train], df.loc[idx_test]
    train_identities = set(df_train['identity'])
    test_identities = set(df_test['identity'])
    closed_identities = train_identities.intersection(test_identities)
    open_identities = test_identities - closed_identities

    closed_identity_to_label = {identity: idx for idx, identity in enumerate(sorted(closed_identities))}
    open_identity_to_label = {identity: idx + len(closed_identity_to_label) for idx, identity in enumerate(sorted(open_identities))}
    identity_to_label = {**closed_identity_to_label, **open_identity_to_label}

    df_train['label'] = df_train['identity'].map(identity_to_label)
    df_test['label'] = df_test['identity'].map(identity_to_label)

    train_paths = df_train[df_train['identity'].isin(closed_identities)]['path'].tolist()
    closed_test_paths = df_test[df_test['identity'].isin(closed_identities)]['path'].tolist()
    open_test_paths = df_test[df_test['identity'].isin(open_identities)]['path'].tolist()

    train_labels = df_train[df_train['identity'].isin(closed_identities)]['label'].tolist()
    closed_test_labels = df_test[df_test['identity'].isin(closed_identities)]['label'].tolist()
    open_test_labels = df_test[df_test['identity'].isin(open_identities)]['label'].tolist()

    full_train_paths = [root + elem for elem in train_paths]
    full_closed_test_paths = [root + elem for elem in closed_test_paths]
    full_open_test_paths = [root + elem for elem in open_test_paths]

    train_dataset = CustomDataset(full_train_paths, train_labels, transformation)
    closed_test_dataset = CustomDataset(full_closed_test_paths, closed_test_labels, transformation)
    open_test_dataset = CustomDataset(full_open_test_paths, open_test_labels, transformation)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    closedtestloader = torch.utils.data.DataLoader(closed_test_dataset, batch_size=batch_size, shuffle=True)
    opentestloader = torch.utils.data.DataLoader(open_test_dataset, batch_size=batch_size, shuffle=True)

    return trainloader, closedtestloader, opentestloader

def compute_embeddings(dataloaders, feature_model, device):
    embeddings = []
    labels = []
    for loader in dataloaders:
        e, l = compute_full_embeddings(loader, feature_model, device)
        embeddings.append(e)
        labels.append(l)
    return embeddings, labels

def flatten_embeddings(embeddings, labels, use_avgpool, use_class, use_attentive_pooling):
    embeddings_f = []
    if use_attentive_pooling:
        
        for t in embeddings:
            embeddings_f += list([np.array(tr.cpu()) for tr in create_linear_input(t, use_avgpool=use_avgpool, use_class=use_class)])
    labels_f = np.array([np.array(l.cpu()) for label in labels for l in label])
    return np.array(embeddings_f), labels_f

def evaluate_knn(train_embeddings, test_embeddings, train_labels, test_labels):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine').fit(train_embeddings)
    test_knn_distances, test_knn_indices = neighbors.kneighbors(test_embeddings)

    t1_hits = sum(1 for i, label in enumerate(test_labels) if train_labels[test_knn_indices[i][0]] == label)
    top1_accuracy = t1_hits / len(test_labels)

    min_distances = [elem[0] for elem in test_knn_distances]
    return min_distances, top1_accuracy

def main():
    args = parse_args()

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device: {device}')

    # Load model
    feature_model = load_model(device)
    logging.info('Model loaded successfully')

    # Define transformations
    transformation = get_transformation()

    # Load dataset
    root = get_dataset_root(args.dataset)
    logging.info(root)
    d = prepare_datasets(root, args.dataset)
    logging.info('Dataset loaded successfully')

    # Split dataset
    df, idx_train, idx_test = split_dataset(d)
    logging.info('Dataset split successfully')

    # Create dataloaders
    trainloader, closedtestloader, opentestloader = create_dataloaders(root, df, idx_train, idx_test, transformation, args.batch_size)
    logging.info('Dataloaders created successfully')

    # Compute embeddings
    dataloaders = [trainloader, closedtestloader, opentestloader]
    embeddings, labels = compute_embeddings(dataloaders, feature_model, device)
    train_embeddings, closed_test_embeddings, open_test_embeddings = embeddings
    train_labels, closed_test_labels, open_test_labels = labels

    # Run experiments with different configurations
    configs = [
        {'use_avgpool': False, 'use_class': True},
        {'use_avgpool': True, 'use_class': False},
        {'use_avgpool': True, 'use_class': True},
        # Add more configurations if needed
    ]

    for config in configs:
        logging.info(f'Running experiment with config: {config}')
        train_embeddings_f, train_labels_f = flatten_embeddings(train_embeddings, train_labels, config['use_avgpool'], config['use_class'])
        closed_test_embeddings_f, closed_test_labels_f = flatten_embeddings(closed_test_embeddings, closed_test_labels, config['use_avgpool'], config['use_class'])
        open_test_embeddings_f, open_test_labels_f = flatten_embeddings(open_test_embeddings, open_test_labels, config['use_avgpool'], config['use_class'])

        # Evaluate KNN for closed test set
        closed_min_dist, closed_top1_acc = evaluate_knn(train_embeddings_f, closed_test_embeddings_f, train_labels_f, closed_test_labels_f)

        # Evaluate KNN for open test set
        open_min_dist, open_top1_acc = evaluate_knn(train_embeddings_f, open_test_embeddings_f, train_labels_f, open_test_labels_f)

        # Plot ROC curve
        roc_auc = plot_KNN_ROC(closed_min_dist, open_min_dist)
        logging.info(f'Top-1 acc. for config {config}: {closed_top1_acc:.4f}')
        logging.info(f'ROC AUC for config {config}: {roc_auc:.4f}')

if __name__ == '__main__':
    main()
