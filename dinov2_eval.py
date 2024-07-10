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
import json
import timm

from wildlife_datasets import datasets, splits
from models import ModelWithIntermediateLayers, AttentiveEmbedder, create_linear_input
from datasets import CustomDataset, compute_full_embeddings
from config import DATASETS, MODEL, BATCH_SIZE, CONFIG_PATH, get_dataset_root
from utils import plot_KNN_ROC

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="KNN Classification Script")
    parser.add_argument('--datasets', type=list, default=DATASETS, help='Datasets to use')
    parser.add_argument('--model', type=str, default=MODEL, help='Feature extractor to use')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for data loading')
    parser.add_argument('--configs', type=str, default=CONFIG_PATH, help='Path to JSON file with list of configurations')
    return parser.parse_args()

def load_model(name, device):
    if name == 'dinov2':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14', skip_validation=True).to(device)
        n_last_blocks = 1
        autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.float)
        return ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx).to(device)
    if name == 'megadescriptor':
        model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
        return model
    else:
        raise ValueError(f"Unsupported feature extractor: {name}")
def get_transformation():
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=PIL.Image.Resampling.BILINEAR, antialias=True),
        transforms.ToTensor()
    ])

def prepare_datasets(root, dataset_name):
    if dataset_name == 'AerialCattle2017':
        datasets.AerialCattle2017.get_data(root)
        d = datasets.AerialCattle2017(root)
    elif dataset_name == 'CTai':
        datasets.CTai.get_data(root)
        d = datasets.CTai(root)
    elif dataset_name == 'CZoo':
        datasets.CZoo.get_data(root)
        d = datasets.CZoo(root)
    elif dataset_name == 'DogFaceNet':
        datasets.DogFaceNet.get_data(root)
        d = datasets.DogFaceNet(root)
    elif dataset_name == 'FriesianCattle2015v2':
        datasets.FriesianCattle2015v2.get_data(root)
        d = datasets.FriesianCattle2015v2(root)
    elif dataset_name == 'IPanda50':
        datasets.IPanda50.get_data(root)
        d = datasets.IPanda50(root)
    elif dataset_name == 'GreenSeaTurtles':
        datasets.GreenSeaTurtles.get_data(root)
        d = datasets.GreenSeaTurtles(root)
    elif dataset_name == 'MPDD':
        datasets.MPDD.get_data(root)
        d = datasets.MPDD(root)
    elif dataset_name == 'NyalaData':
        datasets.NyalaData.get_data(root)
        d = datasets.NyalaData(root)
    elif dataset_name == 'PolarBearVidID':
        datasets.PolarBearVidID.get_data(root)
        d = datasets.PolarBearVidID(root)
    elif dataset_name == 'SeaTurtleIDHeads':
        datasets.SeaTurtleIDHeads.get_data(root)
        d = datasets.SeaTurtleIDHeads(root)
    elif dataset_name == 'StripeSpotter':
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

def main():
    args = parse_args()

    with open(args.configs, 'r') as f:
        configs = json.load(f)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device: {device}')

    # Define transformations
    transformation = get_transformation()

    for dataset in args.datasets:
        # Load dataset
        root = get_dataset_root(dataset)
        logging.info(root)
        d = prepare_datasets(root, dataset)
        logging.info(f'Dataset {dataset} loaded successfully')

        # Split dataset
        df, idx_train, idx_test = split_dataset(d)
        logging.info('Dataset split successfully')

        # Create dataloaders
        trainloader, closedtestloader, opentestloader = create_dataloaders(root, df, idx_train, idx_test, transformation, args.batch_size)
        logging.info('Dataloaders created successfully')

        # Load feature extractor
        feature_extractor = load_model(args.model, device)
        logging.info('Feature extractor loaded successfully')

        # Compute embeddings
        dataloaders = [trainloader, closedtestloader, opentestloader]
        embeddings, labels = compute_embeddings(dataloaders, feature_extractor, device)
        train_embeddings, closed_test_embeddings, open_test_embeddings = embeddings
        train_labels, closed_test_labels, open_test_labels = labels

        for config in configs:
            if args.model == 'dinov2' and config['pooling_method'] == 'none' and not config['use_class']:
                raise ValueError("Invalid configuration: pooling_method='none' and use_class=False is not allowed.")

            # Initialize attentive pooler if needed
            attentive_embedder = AttentiveEmbedder() if config['pooling_method'] == 'attentive' else None

            logging.info(f'Running experiment with dataset: {dataset}, model: {args.model}, pooling method: {config["pooling_method"]}, use_class: {config["use_class"]}')
            train_embeddings_f, train_labels_f = flatten_embeddings(train_embeddings, train_labels, config['pooling_method'], config['use_class'], attentive_embedder)
            closed_test_embeddings_f, closed_test_labels_f = flatten_embeddings(closed_test_embeddings, closed_test_labels, config['pooling_method'], config['use_class'], attentive_embedder)
            open_test_embeddings_f, open_test_labels_f = flatten_embeddings(open_test_embeddings, open_test_labels, config['pooling_method'], config['use_class'], attentive_embedder)

            # Evaluate KNN for closed test set
            closed_min_dist, closed_top1_acc = evaluate_knn(train_embeddings_f, closed_test_embeddings_f, train_labels_f, closed_test_labels_f)

            # Evaluate KNN for open test set
            open_min_dist, open_top1_acc = evaluate_knn(train_embeddings_f, open_test_embeddings_f, train_labels_f, open_test_labels_f)

            # Plot ROC curve
            roc_auc = plot_KNN_ROC(closed_min_dist, open_min_dist)
            logging.info(f'Top-1 acc. for config: {closed_top1_acc:.4f}')
            logging.info(f'ROC AUC for config: {roc_auc:.4f}')  

if __name__ == '__main__':
    main()