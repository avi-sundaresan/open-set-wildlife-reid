import pandas as pd
import numpy as np
import torch
from torchvision import transforms
import PIL
from functools import partial
from sklearn.neighbors import NearestNeighbors
from wildlife_datasets import datasets, splits
from models import ModelWithIntermediateLayers, create_linear_input
from datasets import CustomDataset, compute_full_embeddings
from config import DATASET, get_dataset_root, BATCH_SIZE
from utils import plot_KNN_ROC

from models import ModelWithIntermediateLayers, create_linear_input
from datasets import CustomDataset, EmbeddingsDataset, compute_full_embeddings
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14', skip_validation=True).to(device)
n_last_blocks = 1
autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.float)
feature_model = ModelWithIntermediateLayers(dinov2_vitg14, n_last_blocks, autocast_ctx).to("cuda")

transformation = transforms.Compose([
        transforms.Resize((224, 224), interpolation= PIL.Image.Resampling.BILINEAR, antialias=True),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet default mean and sd
    ])

root = get_dataset_root(DATASET)

if DATASET == 'StripeSpotter':
    root = root_dir + 'StripeSpotter/'
    datasets.StripeSpotter.get_data(root)
    d = datasets.StripeSpotter(root)
elif DATASET == 'FriesianCattle2017':
    root = root_dir + 'FriesianCattle2017/'
    datasets.FriesianCattle2017.get_data(root)
    d = datasets.FriesianCattle2017(root)

df = d.df
splitter = splits.OpenSetSplit(0.8, 0.1)
split = splitter.split(df)
idx_train, idx_test = split[0]

splits.analyze_split(df, idx_train, idx_test)

df_train, df_test = df.loc[idx_train], df.loc[idx_test]

train_identities = set(df_train['identity'])
test_identities = set(df_test['identity'])
closed_identities = train_identities.intersection(test_identities)
open_identities = test_identities - closed_identities

closed_identities_sorted = sorted(list(closed_identities))
closed_identity_to_label = {identity: idx for idx, identity in enumerate(closed_identities_sorted)}
open_identities_sorted = sorted(list(open_identities))
open_identity_to_label = {identity: idx + len(closed_identity_to_label) for idx, identity in enumerate(open_identities_sorted)}
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

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
closedtestloader = torch.utils.data.DataLoader(closed_test_dataset, batch_size=BATCH_SIZE, shuffle=True)
opentestloader = torch.utils.data.DataLoader(open_test_dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14', skip_validation=True).to(device)
n_last_blocks = 1
autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.float)
feature_model = ModelWithIntermediateLayers(dinov2_vitg14, n_last_blocks, autocast_ctx).to("cuda")

train_embeddings, train_labels = compute_full_embeddings(trainloader, feature_model, device)
closed_test_embeddings, closed_test_labels = compute_full_embeddings(closedtestloader, feature_model, device)
open_test_embeddings, open_test_labels = compute_full_embeddings(opentestloader, feature_model, device)

train_embeddings_f = []
for t in train_embeddings:
    train_embeddings_f += list([np.array(tr.cpu()) for tr in create_linear_input(t, use_avgpool=False, use_class=True)])
train_embeddings_f = np.array(train_embeddings_f)

closed_test_embeddings_f = []
for t in closed_test_embeddings:
    closed_test_embeddings_f += list([np.array(tr.cpu()) for tr in create_linear_input(t, use_avgpool=False, use_class=True)])
closed_test_embeddings_f = np.array(closed_test_embeddings_f)

open_test_embeddings_f = []
for t in open_test_embeddings:
    open_test_embeddings_f += list([np.array(tr.cpu()) for tr in create_linear_input(t, use_avgpool=False, use_class=True)])
open_test_embeddings_f = np.array(open_test_embeddings_f)

train_labels_f = np.array([np.array(l.cpu()) for l in train_labels])
closed_test_labels_f = np.array([np.array(l.cpu()) for l in closed_test_labels])
open_test_labels_f = np.array([np.array(l.cpu()) for l in open_test_labels])


neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute',metric='cosine').fit(train_embeddings_f)
train_knn_distances, train_knn_indices = neighbors.kneighbors(train_embeddings_f)
test_knn_distances, test_knn_indices = neighbors.kneighbors(closed_test_embeddings_f)
otest_knn_distances, otest_knn_indices = neighbors.kneighbors(open_test_embeddings_f)

t1_hits = 0
for i in range(len(closed_test_labels_f)):
    label = closed_test_labels_f[i]
    t1 = train_labels_f[test_knn_indices[i][0]]
    if t1 == label:
        t1_hits += 1
print('top 1:', t1_hits / len(closed_test_labels_f))

closed_min_dist = [elem[0] for elem in test_knn_distances]
open_min_dist = [elem[0] for elem in otest_knn_distances]



roc_auc = plot_KNN_ROC(closed_min_dist, open_min_dist)
print(roc_auc)