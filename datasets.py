import torch
from torch.utils.data import Dataset
from PIL import Image
from wildlife_datasets import datasets, splits

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
    elif dataset_name == 'MacaqueFaces':
        datasets.MacaqueFaces(root)
        d = datasets.MacaqueFaces(root)
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
