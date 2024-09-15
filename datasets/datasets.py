import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
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
    elif dataset_name == 'SealID':
        datasets.SealID.get_data(root)
        d = datasets.SealID(root)
    elif dataset_name == 'AAUZebraFish':
        datasets.AAUZebraFish.get_data(root)
        d = datasets.AAUZebraFish(root)
    elif dataset_name == 'ATRW':
        datasets.ATRW.get_data(root)
        d = datasets.ATRW(root)
    elif dataset_name == 'BirdIndividualID':
        datasets.BirdIndividualID.get_data(root)
        d = datasets.BirdIndividualID(root)
    elif dataset_name == 'CatIndividualImages':
        datasets.CatIndividualImages.get_data(root)
        d = datasets.CatIndividualImages(root)
    elif dataset_name == 'CowDataset':
        datasets.CowDataset.get_data(root)
        d = datasets.CowDataset(root)
    elif dataset_name == 'Cows2021':
        datasets.Cows2021.get_data(root)
        d = datasets.Cows2021(root)
    elif dataset_name == 'Giraffes':
        datasets.Giraffes.get_data(root)
        d = datasets.Giraffes(root)
    elif dataset_name == 'GiraffeZebraID':
        datasets.GiraffeZebraID.get_data(root)
        d = datasets.GiraffeZebraID(root)
    elif dataset_name == 'HyenaID2022':
        datasets.HyenaID2022.get_data(root)
        d = datasets.HyenaID2022(root)
    elif dataset_name == 'LeopardID2022':
        datasets.LeopardID2022.get_data(root)
        d = datasets.LeopardID2022(root)
    elif dataset_name == 'MPDD':
        datasets.MPDD.get_data(root)
        d = datasets.MPDD(root)
    elif dataset_name == 'NDD20':
        datasets.NDD20.get_data(root)
        d = datasets.NDD20(root)
    elif dataset_name == 'OpenCows2020':
        datasets.OpenCows2020.get_data(root)
        d = datasets.OpenCows2020(root)
    elif dataset_name == 'SeaStarReID2023':
        datasets.SeaStarReID2023.get_data(root)
        d = datasets.SeaStarReID2023(root)
    elif dataset_name == 'SMALST':
        datasets.SMALST.get_data(root)
        d = datasets.SMALST(root)
    elif dataset_name == 'WhaleSharkID':
        datasets.WhaleSharkID.get_data(root)
        d = datasets.WhaleSharkID(root)
    elif dataset_name == 'ZindiTurtleRecall':
        datasets.ZindiTurtleRecall.get_data(root)
        d = datasets.ZindiTurtleRecall(root)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return d

def split_dataset(d):
    df = d.df
    splitter = splits.OpenSetSplit(0.8, 0.1)
    split = splitter.split(df)
    idx_train, idx_test = split[0]
    print(splits.analyze_split(df, idx_train, idx_test))
    return df, idx_train, idx_test

def create_dataloaders(root, df, idx_train, idx_test, transformation, batch_size, val=True):
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

    if val:
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            full_train_paths, train_labels, test_size=0.2, random_state=42
        )
        val_dataset = CustomDataset(val_paths, val_labels, transformation)
        valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    else:
        valloader = None

    train_dataset = CustomDataset(train_paths, train_labels, transformation)
    closed_test_dataset = CustomDataset(full_closed_test_paths, closed_test_labels, transformation)
    open_test_dataset = CustomDataset(full_open_test_paths, open_test_labels, transformation)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    closedtestloader = torch.utils.data.DataLoader(closed_test_dataset, batch_size=batch_size, shuffle=True)
    opentestloader = torch.utils.data.DataLoader(open_test_dataset, batch_size=batch_size, shuffle=True)

    return trainloader, closedtestloader, opentestloader, valloader
