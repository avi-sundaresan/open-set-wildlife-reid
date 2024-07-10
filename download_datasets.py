import pandas as pd
import numpy as np
from wildlife_datasets import datasets, splits

DATASETS = ['AerialCattle2017', 
            'BelugaIDv2',
            'CTai',
            'CZoo',
            'DogFaceNet', 
            'FriesianCattle2015v2',
            'IPanda50',
            'GreenSeaTurtles', 
            'MPDD', 
            'NyalaData', 
            'PolarBearVidID',
            'SeaTurtleIDHeads'
]

for dataset in DATASETS:
    root = '/home/avisund/data/wildlife_datasets/' + dataset + '/'
    
    try:
        if dataset == 'AerialCattle2017':
            datasets.AerialCattle2017.get_data(root)
            d = datasets.AerialCattle2017(root)
        elif dataset == 'BelugaIDv2':
            datasets.BelugaIDv2.get_data(root)
            d = datasets.BelugaIDv2(root)
        elif dataset == 'CTai':
            datasets.CTai.get_data(root)
            d = datasets.CTai(root)
        elif dataset == 'CZoo':
            datasets.CZoo.get_data(root)
            d = datasets.CZoo(root)
        elif dataset == 'DogFaceNet':
            datasets.DogFaceNet.get_data(root)
            d = datasets.DogFaceNet(root)
        elif dataset == 'FriesianCattle2015v2':
            datasets.FriesianCattle2015v2.get_data(root)
            d = datasets.FriesianCattle2015v2(root)
        elif dataset == 'IPanda50':
            datasets.IPanda50.get_data(root)
            d = datasets.IPanda50(root)
        elif dataset == 'GreenSeaTurtles':
            datasets.GreenSeaTurtles.get_data(root)
            d = datasets.GreenSeaTurtles(root)
        elif dataset == 'MPDD':
            datasets.MPDD.get_data(root)
            d = datasets.MPDD(root)
        elif dataset == 'NyalaData':
            datasets.NyalaData.get_data(root)
            d = datasets.NyalaData(root)
        elif dataset == 'PolarBearVidID':
            datasets.PolarBearVidID.get_data(root)
            d = datasets.PolarBearVidID(root)
        elif dataset == 'SeaTurtleIDHeads':
            datasets.SeaTurtleIDHeads.get_data(root)
            d = datasets.SeaTurtleIDHeads(root)
        else:
            print(f"Dataset {dataset} not recognized.")
    
    except Exception as e:
        print(f"Error downloading {dataset}: {e}")
        continue

    try: 
        df = d.df
        splitter = splits.OpenSetSplit(0.8, 0.1)
        split = splitter.split(df)
        idx_train, idx_test = split[0]
    except Exception as e:
        print(f"Error with open-set split for dataset {dataset}: {e}")


