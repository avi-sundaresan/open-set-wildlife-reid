import pandas as pd
import numpy as np
from wildlife_datasets import datasets, splits

DATASETS = ['AerialCattle2017', 
            'BelugaIDv2',
            'GreenSeaTurtles', 
            'MPDD', 
            'PolarBearVidID',
            'SealID', 
            'AAUZebraFish',
            'ATRW',
            'BirdIndividualID',
            'CatIndividualImages',
            'CowDataset',
            'Cows2021',
            'Giraffes',
            'GiraffeZebraID',
            'HyenaID2022',
            'LeopardID2022',
            'MPDD',
            'NDD20',
            'OpenCows2020',
            'SeaStarReID2023',
            'SMALST',
            'WhaleSharkID',
            'ZindiTurtleRecall'
]

DATASETS = ['AerialCattle2017', 
            'MPDD', 
            'PolarBearVidID',
            'CatIndividualImages',
            'CowDataset',
            'Cows2021',
            'Giraffes',
            'MPDD',
            'NDD20',
            'OpenCows2020',
            'SeaStarReID2023',
            'ZindiTurtleRecall'
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
        elif dataset == 'SealID':
            datasets.SealID.get_data(root)
            d = datasets.SealID(root)
        elif dataset == 'AAUZebraFish':
            datasets.AAUZebraFish.get_data(root)
            d = datasets.AAUZebraFish(root)
        elif dataset == 'ATRW':
            datasets.ATRW.get_data(root)
            d = datasets.ATRW(root)
        elif dataset == 'BirdIndividualID':
            datasets.BirdIndividualID.get_data(root)
            d = datasets.BirdIndividualID(root)
        elif dataset == 'CatIndividualImages':
            datasets.CatIndividualImages.get_data(root)
            d = datasets.CatIndividualImages(root)
        elif dataset == 'CowDataset':
            datasets.CowDataset.get_data(root)
            d = datasets.CowDataset(root)
        elif dataset == 'Cows2021':
            datasets.Cows2021.get_data(root)
            d = datasets.Cows2021(root)
        elif dataset == 'Giraffes':
            datasets.Giraffes.get_data(root)
            d = datasets.Giraffes(root)
        elif dataset == 'GiraffeZebraID':
            datasets.GiraffeZebraID.get_data(root)
            d = datasets.GiraffeZebraID(root)
        elif dataset == 'HyenaID2022':
            datasets.HyenaID2022.get_data(root)
            d = datasets.HyenaID2022(root)
        elif dataset == 'LeopardID2022':
            datasets.LeopardID2022.get_data(root)
            d = datasets.LeopardID2022(root)
        elif dataset == 'MPDD':
            datasets.MPDD.get_data(root)
            d = datasets.MPDD(root)
        elif dataset == 'NDD20':
            datasets.NDD20.get_data(root)
            d = datasets.NDD20(root)
        elif dataset == 'OpenCows2020':
            datasets.OpenCows2020.get_data(root)
            d = datasets.OpenCows2020(root)
        elif dataset == 'SeaStarReID2023':
            datasets.SeaStarReID2023.get_data(root)
            d = datasets.SeaStarReID2023(root)
        elif dataset == 'SMALST':
            datasets.SMALST.get_data(root)
            d = datasets.SMALST(root)
        elif dataset == 'WhaleSharkID':
            datasets.WhaleSharkID.get_data(root)
            d = datasets.WhaleSharkID(root)
        elif dataset == 'ZindiTurtleRecall':
            datasets.ZindiTurtleRecall.get_data(root)
            d = datasets.ZindiTurtleRecall(root)
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

# not_downloaded = []

# for dataset in DATASETS:
#     root = '/home/avisund/data/wildlife_datasets/' + dataset + '/'
    
#     try:
#         if dataset == 'AerialCattle2017':
#             d = datasets.AerialCattle2017(root)
#         elif dataset == 'BelugaIDv2':
#             d = datasets.BelugaIDv2(root)
#         elif dataset == 'CTai':
#             d = datasets.CTai(root)
#         elif dataset == 'CZoo':
#             d = datasets.CZoo(root)
#         elif dataset == 'DogFaceNet':
#             d = datasets.DogFaceNet(root)
#         elif dataset == 'FriesianCattle2015v2':
#             d = datasets.FriesianCattle2015v2(root)
#         elif dataset == 'IPanda50':
#             d = datasets.IPanda50(root)
#         elif dataset == 'GreenSeaTurtles':
#             d = datasets.GreenSeaTurtles(root)
#         elif dataset == 'MPDD':
#             d = datasets.MPDD(root)
#         elif dataset == 'NyalaData':
#             d = datasets.NyalaData(root)
#         elif dataset == 'PolarBearVidID':
#             d = datasets.PolarBearVidID(root)
#         elif dataset == 'SeaTurtleIDHeads':
#             d = datasets.SeaTurtleIDHeads(root)
#         elif dataset == 'SealID':
#             d = datasets.SealID(root)
#         elif dataset == 'AAUZebraFish':
#             d = datasets.AAUZebraFish(root)
#         elif dataset == 'ATRW':
#             d = datasets.ATRW(root)
#         elif dataset == 'BirdIndividualID':
#             d = datasets.BirdIndividualID(root)
#         elif dataset == 'CatIndividualImages':
#             d = datasets.CatIndividualImages(root)
#         elif dataset == 'CowDataset':
#             d = datasets.CowDataset(root)
#         elif dataset == 'Cows2021':
#             d = datasets.Cows2021(root)
#         elif dataset == 'Giraffes':
#             d = datasets.Giraffes(root)
#         elif dataset == 'GiraffeZebraID':
#             d = datasets.GiraffeZebraID(root)
#         elif dataset == 'HyenaID2022':
#             d = datasets.HyenaID2022(root)
#         elif dataset == 'LeopardID2022':
#             d = datasets.LeopardID2022(root)
#         elif dataset == 'MPDD':
#             d = datasets.MPDD(root)
#         elif dataset == 'NDD20':
#             d = datasets.NDD20(root)
#         elif dataset == 'OpenCows2020':
#             d = datasets.OpenCows2020(root)
#         elif dataset == 'SeaStarReID2023':
#             d = datasets.SeaStarReID2023(root)
#         elif dataset == 'SMALST':
#             d = datasets.SMALST(root)
#         elif dataset == 'WhaleSharkID':
#             d = datasets.WhaleSharkID(root)
#         elif dataset == 'ZindiTurtleRecall':
#             d = datasets.ZindiTurtleRecall(root)
#         else:
#             print(f"Dataset {dataset} not recognized.")
#     except Exception as e:
#         print(f"Error downloading {dataset}. {e}")
#         not_downloaded.append(dataset)
#         continue
