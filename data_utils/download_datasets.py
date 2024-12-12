import pandas as pd
import numpy as np
from wildlife_datasets import splits, datasets
from wildlife_datasets.datasets.aerial_cattle import AerialCattle2017
from wildlife_datasets.datasets.beluga_id import BelugaIDv2
from wildlife_datasets.datasets.ctai import CTai
from wildlife_datasets.datasets.czoo import CZoo
from wildlife_datasets.datasets.dog_face_net import DogFaceNet
from wildlife_datasets.datasets.friesian_cattle import FriesianCattle2015v2
from wildlife_datasets.datasets.ipanda import IPanda50
from wildlife_datasets.datasets.mpdd import MPDD
from wildlife_datasets.datasets.nyala_data import NyalaData
from wildlife_datasets.datasets.polar_bear_vid_id import PolarBearVidID
from wildlife_datasets.datasets.sea_turtle_id import SeaTurtleIDHeads
from wildlife_datasets.datasets.seal_id import SealID
from wildlife_datasets.datasets.aau_zebrafish import AAUZebraFish
from wildlife_datasets.datasets.atrw import ATRW
from wildlife_datasets.datasets.bird_individual_id import BirdIndividualID
from wildlife_datasets.datasets.cat_individual_images import CatIndividualImages
from wildlife_datasets.datasets.cow_dataset import CowDataset
from wildlife_datasets.datasets.cows import Cows2021
from wildlife_datasets.datasets.giraffes import Giraffes
from wildlife_datasets.datasets.giraffe_zebra_id import GiraffeZebraID
from wildlife_datasets.datasets.hyena_id import HyenaID2022
from wildlife_datasets.datasets.leopard_id import LeopardID2022
from wildlife_datasets.datasets.ndd import NDD20
from wildlife_datasets.datasets.open_cows import OpenCows2020
# from wildlife_datasets.datasets.sea_star_reid import SeaStarReID2023
from wildlife_datasets.datasets.smalst import SMALST
from wildlife_datasets.datasets.whaleshark_id import WhaleSharkID
from wildlife_datasets.datasets.zindi_turtle_recall import ZindiTurtleRecall
from wildlife_datasets.datasets.amvrakikos_turtles import AmvrakikosTurtles
from wildlife_datasets.datasets.reunion_turtles import ReunionTurtles
from wildlife_datasets.datasets.southern_province_turtles import SouthernProvinceTurtles
from wildlife_datasets.datasets.drosophila import Drosophila
from wildlife_datasets.datasets.chicks4free_id import Chicks4FreeID

DATASETS = [
    'AmvrakikosTurtles',
    'ReunionTurtles',
    'SouthernProvinceTurtles',
    'SeaStarReID2023'
]


for dataset in DATASETS:
    root = '/home/avisund/data/wildlife_datasets/' + dataset + '/'
    try:
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
        elif dataset_name == 'AmvrakikosTurtles':
            datasets.AmvrakikosTurtles.get_data(root)
            d = datasets.AmvrakikosTurtles(root)
        elif dataset_name == 'ReunionTurtles':
            datasets.ReunionTurtles.get_data(root)
            d = datasets.ReunionTurtles(root)
        elif dataset_name == 'SouthernProvinceTurtles':
            datasets.SouthernProvinceTurtles.get_data(root)
            d = datasets.SouthernProvinceTurtles(root)
        elif dataset_name == 'Drosophila':
            datasets.Drosophila.get_data(root)
            d = datasets.Drosophila(root)
        elif dataset_name == 'Chicks4FreeID':
            datasets.Chicks4FreeID.get_data(root)
            d = datasets.Chicks4FreeID(root)
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
        print(f"Split dataset {dataset} successfully.")
    except Exception as e:
        print(f"Error with open-set split for dataset {dataset}: {e}")
