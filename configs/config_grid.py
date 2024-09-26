import os

MODEL = 'megadescriptor'

# DATASETS = ['FriesianCattle2015v2', 
#             'CTai',
#             'CZoo',
#             'DogFaceNet', 
#             'FriesianCattle2015v2',
#             'FriesianCattle2017',
#             'IPanda50',
#             'MacaqueFaces',
#             'NyalaData', 
#             'SeaTurtleIDHeads',
#             'StripeSpotter'
# ]

DATASETS = ['FriesianCattle2015v2', 
            'CTai',
            'CZoo',
            'DogFaceNet', 
            'FriesianCattle2017',
            'IPanda50',
            'MacaqueFaces',
            'NyalaData', 
            'SeaTurtleIDHeads',
            'StripeSpotter'
            'AerialCattle2017', 
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

# DATASETS = ['AerialCattle2017', 
#             'CTai',
#             'CZoo',
#             'DogFaceNet', 
#             'FriesianCattle2015v2',
#             'FriesianCattle2017',
#             'IPanda50',
#             'GreenSeaTurtles', 
#             'MacaqueFaces',
#             'MPDD', 
#             'NyalaData', 
#             'PolarBearVidID',
#             'SeaTurtleIDHeads'
#             'StripeSpotter'
# ]
ROOT_DIR = '/home/avisund/data/wildlife_datasets/'
BATCH_SIZE = [8, 16, 32, 64, 128]
LEARNING_RATE = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1]
CONFIG_PATH = 'configs/md-configs.json'

def get_dataset_root(dataset_name):
    return os.path.join(ROOT_DIR, dataset_name) + '/'