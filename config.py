import os

MODEL = 'dinov2'
# MODEL = 'megadescriptor'

DATASETS = ['StripeSpotter']
# DATASETS = ['CTai',
#             'CZoo',
#             'DogFaceNet', 
#             'FriesianCattle2017',
#             'IPanda50',
#             'MacaqueFaces',
#             'MPDD', 
#             'PolarBearVidID',
# ]

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
BATCH_SIZE = 1
CONFIG_PATH = 'dinov2-configs.json'
# CONFIG_PATH = 'md-configs.json'

def get_dataset_root(dataset_name):
    return os.path.join(ROOT_DIR, dataset_name) + '/'