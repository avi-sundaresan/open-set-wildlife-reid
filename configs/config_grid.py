import os

MODEL = 'megadescriptor'

DATASETS = ['FriesianCattle2015v2', 
            'CTai',
            'CZoo',
            'DogFaceNet', 
            'FriesianCattle2015v2',
            'FriesianCattle2017',
            'IPanda50',
            'MacaqueFaces',
            'NyalaData', 
            'SeaTurtleIDHeads',
            'StripeSpotter'
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
BATCH_SIZE = [1]
LEARNING_RATE = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]
CONFIG_PATH = 'md-configs.json'

def get_dataset_root(dataset_name):
    return os.path.join(ROOT_DIR, dataset_name) + '/'