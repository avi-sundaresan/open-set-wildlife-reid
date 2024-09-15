import os

# MODEL = 'dinov2_reg'
# MODEL = 'dinov2'
MODEL = 'megadescriptor'

DATASETS = ['CTai',
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
BATCH_SIZE = 1
# CONFIG_PATH = 'dinov2-configs-2.json'
CONFIG_PATH = 'configs/md-configs.json'

BEST_LEARNING_RATES = {
    "FriesianCattle2015v2": {
        "linear": 5.0e-03,
        "attentive": 1.0e-04,
    },
    "CTai": {
        "linear": 2.0e-05,
        "attentive": 2.0e-05,
    },
    "CZoo": {
        "linear": 5.0e-05,
        "attentive": 1.0e-05,
    },
    "DogFaceNet": {
        "linear": 1.0e-04,
        "attentive": 2.0e-05,
    },
    "FriesianCattle2017": {
        "linear": 1.0e-04,
        "attentive": 1.0e-05,
    },
    "IPanda50": {
        "linear": 5.0e-04,
        "attentive": 2.0e-05,
    },
    "MacaqueFaces": {
        "linear": 2.0e-05,
        "attentive": 1.0e-05,
    },
    "NyalaData": {
        "linear": 1.0e-05,
        "attentive": 5.0e-05,
    },
    "SeaTurtleIDHeads": {
        "linear": 2.0e-05,
        "attentive": 2.0e-05,
    },
    "StripeSpotter": {
        "linear": 2.0e-03,
        "attentive": 1.0e-04,
    }
}

def get_dataset_root(dataset_name):
    return os.path.join(ROOT_DIR, dataset_name) + '/'