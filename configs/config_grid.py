import os

MODEL = 'megadescriptor'

# DATASETS = [
#     'AmvrakikosTurtles',
#     'SouthernProvinceTurtles'
# ]

DATASETS = ['FriesianCattle2015v2']

ROOT_DIR = '/home/avisund/data/wildlife_datasets/'
BATCH_SIZE = [8, 16, 32, 64, 128]
LEARNING_RATE = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1]
CONFIG_PATH = 'configs/md-configs.json'

def get_dataset_root(dataset_name):
    return os.path.join(ROOT_DIR, dataset_name) + '/'