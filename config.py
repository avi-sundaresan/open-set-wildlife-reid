import os

MODEL = 'dinov2'
# DATASET = 'StripeSpotter'
DATASET = 'FriesianCattle2017'
ROOT_DIR = '/home/avisund/data/wildlife_datasets/'
BATCH_SIZE = 1
CONFIG_PATH = 'configs.json'

def get_dataset_root(dataset_name):
    return os.path.join(ROOT_DIR, dataset_name) + '/'