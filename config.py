import os

DATASET = 'StripeSpotter'
ROOT_DIR = '/home/avisund/data/wildlife_datasets/'
BATCH_SIZE = 1

def get_dataset_root(dataset_name):
    return os.path.join(ROOT_DIR, dataset_name)
