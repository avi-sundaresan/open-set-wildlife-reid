import torch
import argparse
import logging
import json
import timm
import os
from functools import partial
from PIL import Image

from datasets.datasets import prepare_datasets, split_dataset, create_dataloaders
from configs.config import DATASETS, MODEL, CONFIG_PATH, get_dataset_root

# Initialize logging
logging.basicConfig(filename='logs/dataset_info.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset stats")
    parser.add_argument('--datasets', type=list, default=DATASETS, help='Datasets to use')
    parser.add_argument('--configs', type=str, default=CONFIG_PATH, help='Path to JSON file with list of configurations')
    return parser.parse_args()

def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size / (1024 ** 3)

def main():
    args = parse_args()

    with open(args.configs, 'r') as f:
        configs = json.load(f)

    for dataset in args.datasets:
        # Load dataset
        root = get_dataset_root(dataset)
        logging.info(root)
        d = prepare_datasets(root, dataset)
        logging.info(f'Dataset {dataset} loaded successfully')

        # Split dataset
        df, idx_train, idx_test = split_dataset(d)
        logging.info('Dataset split successfully')

        logging.info(f'Number of dataset images: {len(df)}')
        id_num = len(set(df['identity']))
        logging.info(f'Number of dataset identities: { id_num }')

        image_filepaths = df['path']
        widths = []
        heights = []
        for path in image_filepaths:
            image = Image.open(root + path)
            width, height = image.size
            widths.append(width)
            heights.append(height)

        logging.info(f'Avg. image width: {sum(widths) / len(widths)}')
        logging.info(f'Avg. image height: {sum(heights) / len(heights)}')

        size = get_size(root)
        logging.info(f'Dataset size: {size} GB')

if __name__ == '__main__':
    main()