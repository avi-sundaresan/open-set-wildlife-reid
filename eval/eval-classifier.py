import torch
import argparse
import logging
import json
import timm
from functools import partial

from models import ModelWithIntermediateLayers, ModelWithIntermediateLayersMD
from datasets.datasets import prepare_datasets, split_dataset, create_dataloaders
from configs.config import DATASETS, MODEL, BATCH_SIZE, CONFIG_PATH, BEST_LEARNING_RATES, get_dataset_root
from utils.utils import get_ROC, compute_embeddings, get_transformation, train_attentive_classifier, train_linear_classifier, eval_closed_set, eval_open_set

# Initialize logging
logging.basicConfig(filename='open_set_results.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Classifier Script")
    parser.add_argument('--datasets', type=list, default=DATASETS, help='Datasets to use')
    parser.add_argument('--model', type=str, default=MODEL, help='Feature extractor to use')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for data loading')
    parser.add_argument('--configs', type=str, default=CONFIG_PATH, help='Path to JSON file with list of configurations')
    return parser.parse_args()

def load_model(name, device):
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.float)
    if name == 'dinov2':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14', skip_validation=True).to(device)
        n_last_blocks = 1
        return ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx).to(device)
    if name == 'dinov2_reg':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg', skip_validation=True).to(device)
        n_last_blocks = 1
        return ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx).to(device)
    if name == 'megadescriptor':
        model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
        autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.float)
        return ModelWithIntermediateLayersMD(model, autocast_ctx).to(device)
    else:
        raise ValueError(f"Unsupported feature extractor: {name}")

def main():
    args = parse_args()

    with open(args.configs, 'r') as f:
        configs = json.load(f)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device: {device}')

    # Define transformations
    transformation = get_transformation(args.model)
    # print(f"Model: {args.model}")
    # print(f"Transformation: {transformation}")

    for dataset in args.datasets:
        # Load dataset
        root = get_dataset_root(dataset)
        logging.info(root)
        d = prepare_datasets(root, dataset)
        logging.info(f'Dataset {dataset} loaded successfully')

        # Split dataset
        df, idx_train, idx_test = split_dataset(d)
        logging.info('Dataset split successfully')

        # Create dataloaders
        trainloader, closedtestloader, opentestloader, valloader = create_dataloaders(root, df, idx_train, idx_test, transformation, args.batch_size)
        logging.info('Dataloaders created successfully')

        # Load feature extractor
        feature_extractor = load_model(args.model, device)
        logging.info('Feature extractor loaded successfully')

        # Compute embeddings
        dataloaders = [trainloader, closedtestloader, opentestloader]
        embeddings, labels = compute_embeddings(dataloaders, feature_extractor, device)
        train_embeddings, closed_test_embeddings, open_test_embeddings = embeddings
        train_labels, closed_test_labels, open_test_labels = labels
        
        for config in configs:
            if args.model == 'dinov2' and config['pooling_method'] == 'none' and not config['use_class']:
                raise ValueError("Invalid configuration: pooling_method='none' and use_class=False is not allowed.")
            
            # Load the best learning rate for the current dataset and pooling method
            best_lr = BEST_LEARNING_RATES[dataset][config['pooling_method']]
            logging.info(f"Using best learning rate: {best_lr} for dataset: {dataset}, pooling method: {config['pooling_method']}")

            num_classes = int(max(train_labels).item() + 1)
            if config['pooling_method'] == 'attentive':
                classifier = train_attentive_classifier(train_embeddings, train_labels, use_class=config['use_class'], device=device, num_classes=num_classes, learning_rate=best_lr)
            elif config['pooling_method'] == 'linear':
                classifier = train_linear_classifier(train_embeddings, train_labels, use_class=config['use_class'], use_avgpool=True, device=device, num_classes=num_classes, learning_rate=best_lr)
            elif config['pooling_method'] == 'none':
                classifier = train_linear_classifier(train_embeddings, train_labels, use_class=config['use_class'], use_avgpool=False, device=device, num_classes=num_classes, learning_rate=best_lr)

            closed_top1_acc, closed_msp, closed_mls = eval_closed_set(closed_test_embeddings, closed_test_labels, classifier)
            open_msp, open_mls = eval_open_set(open_test_embeddings, open_test_labels, classifier)

            logging.info(f'Running linear classifier experiment with dataset: {dataset}, model: {args.model}, pooling method: {config["pooling_method"]}, use_class: {config["use_class"]}')

            # Plot ROC curve 
            msp_roc_auc = get_ROC(closed_msp, open_msp, knn=False)
            mls_roc_auc = get_ROC(closed_mls, open_mls, knn=False)
            logging.info(f'Top-1 acc. for config: {closed_top1_acc:.4f}')
            logging.info(f'MSP ROC AUC for config: {msp_roc_auc:.4f}')  
            logging.info(f'MLS ROC AUC for config: {mls_roc_auc:.4f}')

if __name__ == '__main__':
    main()