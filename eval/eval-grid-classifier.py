import torch
import argparse
import logging
import json
import timm
import gc
from functools import partial

from models import ModelWithIntermediateLayers, ModelWithIntermediateLayersMD
from datasets.datasets import prepare_datasets, split_dataset, create_dataloaders
from configs.config_grid import DATASETS, MODEL, BATCH_SIZE, CONFIG_PATH, LEARNING_RATE, get_dataset_root
from utils.utils import get_ROC, compute_embeddings, get_transformation, train_attentive_classifier, train_linear_classifier, eval_closed_set, eval_open_set, combine_train_val

# Initialize logging
logging.basicConfig(filename='attentive_grid_results.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Classifier Grid Search Script")
    parser.add_argument('--datasets', type=list, default=DATASETS, help='Datasets to use')
    parser.add_argument('--model', type=str, default=MODEL, help='Feature extractor to use')
    parser.add_argument('--batch_sizes', type=list, default=BATCH_SIZE, help='Batch sizes for data loading')
    parser.add_argument('--learning_rates', type=list, default=LEARNING_RATE, help='Learning rates for classifier')
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

    best_acc = 0
    best_config = None
    best_classifier = None

    for dataset in args.datasets:
        # Load dataset
        try:
            torch.cuda.empty_cache()
            root = get_dataset_root(dataset)
            logging.info(root)
            d = prepare_datasets(root, dataset)
            logging.info(f'Dataset {dataset} loaded successfully')

            # Split dataset
            df, idx_train, idx_test = split_dataset(d)
            logging.info('Dataset split successfully')

            # Create dataloaders
            for batch_size in args.batch_sizes:
                trainloader, closedtestloader, opentestloader, valloader = create_dataloaders(root, df, idx_train, idx_test, get_transformation(args.model), batch_size, val=True)
                logging.info(f'Dataloaders created successfully with batch size {batch_size}')

                # Load feature extractor
                feature_extractor = load_model(args.model, device)
                logging.info('Feature extractor loaded successfully')

                # Compute embeddings
                dataloaders = [trainloader, valloader, closedtestloader, opentestloader]
                embeddings, labels = compute_embeddings(dataloaders, feature_extractor, device)
                train_embeddings, val_embeddings, closed_test_embeddings, open_test_embeddings = embeddings
                train_labels, val_labels, closed_test_labels, open_test_labels = labels

                for lr in args.learning_rates:
                    for config in configs:
                        if args.model == 'dinov2' and config['pooling_method'] == 'none' and not config['use_class']:
                            raise ValueError("Invalid configuration: pooling_method='none' and use_class=False is not allowed.")

                        # Initialize classifier based on config
                        num_classes = int(max(train_labels).item() + 1)
                        if config['pooling_method'] == 'attentive':
                            classifier = train_attentive_classifier(train_embeddings, train_labels, use_class=config['use_class'], device=device, num_classes=num_classes, learning_rate=lr)
                        elif config['pooling_method'] == 'linear':
                            classifier = train_linear_classifier(train_embeddings, train_labels, use_class=config['use_class'], use_avgpool=True, device=device, num_classes=num_classes)
                        elif config['pooling_method'] == 'none':
                            classifier = train_linear_classifier(train_embeddings, train_labels, use_class=config['use_class'], use_avgpool=False, device=device, num_classes=num_classes)
                        else:
                            classifier = None

                        logging.info(f'Running experiment with dataset: {dataset}, model: {args.model}, pooling method: {config["pooling_method"]}, use_class: {config["use_class"]}, learning rate: {lr}, batch size: {batch_size}')

                        # Evaluate closed set
                        closed_top1_acc, closed_msp, closed_mls = eval_closed_set(closed_test_embeddings, closed_test_labels, classifier)
                        logging.info(f'Closed test set Top-1 acc. for config: {closed_top1_acc:.4f}')

                        # Evaluate open set
                        open_msp, open_mls = eval_open_set(open_test_embeddings, open_test_labels, classifier)
                        logging.info(f'Open set evaluation complete for config.')

                        # Plot ROC curve
                        msp_roc_auc = get_ROC(closed_msp, open_msp, knn=False)
                        mls_roc_auc = get_ROC(closed_mls, open_mls, knn=False)
                        logging.info(f'MSP ROC AUC for config: {msp_roc_auc:.4f}')
                        logging.info(f'MLS ROC AUC for config: {mls_roc_auc:.4f}')

                        if closed_top1_acc > best_acc:
                            best_acc = closed_top1_acc
                            best_config = (batch_size, lr, config)
                            best_classifier = classifier

            combined_train_embeddings, combined_train_labels = combine_train_val(train_embeddings, train_labels, val_embeddings, val_labels)

            # Retrain on the combined train and validation set with the best config
            if best_config[2]['pooling_method'] == 'attentive':
                best_classifier = train_attentive_classifier(combined_train_embeddings, combined_train_labels, use_class=best_config[2]['use_class'], device=device, num_classes=num_classes, learning_rate=best_config[1])

            # Final evaluation on test set with best configuration
            logging.info(f'Best configuration: batch size={best_config[0]}, learning rate={best_config[1]}, config={best_config[2]}')

            closed_top1_acc, closed_msp, closed_mls = eval_closed_set(closed_test_embeddings, closed_test_labels, best_classifier)
            open_msp, open_mls = eval_open_set(open_test_embeddings, open_test_labels, best_classifier)
            logging.info(f'Final closed test set Top-1 acc. for best config: {closed_top1_acc:.4f}')

            # Plot final ROC curve
            msp_roc_auc = get_ROC(closed_msp, open_msp, knn=False)
            mls_roc_auc = get_ROC(closed_mls, open_mls, knn=False)
            logging.info(f'Final MSP ROC AUC for best config: {msp_roc_auc:.4f}')
            logging.info(f'Final MLS ROC AUC for best config: {mls_roc_auc:.4f}')
        
        except Exception as e:
            logging.error(f"Error occurred for dataset {dataset}: {str(e)}")
            continue
        finally:
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
