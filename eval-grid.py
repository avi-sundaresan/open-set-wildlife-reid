import torch
import argparse
import logging
import json
import timm
import gc
from functools import partial

from models import ModelWithIntermediateLayers, ModelWithIntermediateLayersMD
from datasets import prepare_datasets, split_dataset, create_dataloaders
from config_grid import DATASETS, MODEL, BATCH_SIZE, CONFIG_PATH, LEARNING_RATE, get_dataset_root
from utils import get_ROC, flatten_embeddings, evaluate_knn, compute_embeddings, get_transformation, train_attentive_classifier, combine_train_val

# Initialize logging
logging.basicConfig(filename='attentive_grid_results.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="KNN Classification Script")
    parser.add_argument('--datasets', type=list, default=DATASETS, help='Datasets to use')
    parser.add_argument('--model', type=str, default=MODEL, help='Feature extractor to use')
    parser.add_argument('--batch_sizes', type=list, default=BATCH_SIZE, help='Batch sizes for data loading')
    parser.add_argument('--learning-rates', type=list, default=LEARNING_RATE, help='Learning rates for model')
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

    best_acc = 0
    best_config = None
    best_attentive_classifier = None

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
                trainloader, closedtestloader, opentestloader, valloader = create_dataloaders(root, df, idx_train, idx_test, transformation, batch_size, val=True)
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

                        # Initialize attentive pooler if needed
                        attentive_classifier = None
                        if config['pooling_method'] == 'attentive':
                            num_classes = int(max(train_labels).item() + 1)
                            attentive_classifier = train_attentive_classifier(train_embeddings, train_labels, use_class=config['use_class'], num_classes=num_classes, learning_rate=lr)

                        logging.info(f'Running experiment with dataset: {dataset}, model: {args.model}, pooling method: {config["pooling_method"]}, use_class: {config["use_class"]}, learning rate: {lr}, batch size: {batch_size}')
                        train_embeddings_f, train_labels_f = flatten_embeddings(train_embeddings, train_labels, config['pooling_method'], config['use_class'], attentive_classifier)
                        val_embeddings_f, val_labels_f = flatten_embeddings(val_embeddings, val_labels, config['pooling_method'], config['use_class'], attentive_classifier)

                        # Evaluate KNN for validation set
                        _, val_top1_acc = evaluate_knn(train_embeddings_f, val_embeddings_f, train_labels_f, val_labels_f)
                        logging.info(f'Validation Top-1 acc. for config: {val_top1_acc:.4f}')

                        if val_top1_acc > best_acc:
                            best_acc = val_top1_acc
                            best_config = (batch_size, lr, config)
                            best_attentive_classifier = attentive_classifier
        
            combined_train_embeddings, combined_train_labels = combine_train_val(train_embeddings, train_labels, val_embeddings, val_labels)

            # Retrain on the combined train and validation set
            best_attentive_classifier = train_attentive_classifier(combined_train_embeddings, combined_train_labels, use_class=best_config[2]['use_class'], num_classes=num_classes, learning_rate=best_config[1])

            # Evaluate on test set with best configuration
            logging.info(f'Best configuration: batch size={best_config[0]}, learning rate={best_config[1]}, config={best_config[2]}')

            train_embeddings_f, train_labels_f = flatten_embeddings(train_embeddings, train_labels, best_config[2]['pooling_method'], best_config[2]['use_class'], best_attentive_classifier)
            closed_test_embeddings_f, closed_test_labels_f = flatten_embeddings(closed_test_embeddings, closed_test_labels, best_config[2]['pooling_method'], best_config[2]['use_class'], best_attentive_classifier)
            open_test_embeddings_f, open_test_labels_f = flatten_embeddings(open_test_embeddings, open_test_labels, best_config[2]['pooling_method'], best_config[2]['use_class'], best_attentive_classifier)

            # Evaluate KNN for closed test set
            closed_min_dist, closed_top1_acc = evaluate_knn(train_embeddings_f, closed_test_embeddings_f, train_labels_f, closed_test_labels_f)
            logging.info(f'Closed test set Top-1 acc. for best config: {closed_top1_acc:.14f}')

            # Evaluate KNN for open test set
            open_min_dist, _ = evaluate_knn(train_embeddings_f, open_test_embeddings_f, train_labels_f, open_test_labels_f)
            logging.info(f'Open test set evaluation complete for best config.')

            # Plot ROC curve
            roc_auc = get_ROC(closed_min_dist, open_min_dist)
            logging.info(f'ROC AUC for best config: {roc_auc:.4f}')  
        except Exception as e:
            logging.error(f"Error occurred for dataset {dataset}: {str(e)}")
            continue
        finally:
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
