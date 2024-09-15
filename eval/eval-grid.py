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
from utils.utils import get_ROC, compute_embeddings, get_transformation, train_val_attentive_classifier, train_val_linear_classifier, eval_closed_set, eval_open_set

# Initialize logging
logging.basicConfig(filename='logs/attentive_grid_results-new.log', level=logging.INFO, 
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
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device: {device}')

    best_results = []

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

            trainloader, closedtestloader, opentestloader, valloader = create_dataloaders(root, df, idx_train, idx_test, get_transformation(args.model), val=True, batch_size=None)
            logging.info(f'Dataloaders created successfully')

            # Load feature extractor
            feature_extractor = load_model(args.model, device)
            logging.info('Feature extractor loaded successfully')

            # Compute embeddings
            dataloaders = [trainloader, valloader, closedtestloader, opentestloader]
            embeddings, labels = compute_embeddings(dataloaders, feature_extractor, device)
            train_embeddings, val_embeddings, closed_test_embeddings, open_test_embeddings = embeddings
            train_labels, val_labels, closed_test_labels, open_test_labels = labels
            logging.info('Embeddings computed successfully')

            num_classes = int(max(train_labels).item() + 1)
            
            for config in configs: 
                if args.model == 'dinov2' and config['pooling_method'] == 'none' and not config['use_class']:
                    raise ValueError("Invalid configuration: pooling_method='none' and use_class=False is not allowed.")
                
                best_val_acc = 0
                best_batch = None
                best_lr = None
                best_epoch = None

                for batch_size in args.batch_sizes:
                    for lr in args.learning_rates:
                        logging.info(f'Grid eval for classifiers: Running experiment with dataset: {dataset}, model: {args.model}, pooling method: {config["pooling_method"]}, use_class: {config["use_class"]}, learning rate: {lr}, batch size: {batch_size}')

                        if config['pooling_method'] == 'attentive':
                            epoch, val_acc = train_val_attentive_classifier(
                                train_embeddings, train_labels, val_embeddings, val_labels, 
                                 use_class=config['use_class'], device=device, num_classes=num_classes, learning_rate=lr, batch_size=batch_size
                            )
                            logging.info(f"Training stopped at epoch {epoch} for config: {config}")
                        elif config['pooling_method'] == 'linear':
                            epoch, val_acc  = train_val_linear_classifier(
                                train_embeddings, train_labels, val_embeddings, val_labels, 
                                use_class=config['use_class'], use_avgpool=True, device=device, num_classes=num_classes, learning_rate=lr, batch_size=batch_size
                            )
                            logging.info(f"Training stopped at epoch {epoch} for config: {config}")
                        elif config['pooling_method'] == 'none':
                            epoch, val_acc  = train_val_linear_classifier(
                                train_embeddings, train_labels, val_embeddings, val_labels, 
                                use_class=config['use_class'], use_avgpool=False, device=device, num_classes=num_classes, learning_rate=lr, batch_size=batch_size
                            )
                            logging.info(f"Training stopped at epoch {epoch} for config: {config}")

                        if val_acc > best_val_acc:
                            best_batch = batch_size
                            best_lr = lr
                            best_epoch = epoch
                            best_val_acc = val_acc
                
                best_results.append({
                            "dataset": dataset,
                            "pooling_method": config["pooling_method"],
                            "batch_size": best_batch,
                            "learning_rate": best_lr,
                            "best_epoch": best_epoch,
                            "best_val_acc": best_val_acc
                })

        except Exception as e:
            logging.error(f"Error occurred for dataset {dataset}, {str(e)}")
            continue
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    
    with open('configs/best_grid_results.json', 'w') as f:
        json.dump(best_results, f, indent=4)
    
    logging.info('Best results saved')
    
if __name__ == '__main__':
    main()
