import re
from collections import defaultdict

# Initialize a dictionary to store the best results for each dataset and pooling method
best_results = defaultdict(lambda: {'top1_acc': 0, 'learning_rate': None, 'msp_auc': None, 'mls_auc': None})

# Regular expressions to extract information from log lines
dataset_pattern = re.compile(
    r"INFO - Running experiment with dataset: (\S+), model: \S+, pooling method: (\S+), use_class: \S+, learning rate: ([\d\.eE+-]+)"
)
top1_acc_pattern = re.compile(r"INFO - Closed test set Top-1 acc. for config: ([\d.]+)")
msp_auc_pattern = re.compile(r"INFO - MSP ROC AUC for config: ([\d.]+)")
mls_auc_pattern = re.compile(r"INFO - MLS ROC AUC for config: ([\d.]+)")

# Function to process the log file
def process_log_file(log_file_path):
    current_config = {}
    
    with open(log_file_path, 'r') as file:
        for line in file:
            # Check if the line contains dataset and config information
            dataset_match = dataset_pattern.search(line)
            if dataset_match:
                current_config['dataset'] = dataset_match.group(1)
                current_config['pooling_method'] = dataset_match.group(2)
                # Convert learning rate to scientific notation and store it as a float
                current_config['learning_rate'] = float(f"{float(dataset_match.group(3)):.1e}")
            
            # Check if the line contains Top-1 accuracy
            top1_acc_match = top1_acc_pattern.search(line)
            if top1_acc_match:
                current_config['top1_acc'] = float(top1_acc_match.group(1))
            
            # Check if the line contains MSP ROC AUC
            msp_auc_match = msp_auc_pattern.search(line)
            if msp_auc_match:
                current_config['msp_auc'] = float(msp_auc_match.group(1))
            
            # Check if the line contains MLS ROC AUC
            mls_auc_match = mls_auc_pattern.search(line)
            if mls_auc_match:
                current_config['mls_auc'] = float(mls_auc_match.group(1))
            
            # Once all required metrics are available, check if this is the best config
            if 'top1_acc' in current_config and 'dataset' in current_config and 'msp_auc' in current_config and 'mls_auc' in current_config:
                key = (current_config['dataset'], current_config['pooling_method'])
                if (current_config['top1_acc'] > best_results[key]['top1_acc']):
                    best_results[key] = {
                        'top1_acc': current_config['top1_acc'],
                        'learning_rate': current_config['learning_rate'],
                        'msp_auc': current_config['msp_auc'],
                        'mls_auc': current_config['mls_auc']
                    }
                # Reset current_config for the next set of lines
                current_config = {}


# Use the function to process your log file
log_file_path = 'attentive_grid_results2.log'
process_log_file(log_file_path)

# Convert the best results to a list for easy reporting
best_results_list = [
    {'dataset': key[0], 'pooling_method': key[1], **metrics}
    for key, metrics in best_results.items()
]

# Display the results
# import ace_tools as tools; tools.display_dataframe_to_user(name="Best
# Results", dataframe=best_results_list)
# print(best_results_list)

# Print the best learning rates for each dataset and pooling method combination
print("Best learning rates for each dataset and pooling method combination:")
for key, result in best_results.items():
    dataset, pooling_method = key
    print(f"Dataset: {dataset}, Pooling Method: {pooling_method}, Best Learning Rate: {result['learning_rate']:.1e}")

print('done')