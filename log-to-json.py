import json
import re
from collections import defaultdict

def reformat_log_to_json(log_file_path, output_json_path):
    # Create a dictionary to hold the reformatted configuration data
    config_data = defaultdict(lambda: defaultdict(dict))
    
    # Regular expression to match the log entries (with multi-line support)
    log_entry_pattern = re.compile(r'- INFO - Best results for dataset (.+?) and (.+?) pooling method: (\{.*?\})', re.DOTALL)
    
    # Read the entire log file as a string
    with open(log_file_path, 'r') as log_file:
        raw_text = log_file.read()
        
    # Find all matches using the regular expression
    matches = log_entry_pattern.findall(raw_text)
    
    # Process each match to build the nested dictionary
    for dataset_name, pooling_method, config_json_str in matches:
        # Convert the configuration part from string to dictionary
        config_dict = json.loads(config_json_str)
        
        # Remove redundant keys (dataset, pooling_method) from config_dict
        config_dict.pop('dataset', None)
        config_dict.pop('pooling_method', None)
        
        # Store in the nested dictionary
        config_data[dataset_name][pooling_method] = config_dict

    # Write the reformatted dictionary to a JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(config_data, json_file, indent=4)

# Example usage
log_file_path = 'logs/grid-results.log'  # Replace with your log file path
output_json_path = 'configs/reformatted_configs.json'  # Replace with your desired output path
reformat_log_to_json(log_file_path, output_json_path)
