import os
import json

# def discover_models():
#     registry = {}
    
#     # 1. Get the directory where THIS file (e.g., factory.py) lives
#     current_dir = os.path.dirname(os.path.abspath(__file__))
    
#     # 2. Construct the path to the models folder relative to this file
#     # If your folder structure is: vh_checker/factory.py and vh_checker/models/
#     models_dir = os.path.join(current_dir, "models")
    
#     # Safety check: ensure the folder actually exists
#     if not os.path.exists(models_dir):
#         return registry

#     for item in os.listdir(models_dir):
#         config_path = os.path.join(models_dir, item, "config.json")
#         if os.path.exists(config_path):
#             with open(config_path, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#                 data['abs_path'] = os.path.abspath(os.path.dirname(config_path))
#                 # Using .get() for 'name' prevents a KeyError if the JSON is malformed
#                 name = data.get('name', item)
#                 registry[name] = data
                
#     return registry

import os
import json

def discover_models():
    registry = {}
    # Use absolute pathing relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "models") 

    if not os.path.exists(models_dir):
        return registry

    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if not os.path.isdir(item_path):
            continue
            
        config_path = os.path.join(item_path, "config.json")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Ensure the model name is the key
                    model_name = data.get("name", item).lower()
                    
                    # Store everything we need for the dispatcher and the CLI
                    registry[model_name] = {
                        "name": model_name,
                        "class_path": data.get("class_path"),
                        "conda_env": data.get("conda_env"),
                        "description": data.get("description", "No description available."),
                        "link": data.get("link", ""),
                        "abs_path": item_path
                    }
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse config for {item}")
                
    return registry
MODEL_REGISTRY = discover_models()