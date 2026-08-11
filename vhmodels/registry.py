# The files is used to 'discover' all the available models in the package.
# Since the models are decoupled (no package-like connection between them),
# because they don't share an environment, it's necessary.
# Discovering models is e.g. necessary for the vh-checker list
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
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    # Ensure the model name is the key
                    model_name = data.get("name", item).lower()

                    # Store everything we need for the dispatcher and the CLI
                    registry[model_name] = {
                        "name": model_name,
                        "supported_platforms": data.get("supported_platforms", []),
                        "environment_files": data.get("environment_files", {}),
                        "apptainer": data.get("apptainer", {}),
                        "class_path": data.get("class_path"),
                        "conda_env": data.get("conda_env"),
                        "description": data.get(
                            "description", "No description available."
                        ),
                        "link": data.get("link", ""),
                        "abs_path": item_path,
                    }
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse config for {item}")

    return registry


MODEL_REGISTRY = discover_models()
