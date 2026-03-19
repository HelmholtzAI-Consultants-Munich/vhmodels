from .base import REGISTRY

import os
import json
import subprocess
from pathlib import Path
import re
import pandas as pd

project_root = str(Path(__file__).parent.parent.parent.resolve())

# 2. Setup environment variables for the subprocess
# This tells the Conda environment where to find the 'vhmodels' folder
env = os.environ.copy()
env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

class ModelProxy:
    def __init__(self, project, env_name, model=None):
        self.project = project
        self.model = model
        self.env_name = env_name

    def transform(self, input, **kwargs):
        if not self._env_exists():
            raise RuntimeError(
                f"The environment '{self.env_name}' does not exist. "
                f"Please run 'vh-checker create-env {self.project}' first."
            )
        
        if isinstance(input, str) and os.path.exists(input):
        # It's a file path; pass the path string directly
            input = input
        elif isinstance(input, pd.DataFrame):
            input.to_dict(orient="records")
        else:
            # It's a list or dictionary; serialize to JSON string
            input = json.dumps(input)
        
        # We point to the runner inside vh_checker
        cmd = [
            "conda", "run", "-n", self.env_name,
            "python", "-m", "vhmodels.vh_checker.runner",
            "--project", self.project,
            "--input", input
        ]

        if hasattr(self, 'model') and self.model:
            cmd.extend(["--model", self.model])

        try:
            result = subprocess.run(cmd, text=True, check=True)
            raw_output = result.stdout.strip()

            match = re.search(r'(\{.*\}|\[.*\])', raw_output, re.DOTALL)
            
            if match:
                json_string = match.group(1)
                return json.loads(json_string)['output']
            else:
                print(f"DEBUG: No JSON found in output. Raw output was:\n{raw_output}", file=os.sys.stderr)
                raise ValueError("Subprocess output contained no valid JSON object.")
                    
        except subprocess.CalledProcessError as e:
            print(f"Subprocess Error:\n{e.stderr}", file=os.sys.stderr)
            raise

    def _env_exists(self):
        # A quick way to check if a conda env exists
        result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
        return self.env_name in result.stdout

def load_model(project, model=None):
    if project not in REGISTRY.keys():
        raise ValueError(f"Model '{project}' not found.")

    #model_cls = BaseModel.get_class(project)
    #current_env = os.environ.get("CONDA_DEFAULT_ENV")

    # If already in the right env, return real instance
    # if current_env == model_cls.env_name:
    #     instance = model_cls()
    #     instance.load_model(project, model)
    #     return instance
    
    # Otherwise, return the Proxy
    env_name = 'vhmodels-'+project
    return ModelProxy(project, env_name, model)
