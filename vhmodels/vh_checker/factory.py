import os
import json
import subprocess
from .base import BaseModel
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent.resolve())

# 2. Setup environment variables for the subprocess
# This tells the Conda environment where to find the 'vhmodels' folder
env = os.environ.copy()
env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

class ModelProxy:
    def __init__(self, project, model, env_name):
        self.project = project
        self.model = model
        self.env_name = env_name
    
    ## Add some point, add here predict function
    ## Add some point, add here generate function (for Hyformer)

    def transform(self, data, **kwargs):
        if not self._env_exists():
            raise RuntimeError(
                f"The environment '{self.env_name}' does not exist. "
                f"Please run 'vh-checker create-env {self.project}' first."
            )
        
        input_data = json.dumps(data)
        
        # We point to the runner inside vh_checker
        cmd = [
            "conda", "run", "-n", self.env_name,
            "python", "-m", "vhmodels.vh_checker.runner",
            "--project", self.project,
            "--data", input_data
        ]

        if hasattr(self, 'model') and self.model:
            cmd.extend(["--model", self.model])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Subprocess Error:\n{e.stderr}", file=os.sys.stderr)
            raise

    def _env_exists(self):
        # A quick way to check if a conda env exists
        result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
        return self.env_name in result.stdout

def load_model(project, model):
    if project not in BaseModel._registry:
        raise ValueError(f"Model '{project}' not found.")

    model_cls = BaseModel._registry[project]
    current_env = os.environ.get("CONDA_DEFAULT_ENV")

    # If already in the right env, return real instance
    if current_env == model_cls.env_name:
        instance = model_cls()
        instance.load_model(project, model)
        return instance
    
    # Otherwise, return the Proxy
    return ModelProxy(project, model, model_cls.env_name)

