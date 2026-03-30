#from .base import REGISTRY
from vhmodels.registry import MODEL_REGISTRY
from vhmodels.vh_checker.base import BaseModel

import os
import json
import subprocess
from pathlib import Path
import re
import pandas as pd

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
env = os.environ.copy()
env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

class ModelProxy:
    def __init__(self, project, env_name, model=None, runtime='conda'):
        self.project = project
        self.model = model
        self.runtime = runtime
        self.env_name = env_name
    
    def _env_exists(self):
        # A quick way to check if a conda env exists
        result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
        return self.env_name in result.stdout

    def transform(self, input, runtime='conda', **kwargs):
        if not self._env_exists():
            raise RuntimeError(
                f"The environment '{self.env_name}' does not exist. "
                f"Please run 'vh-checker create-env {self.project}' first."
            )
        
        if isinstance(input, str) and os.path.exists(input):
        # It's a file path; pass the path string directly
            input = input
        elif isinstance(input, pd.DataFrame):
            
            input = json.dumps(input.to_dict(orient="records"))
        else:
            # It's a list or dictionary; serialize to JSON string
            input = json.dumps(input)
        
        # Run using conda env
        if self.runtime == 'conda':
            # We point to the runner inside vh_checker
            cmd = [
                "conda", "run", "-n", self.env_name,
                "python", "-m", "vhmodels.vh_checker.runner",
                "--project", self.project,
                "--input", input
            ]

        elif self.runtime == 'docker':
            cmd = [
                "docker", "run", "--rm",
                #"-v", f"{os.getcwd()}:/app",
                f"vhmodels-{self.project}",
                "micromamba", "run", "-n", f"vhmodels-{self.project}",
                "python", "-m", "vhmodels.vh_checker.runner",
                "--project", self.project,
                "--input", input
            ]
        elif self.runtime == 'singularity':
            raise NotImplementedError("Singularity not implemented yet!")
        else:
            raise RuntimeError(
                f"Unsupported runtime: '{runtime}'"
                f"Supported environments: conda, docker, singularity."
            )
        
        if hasattr(self, 'model') and self.model:
                cmd.extend(["--model", self.model])

        try:
            result = subprocess.run(cmd, 
                                    env=env, 
                                    capture_output=True, 
                                    text=True, 
                                    encoding="utf-8",  # <-- add this
                                    errors="replace",  # <-- optional, replaces undecodable bytes with �
                                    check=True)
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

def load_model(project, model=None, runtime='conda'):
    if project not in list(MODEL_REGISTRY.keys()):
        raise ValueError(f"Model '{project}' not found.")

    # model = MODEL_REGISTRY[project]
    # current_env = os.environ.get("CONDA_DEFAULT_ENV")

    # # If already in the right env, return real instance
    # if current_env == model['conda_env']:
    #     instance = BaseModel.get_class(project)
    #     instance.load_model(project, model)
    #     return instance
    
    # Otherwise, return the Proxy
    return ModelProxy(project=project, env_name='vhmodels-'+project, model=model, runtime=runtime)
