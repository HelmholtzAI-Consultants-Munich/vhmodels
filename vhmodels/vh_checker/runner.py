from vhmodels.vh_checker.base import BaseModel

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent # Goes up to the folder containing 'vhmodels'

# 2. Inject it into the system path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="ID of the project to run")
    parser.add_argument("--model", required=True, help="Exact model in the project")
    parser.add_argument("--input", required=True, help="Input data as a JSON string")
    args = parser.parse_args()

    model_cls = BaseModel.get_class(args.project)

    try:
        # 2. Instantiate and call load_model (this happens inside the sub-env)
        instance = model_cls()
        instance.load_model(args.model)

        # 3. Parse data and execute transformation
        result = instance.transform(args.input)

        # 4. Output the result to STDOUT as JSON
        # The Proxy catches this output.if isinstance(result, pd.DataFrame):
        result['output'].to_dict(orient="records")
        
        print(json.dumps(result))

    except Exception as e:
        # Any errors here are sent to STDERR so the Proxy can report them
        print(f"Internal Model Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()