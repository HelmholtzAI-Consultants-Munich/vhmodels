# The file contains the necessary to run the function 'embed' for the given model.
# It runs *inside* the model's isolated environment as a subprocess. The parent
# (ModelProxy) sends the input as a JSON document on stdin and reads the result
# framed between RESULT_MARKERs on stdout.
from vhmodels.vh_checker.base import BaseModel
from vhmodels.vh_checker.protocol import RESULT_MARKER

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="ID of the project to run")
    parser.add_argument("--model", required=False, help="Exact model in the project")
    args = parser.parse_args()

    model_cls = BaseModel.get_class(args.project)

    try:
        # Read the input document from stdin. Using stdin (rather than a CLI
        # argument) avoids the OS argument-length limit on large inputs.
        payload = sys.stdin.read()
        input_data = json.loads(payload) if payload.strip() else None

        # Instantiate and load weights (this happens inside the sub-env).
        instance = model_cls()
        instance.load_model(args.model)

        # Execute embedding. The model returns its own envelope, e.g.
        # {"output": [...]}; we emit it verbatim and let the parent unwrap.
        result = instance.embed(input_data)

        # Frame the result so the parent can extract it regardless of any other
        # output libraries may have written to stdout.
        sys.stdout.write(RESULT_MARKER + json.dumps(result) + RESULT_MARKER + "\n")
        sys.stdout.flush()

    except Exception as e:
        # Errors go to stderr so the parent can report them; a non-zero exit
        # tells the parent the run failed.
        print(f"Internal Model Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
