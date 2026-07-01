# Adding models

Thank you for taking the time to add your model to `vhmodels`! The process consists of the following steps:

1. upload your weights to Huggingface
2. add a description of your model
3. create a new model entry in `vhmodels`
4. add example data
5. add your model to the quick start instructions

## Step 1: Huggingface upload

Upload your model weights to Huggingface. Use a format that can be easily loaded when your model is used.
You can find a first overview about different formats on the [Huggingface Blog](https://huggingface.co/blog/ngxson/common-ai-model-formats).

## Step 2: Model description

In order for others to easily understand how your model can be used, you need a good description. Please include the following sections:

- Short description: overview about the model
- Model versions (if applicable): different available versions
- Long description: in-depth description of your model
- Metadata
    - Input
        - Description: short description of the input data
        - Input format:
            - Shape: shape of the data
            - Data type (for non-tabular data): type of the data
            - Columns (for tabular data): description of column names with data types
        - Example input file: path to an example input file
    - Model
        - Modality: on which modality does the model work on, e.g. chemical compounds or images
        - Scale: on which scale does the model work on, e.g. one image, one chemical compound, combination of chemical compound and bacterial strain
        - Description: concise description what the model does with the input data
        - Trainig data (if applicable): data used for training
        - Publication (if applicable): link to publication
    - Output
        - Description: short description of the output data
        - Output format: e.g. table or tensor
            - Shape: shape of the data
            - Data type (for non-tabular data): type of the data
            - Columns (for tabular data): description of column names with data types
- Installation: instructions how to install your model, the easiest is to provide an `environment.yml` file to create a conda environment. You'll need this for the next step anyway.
- Example: example code how to perform inference with your model
- References: references cited throughout the description/model card
- Copyright: information about the copyright

## Step 3: add the model to `vhmodels`

Create a new folder for your model in `vhmodels/models`. The folder needs to contain the following files:

- `config.json`
```
{
    "name": "<your_model>",
    "supported_platforms": ["linux-x86_64"], # alternatively also "macos-arm64"
    "environment_files": {
      "linux-x86_64": "environment.linux-x86_64.yml" # ,
      # "macos-arm64": "environment.macos-arm64.yml"
    },
    "conda_env": "vhmodels-<your_model>",
    "class_path": "<Your_model>.model.<Your_model>",
    "description": "short description of <your_model>",
    "link": "link to Huggingface repository"
}
```

- `environment.linux-x86_64.yml`

```
name: vhmodels-<your_model>
channels:
  - conda-forge
  - nodefaults
dependencies:
  - # your dependencies here, you can also add pip dependencies
```

You can only use conda-forge channels, as official anaconda channels are blocked on the HPC.
When adding `torch` via pip, you can specify the CUDA versions.

Additionally, you can also provide an `environment.macos-arm64.yml` file.

- `model.py`

Here, the model loading and calling is implemented. Please have a look at the already implemented
models as examples. It might be necessary to adapt your existing model code so that it can be
easily used in `vhmodels`.

```
from vhmodels.vh_checker.base import BaseModel


class YourModel(BaseModel):
    def __init__(self):
        self.model = None

    def load_model(self, model=None, **kwargs):
        """
        Downloads and loads the necessary artifacts for YourModel model from HuggingFace.

        Returns
        -------
        None
        """

        # implement the downloading of the model weights from Huggingface
        # and load the model to the correct device
        # here, you can also add your tokenizer if necessary

    def embed(self, input, **kwargs):
        """
        Creates embeddings for the provided input data.

        The model is expected to be already loaded before calling this function.

        Parameters
        ----------
        inputs :
            Either the data itself or a path to the data file.

        Returns
        -------
        dict
            A dictionary containing:
            - 'output': list of lists
        """

        # perform the necessary preprocessing and call the model with
        # the input data to generate embeddings
        # make sure that the input data is on the correct device and
        # the output on the CPU again

        return {"output": embeddings}

    # predict and generate functionality is not implemented yet
    def predict(self, input, **kwargs):
        pass

    def generate(self, input, **kwargs):
        pass


if __name__ == "__main__":
    model = YourModel()
    model.load_model()
    result = model.embed("example_data/YourModel/example.data")
    print(result)
```

## Step 4: add example data

If needed, add a small example data file to `example_data/<your_model>` that can
be used in the inference examples.

## Step 5: add quick start instructions

In the README, add your model to the quick start instructions (add all required arguments for the model loading):

```python
import vhmodels

model = vhmodels.load_model(project='<your_model>')
results = model.embed(input='example_data/<your_model>/example.data')
print(results)
```
