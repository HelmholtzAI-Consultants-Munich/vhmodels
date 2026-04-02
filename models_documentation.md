---
description: |
    API documentation for modules: models, models.model_dinobloom, models.model_hyformer, models.model_mole, models.model_prottrans, models.vhmodels.

lang: en

classoption: oneside
geometry: margin=1in
papersize: a4

linkcolor: blue
links-as-notes: true
...



# Module `models` {#models}





## Sub-modules

* [models.model_dinobloom](#models.model_dinobloom)
* [models.model_hyformer](#models.model_hyformer)
* [models.model_mole](#models.model_mole)
* [models.model_prottrans](#models.model_prottrans)
* [models.vhmodels](#models.vhmodels)







# Module `models.model_dinobloom` {#models.model_dinobloom}








## Classes



### Class `DinoBloom` {#models.model_dinobloom.DinoBloom}




>     class DinoBloom






#### Ancestors (in MRO)

* [models.vhmodels.VHModel](#models.vhmodels.VHModel)







#### Methods



##### Method `encode` {#models.model_dinobloom.DinoBloom.encode}




>     def encode(
>         self,
>         inputs,
>         **kwargs
>     )


Creates the embeddings for the input data. The function expects the model to be loaded already.

Transforms the input - resize and normalize.

###### Parameters

**```inputs```** :&ensp;<code>str</code>
:   Path to folder or image

Returns
-------

numpy.ndarray
    Embeddings of the input data


##### Method `generate` {#models.model_dinobloom.DinoBloom.generate}




>     def generate(
>         self,
>         num_samples,
>         **kwargs
>     )





##### Method `load_model` {#models.model_dinobloom.DinoBloom.load_model}




>     def load_model(
>         self,
>         repo=None,
>         **kwargs
>     )


Downloads and loads the specified model version - S, B, L, G - of the DinoBloom models. This includes the following steps:
- Loading “facebookresearch/dinov2”
- Load “pytorch_model_{variant}.bin”

For more information, check (link to HF repo).

###### Parameters

**```model```** :&ensp;<code>str </code>
:   Version of the model

Returns
-------

dinov2.models.vision_transformer.DinoVisionTransformer (inherits from nn.torch.Module)
    The loaded model


##### Method `predict` {#models.model_dinobloom.DinoBloom.predict}




>     def predict(
>         self,
>         inputs,
>         **kwargs
>     )







# Module `models.model_hyformer` {#models.model_hyformer}








## Classes



### Class `Hyformer` {#models.model_hyformer.Hyformer}




>     class Hyformer






#### Ancestors (in MRO)

* [models.vhmodels.VHModel](#models.vhmodels.VHModel)







#### Methods



##### Method `encode` {#models.model_hyformer.Hyformer.encode}




>     def encode(
>         self,
>         inputs,
>         **kwargs
>     )


Creates embeddings for the provided input.
The function expects the tokenizer and the model to be loaded already.

Parameters
----------
inputs : str
    Path to the raw data file containing molecular representations.

Batch_size : int
    Size of the batch

Device: str
    Device used

Returns
-------
numpy.ndarray
    Embeddings of the input data

###### Example

[[0.12989292, -0.04472789, 1.27521825 ... -0.31017503, -2.61905527, -0.26748869]
[ 0.04795801, -0.71846646, 3.47797537 ...  2.37488675, -0.28063831, 1.84492266]
[-0.00499679, 0.72711295, 0.48343059 ... -1.17737067, 0.93289232, 0.32299849]]


##### Method `generate` {#models.model_hyformer.Hyformer.generate}




>     def generate(
>         self,
>         num_samples,
>         **kwargs
>     )





##### Method `load_model` {#models.model_hyformer.Hyformer.load_model}




>     def load_model(
>         self,
>         model=None,
>         **kwargs
>     )


Downloads and loads the artifacts for the specified model. The available models are:
- hyformer_molecules_50M
- hyformer_peptides_34M
- hyformer_peptides_34_MIC
- hyformer_molecules_8M

The function retrieves and prepares following files:
- vocab.txt: vocabulary of the tokenizer
- tokenizer_config.json: configuration of the tokenizer
- model_config.json: configuration of the model
- downstream_config.json: configuration for the downstream prediction task
- ckpt.pt: weights of the model

For more information, check out (link to HF).

Parameters
--------
model: str
    Name of the model

Returns
-------
Type of the model
    The loaded model


##### Method `predict` {#models.model_hyformer.Hyformer.predict}




>     def predict(
>         self,
>         inputs,
>         **kwargs
>     )







# Module `models.model_mole` {#models.model_mole}








## Classes



### Class `MolE` {#models.model_mole.MolE}




>     class MolE






#### Ancestors (in MRO)

* [models.vhmodels.VHModel](#models.vhmodels.VHModel)







#### Methods



##### Method `encode` {#models.model_mole.MolE.encode}




>     def encode(
>         self,
>         inputs,
>         **kwargs
>     )


Creates embeddings for the provided input data.

The model is expected to be already loaded before calling this function.

Parameters
----------
inputs : str
    Path to the raw data file containing molecular representations.
    The file must contain two columns:
    - "chem_name": Name of the molecule
    - "smiles": SMILES representation of the molecule

Returns
-------
pandas.core.frame.DataFrame
    A DataFrame containing the generated embeddings.
    - The index corresponds to "chem_name".
    - The columns correspond to the 1000 embedding dimensions.

###### Example


###### 0          1          2      ...      997        998        999

Halicin        10.230212  10.007570  0.120784  ...  -1.998857 -20.055006  16.564339
Abaucin        26.006750  42.643391  2.415676  ...  -5.441332 -54.594162   7.772403
Diacerein      16.235847  27.619564  1.328622  ...  -4.330855 -43.452499  22.268299
Tannic acid    49.815845 183.626511  3.261163  ... -19.322300-193.864944 146.213165
Elivitegravir  24.466951  49.919296  2.536460  ...  -5.996571 -60.164993  17.515989
Opicapone      16.270607  30.276501  0.966270  ...  -4.108759 -41.224167  18.833054
Ebastine       36.845181  69.056267  4.710568  ...  -8.217525 -82.448318   7.520126


##### Method `generate` {#models.model_mole.MolE.generate}




>     def generate(
>         self,
>         num_samples,
>         **kwargs
>     )





##### Method `load_model` {#models.model_mole.MolE.load_model}




>     def load_model(
>         self,
>         model=None,
>         **kwargs
>     )


Downloads and loads the necessary artifacts for the MolE model from HuggingFace.

The function retrieves and prepares the following files:
- config.yaml: Contains the transformer configuration.
- model.pth: Contains the trained model weights.
- MolE-XGBoost-08.03.2024_14.20.pkl: Contains the XGBoost model used within MolE.

Returns
-------
Type of the model
    The loaded model


##### Method `predict` {#models.model_mole.MolE.predict}




>     def predict(
>         self,
>         inputs,
>         **kwargs
>     )







# Module `models.model_prottrans` {#models.model_prottrans}








## Classes



### Class `ProtTrans` {#models.model_prottrans.ProtTrans}




>     class ProtTrans






#### Ancestors (in MRO)

* [models.vhmodels.VHModel](#models.vhmodels.VHModel)







#### Methods



##### Method `encode` {#models.model_prottrans.ProtTrans.encode}




>     def encode(
>         self,
>         inputs,
>         **kwargs
>     )


Creates the embeddings for the input data. The function expects the model to be loaded already.

Example input: [“PRTEINO”, “SEQWENCE”]

Parameters
----------
inputs: numpy.ndarray
    Protein sequences to be encoded

Returns
----------
numpy.ndarray
    Embeddings of the input data


##### Method `generate` {#models.model_prottrans.ProtTrans.generate}




>     def generate(
>         self,
>         num_samples,
>         **kwargs
>     )





##### Method `load_model` {#models.model_prottrans.ProtTrans.load_model}




>     def load_model(
>         self,
>         model=None,
>         **kwargs
>     )


Downloads and loads the artifacts for the specified model in the ProtTrans HF repository. Possible options are:
- prot_bert_bfd: loads encoder and two prediction models - prot_bert_bfd_membrane and prot_bert_bfd_ss3
- prot_t5_xl_uniref50: loads  encoder
- prot_t5_xxl_bfd: loads encoder
- prot_t5_xxl_uniref50: loads encoder
- prot_xlnet: loads encoder
- prot_electra_generator_bfd: loads encoder
- prot_electra_discriminator_bfd: loads encoder
- prot_albert: loads encoder
- prot_t5_xl_bfd: loads encoder

For more information, check (link to the HF repo).

Parameters
--------
model: str
    Name of the model

Returns
-------

Type of the model
    The loaded model


##### Method `predict` {#models.model_prottrans.ProtTrans.predict}




>     def predict(
>         self,
>         inputs,
>         **kwargs
>     )







# Module `models.vhmodels` {#models.vhmodels}








## Classes



### Class `VHModel` {#models.vhmodels.VHModel}




>     class VHModel







#### Descendants

* [models.model_dinobloom.DinoBloom](#models.model_dinobloom.DinoBloom)
* [models.model_hyformer.Hyformer](#models.model_hyformer.Hyformer)
* [models.model_mole.MolE](#models.model_mole.MolE)
* [models.model_prottrans.ProtTrans](#models.model_prottrans.ProtTrans)



#### Class variables



##### Variable `capabilities` {#models.vhmodels.VHModel.capabilities}




The type of the None singleton.




#### Static methods



##### `Method list_sources` {#models.vhmodels.VHModel.list_sources}




>     def list_sources()






#### Methods



##### Method `encode` {#models.vhmodels.VHModel.encode}




>     def encode(
>         self,
>         inputs,
>         **kwargs
>     )





##### Method `generate` {#models.vhmodels.VHModel.generate}




>     def generate(
>         self,
>         num_samples,
>         **kwargs
>     )





##### Method `load_model` {#models.vhmodels.VHModel.load_model}




>     def load_model(
>         self,
>         model=None,
>         **kwargs
>     )





##### Method `predict` {#models.vhmodels.VHModel.predict}




>     def predict(
>         self,
>         inputs,
>         **kwargs
>     )





-----
Generated by *pdoc* 0.11.6 (<https://pdoc3.github.io>).
