from vhmodels.vh_checker.base import BaseModel
from vhmodels.models.registry import REGISTRY
from vhmodels.models.source_resolver import SourceResolver

from transformers import T5EncoderModel, T5Tokenizer
from transformers import AlbertModel, AlbertTokenizer
from transformers import BertModel, BertTokenizer
from transformers import (
    ElectraTokenizer,
    ElectraForPreTraining,
    ElectraForMaskedLM,
    ElectraModel,
)
from transformers import XLNetModel, XLNetTokenizer

import torch
import re


class ProtTrans(BaseModel):
    PROJECT = "prottrans"

    # prot_electra_bfd needs generator/discriminator-specific post-processing
    # in embed() even though it shares the encoder-output shape of the other
    # Electra variants; keep the set alongside the variants it identifies.
    _ELECTRA_VARIANTS = frozenset(
        {
            "prot_electra_generator_bfd",
            "prot_electra_discriminator_bfd",
            "prot_electra_bfd",
        }
    )

    def __init__(self):
        self.model_specs = {
            "prot_t5_xl_uniref50": (T5Tokenizer, T5EncoderModel),
            "prot_t5_xxl_uniref50": (T5Tokenizer, T5EncoderModel),
            "prot_t5_xl_bfd": (T5Tokenizer, T5EncoderModel),
            "prot_bert_bfd": (BertTokenizer, BertModel),
            "prot_bert": (BertTokenizer, BertModel),
            "prot_albert": (AlbertTokenizer, AlbertModel),
            "prot_xlnet": (XLNetTokenizer, XLNetModel),
            "prot_electra_generator_bfd": (ElectraTokenizer, ElectraForMaskedLM),
            "prot_electra_discriminator_bfd": (ElectraTokenizer, ElectraForPreTraining),
            "prot_electra_bfd": (ElectraTokenizer, ElectraModel),
        }

        self.tokenizer = None
        self.model = None
        self.device = None
        self.variant = None

    def load_model(self, model=None, **kwargs):
        """
        Downloads and loads the artifacts for the specified model in the ProtTrans HF repository. Possible options are:
        - prot_bert_bfd
        - prot_t5_xl_uniref50
        - prot_t5_xxl_bfd
        - prot_t5_xxl_uniref50
        - prot_xlnet
        - prot_electra_generator_bfd
        - prot_electra_discriminator_bfd
        - prot_electra_bfd
        - prot_albert
        - prot_t5_xl_bfd

        For more information, check (link to the HF repo).

        Parameters
        --------
        model: str
            Name of the model

        device: str
            Device to work on

        Returns
        -------
        None
        """
        self.device = torch.device(
            kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        if model not in self.model_specs:
            raise ValueError(
                f"Unsupported model '{model}'. "
                f"Supported models: {', '.join(self.model_specs.keys())}"
            )
        self.variant = model

        # prot_electra_bfd's tokenizer/weights sources point at the
        # generator/discriminator repos respectively -- see
        # manifests/prot_electra_bfd.json. Every other variant's sources
        # resolve to the same "virtual-human-chc/{variant}" repo, letting
        # transformers.from_pretrained() do the rest.
        manifest = REGISTRY.resolve(self.PROJECT, model)
        resources = SourceResolver().resolve(manifest.sources, manifest.model_dir)

        tokenizer_cls, model_cls = self.model_specs[model]
        self.tokenizer = tokenizer_cls.from_pretrained(
            resources["tokenizer"].repo_id, do_lower_case=False
        )
        self.model = model_cls.from_pretrained(resources["weights"].repo_id)

        self.model = self.model.to(self.device)

        # TODO: Should this code snippet be added?
        # if self.device.type == "cpu":
        #     self.model = self.model.to(torch.float32)

        self.model.eval()

    def _preprocess(self, input):
        """
        Expects input like "MKVILLLLAVVAFGHALCRV".

        Example input: [“PRTEINO”, “SEQWENCE”] or ["AETCZAO","SKTZP"]
        """
        # TODO: Should preprocess handle file and folder as input?
        # TODO: Should preprocess handle spaced input, e.g. P R T E I N O?
        return [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in input]

    def embed(self, input, **kwargs):
        """
        Creates the embeddings for the input data. The function expects the model to be loaded already.

        The input should be already preprocessed. See function "_preprocess".

        The function removes the padding and special tokens before returning to result.

        To get per protein embedding:
        - emb_0.mean(dim=0)

        Parameters
        ----------
        inputs: numpy.ndarray
            Protein sequences to be encoded

        Returns
        ----------
        dict
            A dictionary containing:
            - 'output': list of lists
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        input = self._preprocess(input)

        ids = self.tokenizer.batch_encode_plus(
            input, add_special_tokens=True, padding="longest"
        )
        input_ids = torch.tensor(ids["input_ids"]).to(self.device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(self.device)

        # generate embeddings
        with torch.no_grad():
            embedding_repr = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )

        if self.variant in self._ELECTRA_VARIANTS:
            embedding = embedding_repr[0].cpu().numpy()

            # Remove padding (\<pad\>) and special tokens (\</s\>) that is added by the model
            # TODO: Should the function remove the padding and the special tokens or this should be an option?
            features = []
            for seq_num in range(len(embedding)):
                seq_len = (attention_mask[seq_num] == 1).sum()
                seq_emd = embedding[seq_num][: seq_len - 1]
                features.append(seq_emd.tolist())
        else:
            embedding = embedding_repr.last_hidden_state.cpu().numpy().tolist()

            # Remove padding (\<pad\>) and special tokens (\</s\>) that is added by the model
            # TODO: Should the function remove the padding and the special tokens or this should be an option?
            features = []
            for seq_num in range(len(embedding)):
                seq_len = (attention_mask[seq_num] == 1).sum()
                seq_emd = embedding[seq_num][: seq_len - 1]
                features.append(seq_emd)

        return {"output": features[0]}

    def predict(self, input, **kwargs):
        raise NotImplementedError("ProtTrans does not support predict().")

    def generate(self, input, **kwargs):
        pass


if __name__ == "__main__":
    model = ProtTrans()
    model.load_model("prot_electra_discriminator_bfd")
    result = model.embed(input=["PRTEINO", "SEQWENCE"])

    print(result)
