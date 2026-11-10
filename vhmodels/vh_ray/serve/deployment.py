"""Ray Serve app: one model as an HTTP deployment in its own image or venv."""

import json
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from ray import serve
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from vhmodels.vh_checker.factory import _get_registry, check_input, resolve_ray_model
from vhmodels.vh_ray.actor import ModelActor
from vhmodels.vh_ray.client import _actor_options

api = FastAPI()


class ModelRequest(BaseModel):
    """``input`` plus the keyword arguments of ``embed``/``predict``."""

    model_config = ConfigDict(extra="allow")

    # Sequences or SMILES, a file or folder path, or named paths, per model.
    input: str | list[str] | dict[str, str]


def call_model(method, request, model_input=None):
    """Call ``method`` with the request; return model input errors as HTTP errors."""
    kwargs = request.model_dump()
    try:
        check_input(model_input, kwargs["input"])
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    try:
        return method(**kwargs)
    except (ValueError, TypeError, FileNotFoundError) as error:
        raise HTTPException(400, f"{type(error).__name__}: {error}") from error
    except NotImplementedError as error:
        raise HTTPException(501, f"{type(error).__name__}: {error}") from error


def save_uploads(form, model_input, directory):
    """Write the uploaded files to ``directory``; return their paths as ``input``."""
    if model_input is None or model_input.type == "sequences":
        raise HTTPException(
            415, "This model takes JSON input; POST to /embed or /predict."
        )
    paths = []
    for index, (field, upload) in enumerate(form.multi_items()):
        if not isinstance(upload, UploadFile):
            continue
        # The file name only: models read its suffix, and it stays in ``directory``.
        name = Path(upload.filename or "").name
        path = Path(directory, str(index), name if name not in ("", "..") else "input")
        path.parent.mkdir()
        with path.open("wb") as file:
            shutil.copyfileobj(upload.file, file)
        paths.append((field, str(path)))
    if model_input.type == "named_files":
        return dict(paths)
    inputs = [path for field, path in paths if field == "input"]
    if not inputs:
        raise HTTPException(422, "No files uploaded in the 'input' field.")
    return inputs[0] if model_input.type == "file" and len(inputs) == 1 else inputs


def call_model_with_files(method, form, model_input):
    """Call ``method`` on the uploaded files, saved for this request only."""
    try:
        kwargs = json.loads(form.get("kwargs") or "{}")
    except json.JSONDecodeError as error:
        raise HTTPException(422, f"kwargs is not valid JSON: {error}") from error
    if not isinstance(kwargs, dict) or "input" in kwargs:
        raise HTTPException(422, "kwargs must be a JSON object without 'input'.")
    with tempfile.TemporaryDirectory(prefix="vhmodels-upload-") as directory:
        request = ModelRequest(
            input=save_uploads(form, model_input, directory), **kwargs
        )
        return call_model(method, request, model_input)


# One request at a time per replica, like a plain model actor.
@serve.deployment(max_ongoing_requests=1)
@serve.ingress(api)
class ModelDeployment(ModelActor):
    def __init__(self, project, model, resources, load_kwargs=None, model_input=None):
        super().__init__(project, model, resources, load_kwargs)
        self.model_input = model_input

    @api.post("/embed")
    def http_embed(self, request: ModelRequest):
        return call_model(self.embed, request, self.model_input)

    @api.post("/predict")
    def http_predict(self, request: ModelRequest):
        return call_model(self.predict, request, self.model_input)

    # File contents, for clients that don't share the replica's disk.
    @api.post("/embed/files")
    async def http_embed_files(self, request: Request):
        async with request.form() as form:
            return await run_in_threadpool(
                call_model_with_files, self.embed, form, self.model_input
            )

    @api.post("/predict/files")
    async def http_predict_files(self, request: Request):
        async with request.form() as form:
            return await run_in_threadpool(
                call_model_with_files, self.predict, form, self.model_input
            )

    @api.get("/info")
    def http_info(self):
        model_input = self.model_input and self.model_input.model_dump(exclude_none=True)
        return {**self.info(), "input": model_input}


def build_app(args):
    """Build one app from Serve ``args``: project, model, ray_options, load kwargs."""
    load_kwargs = dict(args)
    project = load_kwargs.pop("project")
    model = load_kwargs.pop("model", None)
    ray_options = load_kwargs.pop("ray_options", None)
    variant, resources, image_path = resolve_ray_model(project, model)
    options = _actor_options(project, load_kwargs, ray_options, image_path)
    model_input = _get_registry().get_model(project).implementation.input
    return ModelDeployment.options(ray_actor_options=options).bind(
        project, variant, resources, load_kwargs, model_input
    )
