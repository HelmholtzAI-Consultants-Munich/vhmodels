"""Ray Serve app: one model as an HTTP deployment in its own image or venv."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from ray import serve

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
