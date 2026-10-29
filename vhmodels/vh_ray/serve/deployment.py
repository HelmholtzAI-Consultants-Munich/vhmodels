"""Ray Serve app: one model as an HTTP deployment in its own image or venv."""

from fastapi import FastAPI
from ray import serve

from vhmodels.vh_checker.factory import resolve_ray_model
from vhmodels.vh_ray.actor import ModelActor
from vhmodels.vh_ray.client import _actor_options

api = FastAPI()


# One request at a time per replica, like a plain model actor.
@serve.deployment(max_ongoing_requests=1)
@serve.ingress(api)
class ModelDeployment(ModelActor):
    @api.post("/embed")
    def http_embed(self, body: dict):
        return self.embed(**body)

    @api.post("/predict")
    def http_predict(self, body: dict):
        return self.predict(**body)

    @api.get("/info")
    def http_info(self):
        return self.info()


def build_app(args):
    """Build one app from Serve config ``args``: ``project``, ``model``, load kwargs."""
    load_kwargs = dict(args)
    project = load_kwargs.pop("project")
    model = load_kwargs.pop("model", None)
    variant, resources, image_path = resolve_ray_model(project, model)
    options = _actor_options(project, load_kwargs, image_path=image_path)
    return ModelDeployment.options(ray_actor_options=options).bind(
        project, variant, resources, load_kwargs
    )
