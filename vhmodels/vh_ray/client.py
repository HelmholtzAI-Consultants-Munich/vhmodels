"""Host-side handle for a model running as a Ray actor in its own venv or image."""

import os
import sys

from vhmodels.vh_checker.factory import _unwrap_model_result
from vhmodels.vh_ray.actor import ModelActor
from vhmodels.vh_ray.envs import image_runtime_env, model_runtime_env


def _driver_paths():
    """Return the directories Ray prepends to every worker's sys.path."""
    paths = [os.getcwd()]
    if sys.argv and sys.argv[0]:
        paths.append(os.path.dirname(os.path.realpath(sys.argv[0])))
    return paths


def _ray_binds():
    """Return Ray's session dir and package dir, which a worker in an image needs."""
    import ray

    session_dir = ray._private.worker._global_node.get_session_dir_path()
    return [session_dir, os.path.dirname(ray.__file__)]


def _actor_options(project, load_kwargs, ray_options=None, image_path=None):
    """Run the actor in the model's venv or image, on a GPU for CUDA devices."""
    gpu = str(load_kwargs.get("device", "auto")).startswith("cuda")
    options = {"num_gpus": 1 if gpu else 0, **(ray_options or {})}
    if image_path:
        gpu = options["num_gpus"] > 0
        runtime_env = image_runtime_env(image_path, _ray_binds(), gpu)
    else:
        runtime_env = model_runtime_env(project)
    return {"runtime_env": runtime_env, **options}


class RayModel:
    """``ModelProxy``-compatible handle to a model running as a Ray actor."""

    def __init__(
        self,
        project,
        model,
        resources,
        load_kwargs=None,
        ray_options=None,
        timeout=None,
        image_path=None,
    ):
        import ray

        if not ray.is_initialized():
            ray.init()
        self.project = project
        self.model = model
        self.timeout = timeout
        load_kwargs = load_kwargs or {}
        self.actor = (
            ray.remote(ModelActor)
            .options(**_actor_options(project, load_kwargs, ray_options, image_path))
            .remote(project, model, resources, load_kwargs, _driver_paths())
        )

    def _call(self, method, *args, **kwargs):
        import ray

        if self.actor is None:
            raise RuntimeError("This model has been closed.")
        ref = getattr(self.actor, method).remote(*args, **kwargs)
        return ray.get(ref, timeout=self.timeout)

    def info(self):
        """Wait for the model to load and describe the actor's process."""
        return self._call("info")

    def embed(self, input, **kwargs):
        return _unwrap_model_result(
            self._call("embed", input, cwd=os.getcwd(), **kwargs)
        )

    def predict(self, input, embedding=None, **kwargs):
        return _unwrap_model_result(
            self._call("predict", input, embedding=embedding, cwd=os.getcwd(), **kwargs)
        )

    def close(self):
        """Release the actor and its model."""
        import ray

        actor, self.actor = self.actor, None
        if actor is not None and ray.is_initialized():
            ray.kill(actor)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
