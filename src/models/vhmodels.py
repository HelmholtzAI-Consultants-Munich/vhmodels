class VHModel:

    _registry = {}

    capabilities = set()  # e.g. {"encode", "predict", "generate"}

    def __init_subclass__(cls, *, name=None, description=None, link=None, **kwargs):
        super().__init_subclass__(**kwargs)

        VHModel._registry[name or cls.__name__] = {
            "class": cls,
            "description": description or "No description provided",
            "capabilities": getattr(cls, "capabilities", set()),
            "link": link or "No link provided"
        }

    @classmethod
    def list_sources(cls):
        return {
            name: {
                "description": meta["description"],
                "capabilities": sorted(meta["capabilities"]),
                "link": meta["link"]
            }
            for name, meta in cls._registry.items()
        }

    def load_model(self, repo_id=None, **kwargs):
        raise NotImplementedError("This model does not support loading.")

    def encode(self, inputs, **kwargs):
        raise NotImplementedError("This model does not support encoding.")

    def predict(self, inputs, **kwargs):
        raise NotImplementedError("This model does not support prediction.")

    def generate(self, num_samples, **kwargs):
        raise NotImplementedError("This model does not support generation.")
    