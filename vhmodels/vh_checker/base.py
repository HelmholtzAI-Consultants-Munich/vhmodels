from abc import ABC, abstractmethod

class BaseModel(ABC):
    _registry = {}

    def __init_subclass__(cls, project=None, description=None, link=None, env_name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        # Only register concrete classes that provide a 'name'
        if project:
            cls.project = project
            # Consistent prefixing helps avoid collisions with other conda envs
            cls.env_name = env_name or f"vhmodels-{project}"
            cls.description = description or "No description provided."
            cls.link = link
            BaseModel._registry[project] = cls

    @classmethod
    def list_available_models(cls):
        """Returns metadata for all discovered models."""
        return [
            {"id": k, "project": v.project, "desc": v.description, "link": v.link} 
            for k, v in cls._registry.items()
        ]

    @abstractmethod
    def load_model(self, model, **kwargs):
        pass
    
    @abstractmethod
    def transform(self, data, **kwargs):
        pass
    
