import random
from dataclasses import dataclass, field
from typing import Any, Union

import llm
from llm_openrouter import OpenRouterChat


@dataclass
class Model:
    name: str
    llm: Union[llm.Model, OpenRouterChat]
    settings: dict[str, bool | int | str | float] = field(default_factory=dict)
    other: dict[str, Any] = field(default_factory=dict)


def build_model_remote_openrouter(
    model_name: str,
    api_key: str | None = None,
    provider: dict[str, str] | None = None,
    **kwargs,
) -> Model:
    model = OpenRouterChat(
        model_id=f"openrouter/{model_name}",
        key=api_key,
        model_name=model_name,
        api_base="https://openrouter.ai/api/v1",
        headers={"HTTP-Referer": "https://llm.datasette.io/", "X-Title": "LLM"},
    )
    settings = {}
    if "settings" in kwargs:
        settings.update(kwargs["settings"])
    if provider is not None:
        settings["provider"] = provider
    return Model(name=model_name, llm=model, settings=settings)


def normalize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_").replace(" ", "_").lower()


class ModelPool:
    def __init__(self):
        self.models = {}

    def add_model(self, model: Model):
        self.models[model.name] = model

    def get_model(self, model_name: str) -> Model | None:
        return self.models.get(model_name)

    def remove_model(self, model_name: str):
        self.models.pop(model_name, None)

    def get_model_random(self, seed: int | None = None) -> Model:
        r = random.Random(seed)
        return r.choice(list(self.models.values()))
