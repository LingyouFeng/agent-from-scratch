from agent import Model, Provider, ProviderId
from agent import deepseek_provder

_PROVIDER_MAP: dict[ProviderId, Provider] = {
    "deepseek": deepseek_provder(),
}


def get_provider_by_id(id: str) -> Provider | None:
    return _PROVIDER_MAP.get(id)


def get_model_by_id(id: str) -> Model | None:
    for provider in _PROVIDER_MAP.values():
        for model in provider.models:
            if model.id == id:
                return model
    return None


def get_builtin_models() -> dict[ProviderId, list[str]]:
    results = {}
    for provider_id, provider in _PROVIDER_MAP.items():
        results[provider_id] = [m.id for m in provider.models]
    return results
