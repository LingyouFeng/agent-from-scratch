from __future__ import annotations

from ai.model import Model, Provider, ProviderId
from ai.provider.deepseek import deepseek_provider
from ai.provider.doubao import doubao_provider


class ProviderRegistry:
    """供应商/模型注册中心：统一管理内置供应商与模型查询。"""

    def __init__(self, providers: list[Provider] | None = None) -> None:
        self._providers: dict[ProviderId, Provider] = {}
        for provider in providers or []:
            self.register(provider)

    def register(self, provider: Provider) -> None:
        """注册一个供应商（同 id 会覆盖旧值）。"""
        self._providers[provider.id] = provider

    def get_provider_by_id(self, id: str) -> Provider | None:
        return self._providers.get(id)

    def get_model_by_id(self, id: str) -> Model | None:
        for provider in self._providers.values():
            for model in provider.models:
                if model.id == id:
                    return model
        return None

    def get_builtin_models(self) -> list[Model]:
        """获取所有内置模型。"""
        return [
            model
            for provider in self._providers.values()
            for model in provider.models
        ]


# 默认注册中心：内置供应商在此登记
default_registry = ProviderRegistry(
    [
        deepseek_provider(),
        doubao_provider(),
    ]
)
