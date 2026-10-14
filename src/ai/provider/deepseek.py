import os

from src.ai.model import Model, Provider


def deepseek_provider() -> Provider:
    return Provider(
        id="deepseek",
        name="deepseek",
        base_url="https://api.deepseek.com",
        api_key=os.getenv("DEEPSEEK_API_KEY") or "sk-c8c146f4b44148859adf06dfd0d09720",
        models=[
            Model(
                id="deepseek-v4-flash",
                name="deepseek-v4-flash",
                provider="deepseek",
                api="openai-completions"
            ),
            Model(
                id="deepseek-v4-pro",
                name="deepseek-v4-pro",
                provider="deepseek",
                api="openai-completions"
            ),
        ]
    )
