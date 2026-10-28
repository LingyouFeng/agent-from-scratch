import os

from dotenv import load_dotenv

from ai.model import Model, Provider


def doubao_provider() -> Provider:
    load_dotenv()
    return Provider(
        id="doubao",
        name="doubao",
        base_url=os.getenv("DOUBAO_BASE_URL"),
        api_key=os.getenv("DOUBAO_API_KEY"),
        models=[
            Model(
                id="doubao-seed-2-0-mini-260428",
                name="doubao-seed-2.0-mini",
                provider="doubao",
                api="openai-completions"
            ),
            Model(
                id="doubao-seed-2-0-lite-260428",
                name="doubao-seed-2.0-lite",
                provider="doubao",
                api="openai-completions"
            ),
            Model(
                id="doubao-seed-2-0-pro-260215",
                name="doubao-seed-2.0-pro",
                provider="doubao",
                api="openai-completions"
            ),
        ]
    )
