import os

from dotenv import load_dotenv

from ai.model import Model, Provider


def deepseek_provider() -> Provider:
    load_dotenv()
    return Provider(
        id="deepseek",
        name="deepseek",
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        models=[
            Model(
                id="cmhi/deepseek-v4-flash",
                name="deepseek-v4-flash",
                provider="deepseek",
                api="openai-completions"
            ),
            Model(
                id="cmhi/deepseek-v4.1-flash",
                name="deepseek-v4.1-flash",
                provider="deepseek",
                api="openai-completions"
            ),
        ]
    )
