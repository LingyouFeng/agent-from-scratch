from ai.api.openai_completion import OpenAICompletionAPI

_API_MAP: dict[str, OpenAICompletionAPI] = {
    "openai-completions": OpenAICompletionAPI(),
}


def get_api_by_api(api: str) -> OpenAICompletionAPI | None:
    return _API_MAP.get(api)
