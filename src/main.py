import sys
import time

from agent.agent_loop import run_loop
from src.agent.tools.builtin import get_builtin_tools
from src.ai.provider.all import get_builtin_models, get_model_by_id
from src.ai.type import Context, TextContent, UserMessage

if __name__ == "__main__":

    print("内置模型：")
    builtin_models = get_builtin_models()
    for provider, models in builtin_models.items():
        print(f"[{provider}]: {', '.join(m for m in models)}")

    # input_model = input("请输入你的模型：")
    input_model = "deepseek-v4-flash"
    model = get_model_by_id(input_model)
    if model is None:
        print("不存在该模型")
        sys.exit(1)

    systemPrompt = "你是一个ai助手"
    messages = []
    registry = get_builtin_tools()

    while True:
        try:
            input_message = input("\033[36m User >> \033[0m")
            user_msg = UserMessage(
                role="user",
                content=[TextContent("text", input_message)],
                timestamp=int(time.time() * 1000),
            )
            print("\033[36m Assistant >> \033[0m")
            messages.append(user_msg)
            context = Context(systemPrompt, messages, registry.get_all_tools())
            run_loop(model, context)
            print()
        except (EOFError, KeyboardInterrupt):
            print("bye~")
            break
