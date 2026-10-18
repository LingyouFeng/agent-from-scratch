import platform
import sys
import time
from pathlib import Path

from agent.runtime.agent_loop import AgentLoop
from agent.tools.registry import ToolRegistry
from src.ai.model import StreamOptions
from src.ai.provider.all import get_builtin_models, get_model_by_id
from src.ai.type import Context, TextContent, UserMessage

WORKDIR = Path.cwd().resolve()
SYSTEM_PROMPT = (f"You are a cli coding agent at {WORKDIR}. System: {platform.uname().system}. "
                 f"Use tools to solve tasks. Act, don't explain.")

if __name__ == "__main__":

    try:
        import readline

        readline.parse_and_bind("set bind-tty-special-chars off")
        readline.parse_and_bind("set input-meta on")
        readline.parse_and_bind("set output-meta on")
        readline.parse_and_bind("set convert-meta off")
    except ImportError:
        pass

    print("内置模型：")
    builtin_models = get_builtin_models()
    for provider, models in builtin_models.items():
        print(f"[{provider}]: {', '.join(m for m in models)}")

    # input_model = input("请输入你的模型：")
    input_model = "cmhi/deepseek-v4.1-flash"
    model = get_model_by_id(input_model)
    if model is None:
        print("不存在该模型")
        sys.exit(1)

    messages = []
    registry = ToolRegistry()
    tools = registry.get_all_tools()
    agentLoop = AgentLoop(WORKDIR, registry)

    while True:
        try:
            stream_option = StreamOptions()

            input_message = input("\033[93mUser >> \033[0m")
            user_msg = UserMessage(
                role="user",
                content=[TextContent("text", input_message)],
                timestamp=int(time.time() * 1000),
            )
            messages.append(user_msg)
            context = Context(SYSTEM_PROMPT, messages, tools)
            agentLoop.run_loop(model, context, stream_option)
        except (EOFError, KeyboardInterrupt):
            print("bye~")
            break
