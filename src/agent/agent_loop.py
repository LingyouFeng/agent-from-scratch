import time

from src.agent.tools.builtin import get_builtin_tools
from src.ai.api.all import get_stream_by_api
from src.ai.model import Model
from src.ai.model import StreamOptions
from src.ai.type import Context, TextContent, ToolResultMessage, AssistantMessage, ToolCall
from util.time_util import get_timestamp_ms

registry = get_builtin_tools()


def run_loop(model: Model, context: Context, stream_options: StreamOptions):
    for _ in range(3):  # 最多循环3次，防止无限循环
        # print("="*100)
        # print(context)
        # print("="*100)
        stream = get_stream_by_api(model.api)
        if stream is None:
            raise ValueError("该模型找不到对应的api协议")

        print("\033[36m Assistant >> \033[0m", end="")

        has_tool_call = False
        assistant_text_content = ""
        tool_calls: list[AssistantMessage] = []
        tool_results: list[ToolResultMessage] = []

        for message in stream(model, context, stream_options):
            for content in message.content:
                if content.type == "text":
                    assistant_text_content += content.text
                    print(content.text, end="", flush=True)

                elif content.type == "thinking":
                    print("thinking...")

                elif content.type == "toolCall":
                    has_tool_call = True
                    tool_calls.append(message)

                    tool_result = registry.execute(content.name, content.arguments)
                    print(f"\033[36m [Tool|{content.name}]:{tool_result} \033[0m")

                    tool_results.append(ToolResultMessage(
                        role="toolResult",
                        tool_call_id=content.id,
                        tool_name=content.name,
                        content=[TextContent(type="text", text=tool_result)],
                        timestamp=int(time.time() * 1000),
                    ))

        context.messages.append(AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text="")],
            timestamp=get_timestamp_ms())
        )
        context.messages.extend(tool_calls)
        context.messages.extend(tool_results)

        print()
        if not has_tool_call:
            break
