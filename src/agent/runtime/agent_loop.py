import time

from agent.hooks.manager import HookManager
from agent.runtime.tool_executor import ToolExecutor
from agent.tools.permission import PermissionPolicy
from agent.tools.registry import ToolRegistry
from src.ai.api.all import get_stream_by_api
from src.ai.model import Model
from src.ai.model import StreamOptions
from src.ai.type import Context, TextContent, ToolResultMessage, AssistantMessage, ToolCall
from util.time_util import get_timestamp_ms


class AgentLoop:

    def __init__(self, tool_registry: ToolRegistry):
        self._tool_executor = ToolExecutor(
            tool_registry,
            PermissionPolicy(),
            HookManager(),
        )

    def run_loop(self, model: Model, context: Context, stream_options: StreamOptions):
        for _ in range(10):  # 最多循环10次，防止无限循环
            # print("="*100)
            # print(context)
            # print("="*100)
            stream = get_stream_by_api(model.api)
            if stream is None:
                raise ValueError("该模型找不到对应的api协议")

            print("\033[36mAssistant >> \033[0m", end="")

            has_tool_call = False
            text_content = ""
            tool_calls: list[ToolCall] = []
            tool_results: list[ToolResultMessage] = []

            for message in stream(model, context, stream_options):
                for content in message.content:
                    if content.type == "text":
                        text_content += content.text
                        print(f"\033[36m{content.text}\033[0m", end="", flush=True)

                    elif content.type == "thinking":
                        print("thinking...")

                    elif content.type == "toolCall":
                        print()
                        has_tool_call = True
                        tool_calls.append(content)

                        tool_result = self._tool_executor.execute(content.id, content.name, content.arguments)
                        print(f"\033[32mTool >> [{content.name}]:{tool_result} \033[0m")

                        tool_results.append(ToolResultMessage(
                            role="toolResult",
                            tool_call_id=content.id,
                            tool_name=content.name,
                            content=[TextContent(type="text", text=tool_result)],
                            timestamp=int(time.time() * 1000),
                        ))

            context.messages.extend([
                AssistantMessage(
                    role="assistant",
                    content=[TextContent(type="text", text=text_content)],
                    timestamp=get_timestamp_ms()),
                AssistantMessage(
                    role="assistant",
                    content=tool_calls,
                    timestamp=get_timestamp_ms()
                ),
                *tool_results
            ])
            print()
            if not has_tool_call:
                break
