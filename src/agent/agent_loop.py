import time

from src.agent.tools.builtin import get_builtin_tools
from src.ai.api.all import get_stream_by_api
from src.ai.model import Model
from src.ai.type import Context, TextContent, ToolResultMessage

registry = get_builtin_tools()

def run_loop(model: Model, context: Context):
    for _ in range(3):  # 最多循环3次，防止无限循环
        print("="*100)
        print(context)
        print("="*100)
        stream = get_stream_by_api(model.api)
        if stream is None:
            raise ValueError("该模型找不到对应的api协议")

        assistant_message = stream(model, context)
        context.messages.append(assistant_message)

        has_tool_call = False
        for content in assistant_message.content:
            if content.type == "toolCall":
                has_tool_call = True
                print(f"调用工具：{content.name}...")
                tool_result = registry.execute(content.name, content.arguments)
                print(f"工具调用结果: {tool_result}")
                context.messages.append(ToolResultMessage(
                    role="toolResult",
                    tool_call_id=content.id,
                    tool_name=content.name,
                    content=[TextContent(type="text", text=tool_result)],
                    timestamp=int(time.time() * 1000),
                ))
        if not has_tool_call:
            break