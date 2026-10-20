import time
from typing import AsyncIterator
from typing import Awaitable, Callable

from src.agent.runtime.tool_executor import ToolExecutor
from src.agent.types import (
    AgentEndEvent,
    AgentEvent,
    AgentMessageEndEvent,
    AgentMessageStartEvent,
    AgentMessageUpdateEvent,
    AgentStartEvent,
    AgentToolExecutionEndEvent,
    AgentToolExecutionStartEvent,
    AgentToolExecutionUpdateEvent,
    AgentTurnEndEvent,
    AgentTurnStartEvent,
)
from src.ai.api.all import get_stream_by_api
from src.ai.model import Model
from src.ai.model import StreamOptions
from src.ai.type import (
    AssistantMessage,
    Context,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
)
from src.util.time_util import get_timestamp_ms


class AgentLoop:

    def __init__(self, tool_executor: ToolExecutor, emit: Callable[[AgentEvent], Awaitable[None]]):
        self._tool_executor = tool_executor
        self._emit = emit

    async def run_loop(self, model: Model, context: Context, stream_options: StreamOptions) -> None:
        await self._emit(AgentStartEvent())
        for _ in range(10):  # 最多循环10次，防止无限循环
            # print("="*100)
            # print(context)
            # print("="*100)
            stream = get_stream_by_api(model.api)
            if stream is None:
                raise ValueError("该模型找不到对应的api协议")

            has_tool_call = False
            text_content = ""
            thinking_content = ""
            tool_calls: list[ToolCall] = []
            tool_results: list[ToolResultMessage] = []

            await self._emit(AgentTurnStartEvent())
            await self._emit(AgentMessageStartEvent())
            async for message in stream(model, context, stream_options):
                for content in message.content:
                    if content.type == "text":
                        text_content += content.text

                    elif content.type == "thinking":
                        thinking_content += content.thinking

                    elif content.type == "toolCall":
                        has_tool_call = True
                        tool_calls.append(content)

                        await self._emit(AgentToolExecutionStartEvent(
                            tc_id=content.id,
                            tool_name=content.name,
                            arguments=content.arguments,
                        ))
                        is_success, tool_result = self._tool_executor.execute(content.id, content.name, content.arguments)
                        await self._emit(AgentToolExecutionEndEvent(
                            tc_id=content.id,
                            tool_name=content.name,
                            arguments=content.arguments,
                            tool_result=tool_result,
                            is_success=is_success,
                        ))

                        tool_results.append(ToolResultMessage(
                            role="toolResult",
                            tool_call_id=content.id,
                            tool_name=content.name,
                            content=[TextContent(type="text", text=tool_result)],
                            timestamp=int(time.time() * 1000),
                        ))
                await self._emit(AgentMessageUpdateEvent(message))
            
            await self._emit(AgentMessageEndEvent())
            await self._emit(AgentTurnEndEvent())

            context.messages.append(
                AssistantMessage(
                    role="assistant",
                    content=[
                        *([ThinkingContent(type="thinking", thinking=thinking_content)] if thinking_content else []),
                        TextContent(type="text", text=text_content)
                    ],
                    timestamp=get_timestamp_ms())
            )
            if len(tool_calls) > 0:
                context.messages.extend([
                    AssistantMessage(
                        role="assistant",
                        content=tool_calls,
                        timestamp=get_timestamp_ms()
                    ),
                    *tool_results
                ])
            if not has_tool_call:
                break

        await self._emit(AgentEndEvent())
