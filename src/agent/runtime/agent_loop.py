import asyncio
import json
from typing import Awaitable, Callable

from src.agent.constant import Constants
from src.agent.runtime.tool_executor import ToolExecutor
from src.agent.status_bar import StatusBar
from src.agent.types import (
    AgentEndEvent,
    AgentEvent,
    AgentMessage,
    AgentMessageEndEvent,
    AgentMessageStartEvent,
    AgentMessageUpdateEvent,
    AgentStartEvent,
    AgentToolExecutionEndEvent,
    AgentToolExecutionStartEvent,
    AgentTurnEndEvent,
    AgentTurnStartEvent,
)
from src.ai.api.all import get_api_by_api
from src.ai.model import Model
from src.ai.model import StreamOptions
from src.ai.types import AssistantMessageEventType
from src.ai.types import (
    AssistantMessage,
    Context,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from src.util.time_util import get_timestamp_ms


class AgentLoop:

    def __init__(self,
                 tool_executor: ToolExecutor,
                 emit: Callable[[AgentEvent], Awaitable[None]]):
        self._tool_executor = tool_executor
        self._emit = emit
        self._status_bar = StatusBar()

    async def run_loop(
            self,
            model: Model,
            prompt: AgentMessage,
            context: Context,
            stream_options: StreamOptions,
            stop_event: asyncio.Event,
    ) -> None:

        await self._emit(AgentStartEvent())
        await self._emit(AgentTurnStartEvent())
        await self._emit(AgentMessageStartEvent(prompt))
        await self._emit(AgentMessageEndEvent(prompt))
        is_first_turn = True

        for _ in range(Constants.MAX_TURN):
            if stop_event.is_set():
                break
            api = get_api_by_api(model.api)
            if api is None:
                raise ValueError("该模型找不到对应的api协议")

            # 每次循环前注入系统状态栏信息
            self._inject_status_bar(context)

            has_tool_call = False

            last_assistant_message = None
            tool_calls: list[ToolCall] = []
            tool_results: list[ToolResultMessage] = []

            if not is_first_turn:
                await self._emit(AgentTurnStartEvent())

            is_first_message = True
            async for message, event in api.stream(model, context, stream_options):
                if stop_event.is_set():
                    break

                if is_first_message:
                    is_first_message = False
                    await self._emit(AgentMessageStartEvent(message))
                else:
                    await self._emit(AgentMessageUpdateEvent(message, event))

                if event.type == AssistantMessageEventType.TOOLCALL_DELTA:
                    has_tool_call = True
                    toolcall = event.partial.content[event.content_index]
                    tool_calls.append(toolcall)

                    await self._emit(
                        AgentToolExecutionStartEvent(
                            tc_id=toolcall.id,
                            tool_name=toolcall.name,
                            arguments=json.dumps(toolcall.arguments),
                        )
                    )
                    is_success, tool_result = await asyncio.to_thread(
                        self._tool_executor.execute,
                        toolcall.id, toolcall.name, toolcall.arguments
                    )
                    await self._emit(
                        AgentToolExecutionEndEvent(
                            tc_id=toolcall.id,
                            tool_name=toolcall.name,
                            arguments=json.dumps(toolcall.arguments),
                            tool_result=tool_result,
                            is_success=is_success,
                        )
                    )

                    tool_results.append(ToolResultMessage(
                        tool_call_id=toolcall.id,
                        tool_name=toolcall.name,
                        content=[TextContent(text=tool_result)],
                        timestamp=get_timestamp_ms(),
                        is_success=is_success,
                    ))

                if event.type == AssistantMessageEventType.DONE:
                    last_assistant_message = event.message
                    await self._emit(AgentMessageEndEvent(event.message))
                    await self._emit(AgentTurnEndEvent(event.message, tool_results))

            if last_assistant_message is not None:
                context.messages.append(last_assistant_message)
            if len(tool_calls) > 0:
                context.messages.extend([
                    AssistantMessage(
                        content=tool_calls,
                        timestamp=get_timestamp_ms()
                    ),
                    *tool_results
                ])
            if not has_tool_call:
                break

        await self._emit(AgentEndEvent(context.messages))

    def _inject_status_bar(self, context: Context) -> None:
        """把当前状态栏作为一条 user message 注入 context。"""
        if self._status_bar is None:
            return

        status_text = f"<status>{self._status_bar.build()}</status>"
        msg = UserMessage(
            content=[TextContent(text=status_text)],
            timestamp=get_timestamp_ms(),
        )
        context.messages.append(msg)
