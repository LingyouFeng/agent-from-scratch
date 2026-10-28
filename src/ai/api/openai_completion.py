"""
接入openai completion api
"""

import json
import logging
from typing import AsyncIterator

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
)

from ai.model import Model, StreamOptions
from ai.provider.registry import default_registry
from ai.types import (
    AssistantMessage,
    AssistantMessageDoneEvent,
    AssistantMessageEvent,
    AssistantMessageStartEvent,
    AssistantMessageTextDeltaEvent,
    AssistantMessageTextEndEvent,
    AssistantMessageTextStartEvent,
    AssistantMessageThinkDeltaEvent,
    AssistantMessageThinkEndEvent,
    AssistantMessageThinkStartEvent,
    AssistantMessageToolCallDeltaEvent,
    AssistantMessageToolCallEndEvent,
    AssistantMessageToolCallStartEvent,
    Context,
    Message,
    TextContent,
    ThinkingContent,
    ToolCall,
)
from ai.types import AssistantMessageErrorEvent
from util.time_util import get_timestamp_ms


class OpenAICompletionAPI:
    """openai completion api 接入：stream 流式 + complete 非流式"""

    async def stream(
        self, model: Model, context: Context, options: StreamOptions | None = None
    ) -> AsyncIterator[tuple[AssistantMessage, AssistantMessageEvent]]:
        """流式调用，逐步产出消息与事件"""
        client = self._create_client(model)
        api_messages = self._build_messages(context)
        api_tools = self._build_tools(context)
        thinking_level = options.thinking_level if options else "off"

        content: list[TextContent | ThinkingContent | ToolCall] = []
        text_index, think_index = None, None
        toolcall_index: dict[int, int] = {}
        text_buffer, think_buffer = "", ""
        toolcall_buffer: dict[int, dict] = {}
        assistant_message = AssistantMessage(
            content=content, timestamp=get_timestamp_ms()
        )

        try:
            response = await client.chat.completions.create(
                model=model.id,
                messages=api_messages,
                stream=True,
                stream_options={"include_usage": True},
                tools=api_tools,
                reasoning_effort=thinking_level,
            )

            # 发送 start 事件
            yield assistant_message, AssistantMessageStartEvent()

            async for chunk in response:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                assistant_message.timestamp = chunk.created

                # 思考文本
                reasoning = getattr(delta, "reasoning", None)
                if reasoning:
                    # 发送 think_start 事件
                    if think_index is None:
                        think_index = len(content)
                        content.append(ThinkingContent(think_buffer))
                        yield assistant_message, AssistantMessageThinkStartEvent(
                            content_index=think_index,
                            partial=assistant_message,
                        )
                    # 发送 think_delta 事件
                    think_buffer += reasoning
                    content[think_index] = ThinkingContent(think_buffer)
                    yield assistant_message, AssistantMessageThinkDeltaEvent(
                        content_index=think_index,
                        delta=reasoning,
                        partial=assistant_message,
                    )

                # 回复文本
                text = delta.content
                if text:
                    # 发送 text_start 事件
                    if text_index is None:
                        text_index = len(content)
                        content.append(TextContent(text_buffer))
                        yield assistant_message, AssistantMessageTextStartEvent(
                            content_index=text_index,
                            partial=assistant_message,
                        )
                    # 发送 text_delta 事件
                    text_buffer += text
                    content[text_index] = TextContent(text_buffer)
                    yield assistant_message, AssistantMessageTextDeltaEvent(
                        content_index=text_index,
                        delta=text,
                        partial=assistant_message,
                    )

                # 工具调用
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.function is None:
                            continue
                        # 发送 toolcall_start 事件
                        if toolcall_index.get(tc.index, None) is None:
                            toolcall_index[tc.index] = len(content)
                            toolcall_buffer[tc.index] = {
                                "id": tc.id,
                                "name": tc.function.name,
                                "args": "",
                            }

                            content.append(toolcall_buffer[tc.index])
                            yield assistant_message, AssistantMessageToolCallStartEvent(
                                content_index=toolcall_index[tc.index],
                                partial=assistant_message,
                            )
                        # 发送 toolcall_delta 事件
                        toolcall_buffer[tc.index]["args"] += tc.function.arguments or ""

                        # 尝试去解析工具调用，这里只解析完整的json，不支持增量解析
                        raw_json = toolcall_buffer[tc.index]["args"]
                        try:
                            arguments = json.loads(raw_json)
                            content[toolcall_index[tc.index]] = ToolCall(
                                id=toolcall_buffer[tc.index]["id"],
                                name=toolcall_buffer[tc.index]["name"],
                                arguments=arguments,
                            )
                            yield assistant_message, AssistantMessageToolCallDeltaEvent(
                                content_index=toolcall_index[tc.index],
                                delta=raw_json,
                                partial=assistant_message,
                            )
                        except json.JSONDecodeError:
                            pass

                # 结束
                if choice.finish_reason:
                    if think_index is not None:
                        yield assistant_message, AssistantMessageThinkEndEvent(
                            content_index=think_index,
                            content=think_buffer,
                            partial=assistant_message,
                        )
                    if text_index is not None:
                        yield assistant_message, AssistantMessageTextEndEvent(
                            content_index=text_index,
                            content=text_buffer,
                            partial=assistant_message,
                        )
                    for idx in toolcall_index.values():
                        yield assistant_message, AssistantMessageToolCallEndEvent(
                            content_index=idx,
                            toolcall=content[idx],
                            partial=assistant_message,
                        )
                    # 发送 end 事件
                    yield assistant_message, AssistantMessageDoneEvent(
                        reason=choice.finish_reason,
                        message=assistant_message,
                    )

        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            logging.error(f"\n遇到异常 {type(e).__name__}")
            yield assistant_message, AssistantMessageErrorEvent(
                reason="error",
                error=assistant_message,
            )
            raise

    async def complete(
        self, model: Model, context: Context, options: StreamOptions | None = None
    ) -> AssistantMessage:
        """非流式调用，一次性返回完整消息"""
        client = self._create_client(model)
        api_messages = self._build_messages(context)
        api_tools = self._build_tools(context)
        thinking_level = options.thinking_level if options else "off"

        content: list[TextContent | ThinkingContent | ToolCall] = []
        assistant_message = AssistantMessage(
            content=content, timestamp=get_timestamp_ms()
        )

        try:
            response = await client.chat.completions.create(
                model=model.id,
                messages=api_messages,
                stream=False,
                tools=api_tools,
                reasoning_effort=thinking_level,
            )
            choice = response.choices[0]
            message = choice.message
            assistant_message.timestamp = response.created * 1000

            if message:
                # 思考文本
                reasoning = getattr(message, "reasoning", None)
                if reasoning:
                    content.append(ThinkingContent(thinking=reasoning))

                # 回复文本
                if message.content:
                    content.append(TextContent(text=message.content))

                # 工具调用
                for tc in message.tool_calls or []:
                    try:
                        arguments = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"Tool {tc.function.name} arguments JSON 解析失败"
                        ) from e
                    content.append(
                        ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=arguments,
                        )
                    )

        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            logging.error(f"\n遇到异常 {type(e).__name__}")
            raise

        return assistant_message

    def _create_client(self, model: Model) -> AsyncOpenAI:
        provider = default_registry.get_provider_by_id(model.provider)
        if provider is None:
            raise ValueError(
                f"No built-in provider {model.provider} with model {model.id}"
            )

        if provider.api_key is None:
            raise ValueError(
                f"Provider {provider.id} with model {model.id} has not configure api key"
            )

        if provider.base_url is None:
            raise ValueError(
                f"Provider {provider.id} with model {model.id} has not configure base url"
            )

        return AsyncOpenAI(api_key=provider.api_key, base_url=provider.base_url)

    def _build_messages(self, context: Context) -> list[ChatCompletionMessageParam]:
        # 组装请求messages：系统提示 + 历史消息
        api_messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system", content=context.systemPrompt
            )
        ]
        api_messages.extend(
            converted for msg in context.messages if (converted := convert_message(msg))
        )
        return api_messages

    def _build_tools(self, context: Context) -> list[ChatCompletionFunctionToolParam]:
        # 注入工具
        api_tools: list[ChatCompletionFunctionToolParam] = []
        if context.tools:
            for t in context.tools:
                tool = ChatCompletionFunctionToolParam(
                    type="function",
                    function={
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                )
                api_tools.append(tool)
        return api_tools


def convert_message(msg: Message) -> ChatCompletionMessageParam | None:
    """转换自定义的Message -> openai api messages格式"""
    if msg.role == "user":
        if isinstance(msg.content, str):
            return {"role": "user", "content": msg.content}
        elif isinstance(msg.content, list):
            content_parts = []
            for item in msg.content:
                if item.type == "text":
                    content_parts.append({"type": "text", "text": item.text})
                elif item.type == "image":
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{item.mimeType};base64,{item.data}"
                            },
                        }
                    )
            return {"role": "user", "content": content_parts}
    elif msg.role == "assistant":
        reasoning_text, content_parts, tool_calls = "", [], []
        for item in msg.content:
            if item.type == "thinking":
                reasoning_text += item.thinking
            elif item.type == "text":
                content_parts.append({"type": "text", "text": item.text})
            elif item.type == "toolCall":
                tool_calls.append(
                    {
                        "id": item.id,
                        "type": "function",
                        "function": {
                            "name": item.name,
                            "arguments": json.dumps(item.arguments),
                        },
                    }
                )
        return {
            "role": "assistant",
            "reasoning_content": reasoning_text,
            "content": content_parts,
            **({"tool_calls": tool_calls} if tool_calls else {}),
        }
    elif msg.role == "toolResult":
        content_parts = []
        for item in msg.content:
            if item.type == "text":
                content_parts.append(item.text)
        return {
            "role": "tool",
            "tool_call_id": msg.tool_call_id,
            "content": "\n".join(content_parts),
        }
    return None
