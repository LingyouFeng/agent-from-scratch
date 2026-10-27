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

from src.ai.model import Model, StreamOptions
from src.ai.provider.registry import default_registry
from src.ai.types import (
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
from src.ai.types import AssistantMessageErrorEvent
from src.util.time_util import get_timestamp_ms


async def stream(
    model: Model, context: Context, options: StreamOptions | None = None
) -> AsyncIterator[tuple[AssistantMessage, AssistantMessageEvent]]:
    provider = default_registry.get_provider_by_id(model.provider)
    if provider is None:
        raise ValueError(f"No built-in provider {provider} with model {model.id}")

    if provider.api_key is None:
        raise ValueError(f"Provider {provider.id} with model {model.id} has not configure api key")
    
    if provider.base_url is None:
        raise ValueError(f"Provider {provider.id} with model {model.id} has not configure base url")

    client = AsyncOpenAI(api_key=provider.api_key, base_url=provider.base_url)

    # 组装请求messages：系统提示 + 历史消息
    api_messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content=context.systemPrompt)
    ]
    api_messages.extend(
        converted for msg in context.messages if (converted := convert_message(msg))
    )

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

    content: list[TextContent | ThinkingContent | ToolCall] = []
    text_index, think_index, toolcall_index = None, None, {}
    text_buffer, think_buffer, toolcall_buffer = "", "", {}
    assistant_message = AssistantMessage(content=content, timestamp=get_timestamp_ms())

    try:
        stream_enabled = options.stream if options else False
        thinking_level = options.thinking_level if options else "none"
        response = await client.chat.completions.create(
            model=model.id,
            messages=api_messages,
            stream=stream_enabled,
            stream_options={"include_usage": True},
            tools=api_tools,
            reasoning_effort=thinking_level,
        )

        # 发送 start 事件
        yield assistant_message, AssistantMessageStartEvent()

        if stream_enabled:
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
                            arguments, action_note = split_action_note(
                                json.loads(raw_json)
                            )
                            content[toolcall_index[tc.index]] = ToolCall(
                                id=toolcall_buffer[tc.index]["id"],
                                name=toolcall_buffer[tc.index]["name"],
                                arguments=arguments,
                                action_note=action_note,
                            )
                            yield assistant_message, AssistantMessageToolCallDeltaEvent(
                                content_index=toolcall_index[tc.index],
                                delta=raw_json,
                                partial=assistant_message,
                            )
                        except json.JSONDecodeError as e:
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
                    for toolcall_index in toolcall_index.values():
                        yield assistant_message, AssistantMessageToolCallEndEvent(
                            content_index=toolcall_index,
                            toolcall=content[toolcall_index],
                            partial=assistant_message,
                        )
                    # 发送 end 事件
                    yield assistant_message, AssistantMessageDoneEvent(
                        reason=choice.finish_reason,
                        message=assistant_message,
                    )

        else:
            choice = response.choices[0]
            message = choice.message
            assistant_message.timestamp = response.created * 1000

            if message:
                think_buffer = getattr(message, "reasoning", None)
                if think_buffer:
                    think_index = len(content)
                    # 发送 think_start 事件
                    yield assistant_message, AssistantMessageThinkStartEvent(
                        content_index=think_index,
                        partial=assistant_message,
                    )
                    content.append(ThinkingContent(thinking=think_buffer))
                    # 发送 think_delta 事件
                    yield assistant_message, AssistantMessageThinkDeltaEvent(
                        content_index=think_index,
                        delta=think_buffer,
                        partial=assistant_message,
                    )
                    # 发送 think_end 事件
                    yield assistant_message, AssistantMessageThinkEndEvent(
                        content_index=think_index,
                        content=think_buffer,
                        partial=assistant_message,
                    )

                text_buffer = message.content
                if text_buffer:
                    text_index = len(content)
                    # 发送 text_start 事件
                    yield assistant_message, AssistantMessageTextStartEvent(
                        content_index=text_index,
                        partial=assistant_message,
                    )
                    content.append(TextContent(text=text_buffer))
                    # 发送 text_delta 事件
                    yield assistant_message, AssistantMessageTextDeltaEvent(
                        content_index=text_index,
                        delta=text_buffer,
                        partial=assistant_message,
                    )
                    # 发送 text_end 事件
                    yield assistant_message, AssistantMessageTextEndEvent(
                        content_index=text_index,
                        content=text_buffer,
                        partial=assistant_message,
                    )

            for idx, tc in enumerate(message.tool_calls or []):
                try:
                    arguments, action_note = split_action_note(
                        json.loads(tc.function.arguments or "{}")
                    )
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Tool {tc.function.name} arguments JSON 解析失败"
                    ) from e
                # 发送 toolcall_start 事件
                toolcall_index[idx] = len(content)
                content.append(
                    ToolCall(id="", name="", arguments={}, action_note=action_note)
                )
                yield assistant_message, AssistantMessageToolCallStartEvent(
                    content_index=toolcall_index[idx],
                    partial=assistant_message,
                )
                # 发送 toolcall_delta 事件
                content[toolcall_index[idx]] = ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments,
                    action_note=action_note,
                )
                yield assistant_message, AssistantMessageToolCallDeltaEvent(
                    content_index=toolcall_index[idx],
                    delta=tc.function.arguments,
                    partial=assistant_message,
                )
                # 发送 toolcall_end 事件
                yield assistant_message, AssistantMessageToolCallEndEvent(
                    content_index=toolcall_index[idx],
                    toolcall=content[toolcall_index[idx]],
                    partial=assistant_message,
                )

            if choice.finish_reason:
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


def split_action_note(arguments: dict[str, any]) -> tuple[dict[str, any], str]:
    """把参数 action_note（模型填写的执行描述）从真实工具参数中分离。

    返回 (真实参数, 执行描述)。执行描述仅用于 UI 展示，永远不该传给工具。
    """
    action = arguments.get("action_note", "")
    real = {k: v for k, v in arguments.items() if k != "action_note"}
    return real, action if isinstance(action, str) else ""
