"""
接入openai completion api
"""

import json
from typing import Iterator

from openai import (
    APIConnectionError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam,
)

from src.ai.model import Model, StreamOptions
from src.ai.provider.all import get_provider_by_id
from src.ai.type import (
    AssistantMessage,
    Context,
    Message,
    TextContent,
    ThinkingContent,
    ToolCall,
)
from src.util.time_util import get_timestamp_ms


def stream(model: Model, context: Context, options: StreamOptions | None = None) -> AssistantMessage | Iterator[AssistantMessage]:
    provider = get_provider_by_id(model.provider)
    if provider is None:
        raise ValueError(f"No built-in provider {provider} with model {model.id}")

    client = OpenAI(api_key=provider.api_key, base_url=provider.base_url)

    # 组装请求messages：系统提示 + 历史消息
    api_messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": context.systemPrompt}
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

    # print("=" * 100)
    # print(api_messages)
    # print("=" * 100)
    try:
        stream_enabled = options.stream if options else False
        response = client.chat.completions.create(
            model=model.name,
            messages=api_messages,
            stream=stream_enabled,
            stream_options={"include_usage": True},
            tools=api_tools,
            reasoning_effort=options.thinking_level,
        )

        if stream_enabled:
            tool_call_buffers: list[ToolCall] = []
            tool_args_buffers: dict[int, str] = {}

            for chunk in response:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta

                # 思考文本
                reasoning = getattr(delta, "reasoning", None)
                if reasoning:
                    yield AssistantMessage(
                        role="assistant",
                        content=[ThinkingContent(type="thinking", thinking=reasoning)],
                        timestamp=get_timestamp_ms(),
                    )

                # 正常响应文本
                if delta.content:
                    yield AssistantMessage(
                        role="assistant",
                        content=[TextContent(type="text", text=delta.content)],
                        timestamp=get_timestamp_ms(),
                    )

                # 收集工具调用
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        # 若 function 为 None，则无法提取有效信息，跳过该次迭代
                        if tc.function is None:
                            continue

                        # 确保 tool_calls 列表长度足够
                        while len(tool_call_buffers) <= tc.index:
                            tool_call_buffers.append(
                                ToolCall(
                                    type="toolCall",
                                    id="",
                                    name="",
                                    arguments={},
                                )
                            )
                        if tc.index not in tool_args_buffers:
                            tool_args_buffers[tc.index] = ""

                        tool_call = tool_call_buffers[tc.index]

                        # 更新 id（可能是在当前或后续 chunk 中提供）
                        if tc.id is not None:
                            tool_call.id = tc.id

                        # 更新 function name（可能是在后续 chunk 中提供）
                        if tc.function.name is not None:
                            tool_call.name = tc.function.name

                        # 拼接 arguments 的原始 JSON 字符串
                        if tc.function.arguments is not None:
                            tool_args_buffers[tc.index] += tc.function.arguments

                        # 尝试去解析工具调用
                        raw_json = tool_args_buffers[tc.index]
                        try:
                            arguments = json.loads(raw_json)
                            tool_call.arguments = arguments
                            # print(f"\ntool {tool_call.name} arguments json解析成功，直接返回")
                            yield AssistantMessage(
                                role="assistant",
                                content=[tool_call],
                                timestamp=get_timestamp_ms(),
                            )

                        except json.JSONDecodeError as e:
                            pass
                            # print(f"\ntool {tool_call.name} arguments json解析失败！继续等待后续片段, raw={raw_json}")

        else:
            # 非流式
            content_list: list[TextContent | ThinkingContent | ToolCall] = []
            message = response.choices[0].message

            if message:
                content_list.append(TextContent(type="text", text=message.content))

            for tc in message.tool_calls or []:
                try:
                    arguments = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Tool {tc.function.name} arguments JSON 解析失败"
                    ) from e

                content_list.append(
                    ToolCall(
                        type="toolCall",
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    )
                )

            return AssistantMessage(
                role="assistant",
                content=content_list,
                timestamp=get_timestamp_ms(),
            )

    except (RateLimitError, APITimeoutError, APIConnectionError) as e:
        print(f"\n遇到异常 {type(e).__name__}")
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
                            "arguments": json.dumps(item.arguments)
                        },
                    }
                )
        return {
            "role": "assistant",
            "reasoning_content": reasoning_text,
            "content": content_parts,
            **({"tool_calls": tool_calls} if tool_calls else {})
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
