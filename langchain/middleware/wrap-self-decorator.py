import os
from typing import Callable

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse, ExtendedModelResponse, \
    wrap_tool_call
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_API_BASE = os.getenv("DEFAULT_API_BASE")

model = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEFAULT_API_BASE,
    extra_body={"thinking": {"type": "disabled"}}
)


@wrap_model_call
def wrap_model_call_middleware(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse | AIMessage | ExtendedModelResponse:
    request.messages[-1].content += " -> wrap_model_call_before <- "
    response = handler(request)
    response.result[0].content += " -> wrap_model_call_after <- "
    return response


@tool
def get_weather(city: str, is_forcast: bool) -> str:
    """
    获取天气信息

    Args:
        city:城市名称
        is_forcast:明天的天气
    """
    if is_forcast:
        msg = f"{city}今天的天气是晴天，明天的天气是雨天"
    else:
        msg = f"{city}的天气是晴天"
    return msg


@wrap_tool_call
def wrap_tool_call_middleware(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    result = handler(request)
    print(f"原始参数：{request.tool_call['args']}")
    print(f"原始参数调用结果： {result}")

    request.tool_call["args"]["is_forcast"] = True
    result = handler(request)
    print(f"更新后的参数：{request.tool_call['args']}")
    print(f"更新参数调用结果： {result}")
    return result


agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[wrap_model_call_middleware, wrap_tool_call_middleware]
)

result = agent.invoke({
    "messages": [
        HumanMessage("你好，佛山今天天气咋样")
    ]
})

for msg in result["messages"]:
    msg.pretty_print()
