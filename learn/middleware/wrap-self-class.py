import os
import random
import time
from typing import Any, Callable

from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse, ExtendedModelResponse
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
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


@tool
def get_weather(city: str, is_forcast: bool) -> str:
    """
    获取天气信息

    Args:
        city:城市名称
        is_forcast:明天的天气
    """
    time.sleep(random.uniform(0.5, 1.5))
    if is_forcast:
        msg = f"{city}今天的天气是晴天，明天的天气是雨天"
    else:
        msg = f"{city}的天气是晴天"
    return msg


class WrapModelCallMiddleWare(AgentMiddleware):
    def __init__(self):
        super().__init__()

    def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage | ExtendedModelResponse:
        request.messages[-1].content += " -> wrap_model_call_before <- "
        response = handler(request)
        response.result[0].content += " -> wrap_model_call_after <- "
        return response

    def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        tool_name = request.tool_call["name"]
        tool_args = request.tool_call.get("args", {})
        start_time = time.time()
        result = handler(request)
        elapsed = time.time() - start_time
        print(f"🔧 开始执行工具：{tool_name}，  参数：{tool_args} ✅ 工具执行成功，耗时：{elapsed:.2f}秒")
        return result


agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[WrapModelCallMiddleWare()]
)

result = agent.invoke({
    "messages": [
        HumanMessage("你好，佛山今天天气咋样")
    ]
})

for msg in result["messages"]:
    msg.pretty_print()
