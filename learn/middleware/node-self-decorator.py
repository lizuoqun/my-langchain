import os
from typing import Any

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model, after_model, before_agent, after_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.runtime import Runtime

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
def get_weather() -> str:
    """获取天气信息"""
    return "今天的天气阳光明媚"


# @before_model
@before_model(can_jump_to=["tools"])
def before_model_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    # state["messages"][-1].content += " -> before_model <- "
    # return None

    # 检查是否已经有工具调用记录（避免死循环）
    for msg in state["messages"]:
        # 如果已经有人工构造的消息，说明已经跳转过，直接返回
        if isinstance(msg, AIMessage) and msg.content.startswith("消息构造"):
            return None

    # 第一次调用时，跳过模型直接执行工具
    return {
        "messages": [
            AIMessage(
                content="消息构造：这是一条消息",
                tool_calls=[
                    {
                        "name": "get_weather",
                        "args": {},
                        "id": "call_force_weather_001",
                    }
                ],
            )],
        "jump_to": "tools",
    }


@after_model
def after_model_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    state["messages"][-1].content += " -> after_model <- "
    return None


@before_agent
def before_agent_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    state["messages"][-1].content += " -> before_agent <- "
    return None


@after_agent
def after_agent_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    state["messages"][-1].content += " -> after_agent <- "
    return None


agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[before_model_middleware, after_model_middleware, before_agent_middleware, after_agent_middleware]
)

result = agent.invoke({
    "messages": [
        HumanMessage("你好")
    ]
})

for msg in result["messages"]:
    msg.pretty_print()

# 你好！今天的天气阳光明媚，是个不错的日子哦～有什么可以帮你的吗？ -> after_model <-  -> after_agent <-