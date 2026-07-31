import os
from typing import Callable, Any

from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse, ExtendedModelResponse, \
    wrap_tool_call, before_model, after_model
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


@before_model
def before_model_middleware1(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    state["messages"][-1].content += " -> before_model1 <- "
    return None


@before_model
def before_model_middleware2(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    state["messages"][-1].content += " -> before_model2 <- "
    return None


@before_model
def before_model_middleware3(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    state["messages"][-1].content += " -> before_model3 <- "
    return None


@after_model
def after_model_middleware1(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    state["messages"][-1].content += " -> after_model1 <- "
    return None


@after_model
def after_model_middleware2(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    state["messages"][-1].content += " -> after_model2 <- "
    return None


@after_model
def after_model_middleware3(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    state["messages"][-1].content += " -> after_model3 <- "
    return None


@wrap_model_call
def wrap_model_call_middleware1(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse | AIMessage | ExtendedModelResponse:
    request.messages[-1].content += " -> wrap_model_call_before1 <- "
    response = handler(request)
    response.result[0].content += " -> wrap_model_call_after1 <- "
    return response


@wrap_model_call
def wrap_model_call_middleware2(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse | AIMessage | ExtendedModelResponse:
    request.messages[-1].content += " -> wrap_model_call_before2 <- "
    response = handler(request)
    response.result[0].content += " -> wrap_model_call_after2 <- "
    return response


@wrap_model_call
def wrap_model_call_middleware3(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse | AIMessage | ExtendedModelResponse:
    request.messages[-1].content += " -> wrap_model_call_before3 <- "
    response = handler(request)
    response.result[0].content += " -> wrap_model_call_after3 <- "
    return response


agent = create_agent(
    model=model,
    middleware=[
        before_model_middleware1,
        before_model_middleware2,
        before_model_middleware3,
        after_model_middleware1,
        after_model_middleware2,
        after_model_middleware3,
        wrap_model_call_middleware1,
        wrap_model_call_middleware2,
        wrap_model_call_middleware3
    ]
)

result = agent.invoke({
    "messages": [
        HumanMessage("打个招呼")
    ]
})

for msg in result["messages"]:
    msg.pretty_print()
