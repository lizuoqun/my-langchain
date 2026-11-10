from random import randint
from typing import Literal, cast

from langchain_core.messages import ToolMessage, AnyMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.messages import HumanMessage

from dotenv import load_dotenv
from langgraph.types import Command

load_dotenv(override=True)

model = ChatDeepSeek(
    model='deepseek-v4-flash',
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    }
)


@tool(parse_docstring=True)
def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息

    Args:
        city: str 城市
    """
    return f"{city} 今天天气晴朗"


@tool(parse_docstring=True)
def get_news(category: str) -> str:
    """
    获取特定类型的新闻

    Args:
        category: str 新闻类型
    """
    return f"{category}今天没有新闻"


tools = [get_weather, get_news]
model_with_tools = model.bind_tools(tools=tools)


class OverAllState(MessagesState):
    user_input: str
    final_answer: str


def input_node(state: OverAllState) -> dict[str, list[HumanMessage]]:
    return {
        "messages": [HumanMessage(state["user_input"])],
    }


def llm_node(state: OverAllState) -> Command[Literal["tool_node", "output_node"]]:
    messages = state["messages"]
    ai_msg = model_with_tools.invoke(messages)
    if ai_msg.tool_calls:
        goto = "tool_node"
    else:
        goto = "output_node"

    return Command(
        goto=goto,
        update={
            "messages": [ai_msg]
        }
    )


def random_err(tool_call) -> ToolMessage | None:
    fail_prob = 6  # 失败概率
    if randint(0, 9) < fail_prob:
        return ToolMessage(
            content=f"网络波动，调用失败，请重试，当前调用id为： {tool_call['id']}",
            tool_call_id=tool_call["id"]
        )
    return None


def tool_node(state: OverAllState) -> dict[str, list[AnyMessage]]:
    messages = state["messages"]
    ai_msg = cast(AIMessage, messages[-1])
    tool_calls = ai_msg.tool_calls

    for tool_call in tool_calls:
        if tool_call["name"] == "get_weather":
            err_msg = random_err(tool_call)
            if err_msg:
                messages.append(err_msg)
            else:
                messages.append(get_weather.invoke(tool_call))
        elif tool_call["name"] == "get_news":
            err_msg = random_err(tool_call)
            if err_msg:
                messages.append(err_msg)
            else:
                messages.append(get_news.invoke(tool_call))
        else:
            messages.append(
                ToolMessage(
                    content="工具名称错误，调用失败，请重试",
                    tool_call_id=tool_call["id"]
                )
            )

    return {
        "messages": messages
    }


def output_node(state: OverAllState) -> dict[str, str]:
    result = state["messages"][-1].content
    return {
        "final_answer": str(result)
    }


builder = StateGraph(state_schema=OverAllState)  # type: ignore[arg-type]
builder.add_node("input_node", input_node)  # type: ignore[arg-type]
builder.add_node("llm_node", llm_node)  # type: ignore[arg-type]
builder.add_node("tool_node", tool_node)  # type: ignore[arg-type]
builder.add_node("output_node", output_node)  # type: ignore[arg-type]

builder.add_edge(START, "input_node")
builder.add_edge("input_node", "llm_node")
builder.add_edge("tool_node", "llm_node")
builder.add_edge("output_node", END)

graph = builder.compile()
ai_res = graph.invoke({
    "user_input": "帮我查询杭州天气和农业这个类型的新闻",
    "messages": [SystemMessage("如果工具调用失败，必须重新调用直至成功")]
})

print("user_input: ", ai_res["user_input"])
print("final_answer: ", ai_res["final_answer"])
for msg in ai_res["messages"]:
    msg.pretty_print()

png_bytes = graph.get_graph().draw_mermaid_png()
raw_mermaid = graph.get_graph().draw_mermaid()
print(raw_mermaid)
