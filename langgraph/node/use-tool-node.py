from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.graph.message import MessagesState
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_deepseek import ChatDeepSeek

from dotenv import load_dotenv

load_dotenv(override=True)


@tool(parse_docstring=True)
def get_weather(city: str) -> str:
    """
    查询指定城市的当日天气

    Args:
        city: 城市名称
    """
    return f"{city} 今天天气不错"


@tool(parse_docstring=True)
def get_news(home_or_abroad: bool) -> str:
    """
    查询国内外新闻

    Args:
        home_or_abroad: 查询国内还是国外新闻，True: 国内新闻，False：国外新闻
    """
    if home_or_abroad:
        return "Kimi 新模型发布"
    return "Anthropic 暂停新模型访问"


tools = [get_weather, get_news]

model = ChatDeepSeek(
    model="deepseek-v4-flash",
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    }
)
model_with_tools = model.bind_tools(tools=tools)


def llm_node(state: MessagesState) -> MessagesState:
    messages = state['messages']
    response = model_with_tools.invoke(messages)

    return {
        "messages": [response]
    }


def router(state: MessagesState) -> Literal["tool_node", "__end__"]:
    if state['messages'][-1].tool_calls:
        return "tool_node"
    return "__end__"


builder = StateGraph(state_schema=MessagesState)  # type: ignore
builder.add_node("llm_node", llm_node)  # type: ignore
builder.add_node("tool_node", ToolNode(tools=tools))
builder.add_edge(START, "llm_node")
builder.add_conditional_edges("llm_node", router, path_map=["tool_node", END])
builder.add_edge("tool_node", "llm_node")

graph = builder.compile()

res = graph.invoke({"messages": [HumanMessage("今天北京天气如何？国内有哪些新闻？")]})
for msg in res['messages']:
    msg.pretty_print()
