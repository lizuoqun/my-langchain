from typing import TypedDict, Literal

from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, START
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


class OverAllState(TypedDict):
    topic: str
    poem: str
    content_type: Literal["poem", "joke"]
    joke: str


# Literal[...] 类型标注只接受字面量表达式，这里可以用"__end__"代替END
def router(state: OverAllState) -> Command[Literal["poem_node", "joke_node", "__end__"]]:
    content_type = state["content_type"]
    if content_type == "poem":
        return Command(goto="poem_node")
    elif content_type == "joke":
        return Command(goto="joke_node")
    else:
        return Command(goto="__end__")


def poem_node(state: OverAllState) -> dict[str, str]:
    topic = state["topic"]
    prompt = f"请生成一首关于 {topic} 的七言绝句"
    poem = model.invoke([HumanMessage(prompt)]).content

    return {
        "poem": str(poem)
    }


def joke_node(state: OverAllState) -> dict[str, str]:
    topic = state["topic"]
    prompt = f"请生成一个关于 {topic} 的冷笑话"
    joke = model.invoke([HumanMessage(prompt)]).content

    return {
        "joke": str(joke)
    }


builder = StateGraph(state_schema=OverAllState)  # type: ignore[arg-type]
builder.add_node("router", router)  # type: ignore[arg-type]
builder.add_node("poem_node", poem_node)  # type: ignore[arg-type]
builder.add_node("joke_node", joke_node)  # type: ignore[arg-type]
builder.add_edge(START, "router")

graph = builder.compile()
res = graph.invoke({"topic": "小猫", "content_type": "poem"})
print(res)

png_bytes = graph.get_graph().draw_mermaid_png()
raw_mermaid = graph.get_graph().draw_mermaid()
print(raw_mermaid)