from typing import TypedDict, Literal

from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, START
from langchain.messages import HumanMessage

from dotenv import load_dotenv

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
    joke: str
    content_type: str


# 返回值类型为OverAllState，加上TypedDict要求完整性返回，这里我们只做部分返回，所以可以采用dict[str,str]
def node_a(state: OverAllState) -> dict[str, str]:
    poem = model.invoke([
        HumanMessage(f"写一首关于 {state['topic']} 的诗，只输出诗"),
    ])
    return {
        # 这里得到的是一个AIMessage、可以进去看其content类型，是str | list[str | dict]
        # 所以在这里进行强转
        "poem": str(poem.content)
    }


def node_b(state: OverAllState) -> dict[str, str]:
    joke = model.invoke([
        HumanMessage(f"写一个关于 {state['topic']} 的笑话，只输出笑话"),
    ])

    return {
        "joke": str(joke.content)
    }


def router(state: OverAllState) -> Literal["node_a", "node_b"]:
    if "诗" in state["content_type"]:
        return "node_a"
    return "node_b"


def router_by_path_map(state: OverAllState) -> Literal["a", "b"]:
    if "诗" in state["content_type"]:
        return "a"
    return "b"


builder = StateGraph(state_schema=OverAllState)  # type: ignore[arg-type]
builder.add_node(node_a)  # type: ignore[arg-type]
builder.add_node(node_b)  # type: ignore[arg-type]

# builder.add_conditional_edges(START, router)
builder.add_conditional_edges(START, router_by_path_map, path_map={
    "a": "node_a",
    "b": "node_b"
})

graph = builder.compile()
result = graph.invoke({"topic": "青山", "content_type": "诗"})
print(result)
