from typing import TypedDict, Literal, Sequence

from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, START, END
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
    ci_poem: str
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


def node_c(state: OverAllState) -> dict[str, str]:
    ci_poem = model.invoke([
        HumanMessage(f"写一首关于 {state['topic']} 的词，只输出词"),
    ])

    return {
        "ci_poem": str(ci_poem.content)
    }


def router(state: OverAllState) -> Sequence[Literal["node_a", "node_b", "node_c"]]:
    if "诗" in state["content_type"]:
        return ["node_a", "node_c"]
    return ["node_b", "node_c"]


def router_by_path_map(state: OverAllState) -> Sequence[Literal["poem", "joke", "ci_poem"]]:
    if "诗" in state["content_type"]:
        return ["poem", "ci_poem"]
    return ["joke", "ci_poem"]


builder = StateGraph(state_schema=OverAllState)  # type: ignore[arg-type]
builder.add_node(node_a)  # type: ignore[arg-type]
builder.add_node(node_b)  # type: ignore[arg-type]
builder.add_node(node_c)  # type: ignore[arg-type]

builder.add_conditional_edges(START, router)
# builder.add_conditional_edges(START, router_by_path_map, path_map={
#     "poem": "node_a",
#     "joke": "node_b",
#     "ci_poem": "node_c",
# })

builder.add_edge("node_a", END)
builder.add_edge("node_b", END)
builder.add_edge("node_c", END)

graph = builder.compile()
result = graph.invoke({"topic": "青山", "content_type": "诗"})
print(result)

png_bytes = graph.get_graph().draw_mermaid_png()
png_filename = "graph.png"
with open(png_filename, "wb") as f:
    f.write(png_bytes)