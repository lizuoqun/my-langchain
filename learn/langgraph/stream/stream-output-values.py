from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class OverAllState(TypedDict):
    initial_state: str
    node_a_output: str
    node_b_output: str


def node_a(state: OverAllState) -> dict[str, str]:
    return {
        "node_a_output": "节点A的输出"
    }


def node_b(state: OverAllState) -> dict[str, str]:
    return {
        "node_b_output": "节点B的输出"
    }


builder = StateGraph(state_schema=OverAllState)  # type: ignore[arg-type]
builder.add_node("node_a", node_a)  # type: ignore[arg-type]
builder.add_node("node_b", node_b)  # type: ignore[arg-type]
builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", END)

graph = builder.compile()

chunks = graph.stream(
    {"initial_state": "初始状态"},
    stream_mode=["values"],
)

for chunk in chunks:
    print(chunk)
