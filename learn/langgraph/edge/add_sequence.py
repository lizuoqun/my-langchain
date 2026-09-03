from operator import add
from typing import TypedDict, Annotated, cast
from langgraph.constants import START, END
from langgraph.graph import StateGraph


class OverAllState(TypedDict):
    logs: Annotated[list[str], add]
    cur_id: str


def node_1(state: OverAllState) -> OverAllState:
    pre_id = state["cur_id"]
    return {
        "logs": ["node_1运行完毕"],
        "cur_id": pre_id + "， node_1"
    }


def node_2(state: OverAllState) -> OverAllState:
    pre_id = state["cur_id"]
    return {
        "logs": ["node_2运行完毕"],
        "cur_id": pre_id + "， node_2"
    }


builder = StateGraph(state_schema=cast(type[TypedDict], OverAllState))
builder.add_edge(START, "node_1")
builder.add_sequence([node_1, node_2])  # type: ignore[arg-type]
builder.add_edge("node_2", END)

graph = builder.compile()
result = graph.invoke({"cur_id": "start"})
print(result)
