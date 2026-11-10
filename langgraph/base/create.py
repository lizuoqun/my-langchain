from operator import add
from typing import TypedDict, Annotated, cast
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from IPython.display import display, Image


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


# 建造者
# builder = StateGraph(state_schema=OverAllState)
builder = StateGraph(state_schema=cast(type[TypedDict], OverAllState))

# 添加节点、边
builder.add_node(node_1) # type: ignore[arg-type]
builder.add_node(node_2) # type: ignore[arg-type]
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", END)

# 编译得到图，执行
graph = builder.compile()
result = graph.invoke({"cur_id": "start"})
print(result)

raw_mermaid = graph.get_graph().draw_mermaid()
print(raw_mermaid)

png_bytes = graph.get_graph().draw_mermaid_png()
png = Image(png_bytes)
display(png)

png_filename = "graph.png"
with open(png_filename, "wb") as f:
    f.write(png_bytes)
