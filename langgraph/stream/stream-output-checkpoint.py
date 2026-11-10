from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.types import interrupt, Command


class OverAllState(TypedDict):
    initial_state: str
    parallel_node_a_1: str
    parallel_node_a_2: str
    node_b_output: str


def parallel_node_a_1(state: OverAllState) -> dict[str, str]:
    return {
        "parallel_node_a_1": "并行节点A-1的输出"
    }


def parallel_node_a_2(state: OverAllState) -> dict[str, str]:
    return {
        "parallel_node_a_2": "并行节点A-2的输出"
    }


def node_b(state: OverAllState) -> dict[str, str]:
    interrupt("hello")
    return {
        "node_b_output": "节点B的输出"
    }


builder = StateGraph(state_schema=OverAllState)  # type: ignore[arg-type]
builder.add_node("parallel_node_a_1", parallel_node_a_1)  # type: ignore[arg-type]
builder.add_node("parallel_node_a_2", parallel_node_a_2)  # type: ignore[arg-type]
builder.add_node("node_b", node_b)  # type: ignore[arg-type]
builder.add_edge(START, "parallel_node_a_1")
builder.add_edge(START, "parallel_node_a_2")
builder.add_edge(["parallel_node_a_1", "parallel_node_a_2"], "node_b")
builder.add_edge("node_b", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config: RunnableConfig = {"configurable": {"thread_id": "123"}}

pre_chunks = graph.stream(
    {"initial_state": "初始状态"},
    stream_mode=["checkpoints"],
    config=config
)

for chunk in pre_chunks:
    print(chunk)

print('=' * 30, '-> 中断前后分界线 <-', '=' * 30)

chunks = graph.stream(
    Command(resume=""),
    stream_mode=["checkpoints"],
    config=config
)

for chunk in chunks:
    print(chunk)
