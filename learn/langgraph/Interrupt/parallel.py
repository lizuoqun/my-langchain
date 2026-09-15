from typing import TypedDict

from langgraph.graph import StateGraph, START
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver


class OverAllState(TypedDict):
    username: str
    age: int


def node_name(state: OverAllState) -> dict[str, str]:
    return {
        "username": interrupt("请输入您的姓名")
    }


def node_age(state: OverAllState) -> dict[str, str]:
    return {
        "age": interrupt("请输入您的年龄")
    }


builder = StateGraph(state_schema=OverAllState)  # type: ignore
builder.add_node("node_name", node_name)  # type: ignore
builder.add_node("node_age", node_age)  # type: ignore
builder.add_edge(START, "node_name")
builder.add_edge(START, "node_age")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config: RunnableConfig = {
    "configurable": {
        "thread_id": "thread-001"
    }
}
interrupted_res = graph.invoke({}, config=config)
print(interrupted_res)

resume_map = {}
for item in interrupted_res['__interrupt__']:
    resume_map[item.id] = input(f"{item.value}")

resumed_res = graph.invoke(Command(resume=resume_map), config=config)

print(resumed_res)
