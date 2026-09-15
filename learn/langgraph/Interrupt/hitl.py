from typing import TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver


# 1. 声明状态
class OverAllState(TypedDict):
    username: str


# 2. 声明节点
def node_a(state: OverAllState) -> OverAllState:
    return {
        "username": interrupt("请输入您的姓名")
    }


# 3. 构建图
builder = StateGraph(state_schema=OverAllState)  # type: ignore
builder.add_node("node_a", node_a)  # type: ignore
builder.add_edge(START, "node_a")

# 4. 想要使用中断 => 必须配置检查点
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config: RunnableConfig = {
    "configurable": {
        "thread_id": "thread-001"
    }
}

interrupt_res = graph.invoke({}, config=config)
print(interrupt_res)

prompt = interrupt_res['__interrupt__'][0].value
username = input(prompt)

resume_res = graph.invoke(Command(resume=username), config=config)
print(resume_res)
