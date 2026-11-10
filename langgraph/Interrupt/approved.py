from typing import TypedDict, Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain_deepseek import ChatDeepSeek
from langchain_core.runnables import RunnableConfig

from dotenv import load_dotenv

load_dotenv(override=True)

model = ChatDeepSeek(
    model="deepseek-v4-flash",
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    }
)


# 1. 声明状态
class OverAllState(TypedDict):
    topic: str
    poem: str
    is_approved: bool


# 2. 声明节点
def approve_node(state: OverAllState) -> Command[Literal["llm_node", "default_node"]]:
    is_approved = interrupt("是否同意调用模型?")
    goto = "llm_node" if is_approved else "default_node"
    return Command(
        goto=goto,
        update={"is_approved": is_approved}
    )


def llm_node(state: OverAllState) -> dict[str, str]:
    topic = state["topic"]
    res = model.invoke([HumanMessage(content=f"帮我写一首关于{topic}主题的七言绝句,只写诗句,不需要赏析")])

    return {
        "poem": str(res.content)
    }


def default_node(state: OverAllState) -> dict[str, str]:
    return {
        "poem": "请求被拒绝"
    }


# 3. 构建图
builder = StateGraph(state_schema=OverAllState)  # type: ignore

builder.add_node("approve_node", approve_node)  # type: ignore
builder.add_node("llm_node", llm_node)  # type: ignore
builder.add_node("default_node", default_node)  # type: ignore
builder.add_edge(START, "approve_node")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config: RunnableConfig = {
    "configurable": {
        "thread_id": "thread-001"
    }
}
interrupt_res = graph.invoke({"topic": "菊花"}, config=config)
print(interrupt_res)

user_approved = input("是否同意调用模型?(y/n)").strip().lower() == 'y'
approved_res = graph.invoke(Command(resume=user_approved), config=config)
print(approved_res)
