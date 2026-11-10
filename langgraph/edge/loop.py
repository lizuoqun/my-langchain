from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.managed import RemainingSteps
from langchain_core.runnables.config import RunnableConfig

from loguru import logger


class OverAllState(TypedDict):
    remaining_steps: RemainingSteps


def loop_node(state: OverAllState, config: RunnableConfig) -> None:
    cur_step = config["metadata"]["langgraph_step"]
    remaining_steps = state["remaining_steps"]
    logger.info("loop_node, cur_step: {}, remaining_step: {}", cur_step, remaining_steps)


def router(state: OverAllState) -> Literal["loop_node", "__end__"]:
    remaining_steps = state["remaining_steps"]
    if remaining_steps < 3:
        logger.info("当前可用超步：{}，已不足2步，终止运行图", remaining_steps)
        return END
    return "loop_node"


builder = StateGraph(state_schema=OverAllState)  # type: ignore[arg-type]
builder.add_node("loop_node", loop_node)  # type: ignore[arg-type]
builder.add_edge(START, "loop_node")  # type: ignore[arg-type]
builder.add_conditional_edges("loop_node", router)  # type: ignore[arg-type]

graph = builder.compile()
config: RunnableConfig = {"recursion_limit": 10}
graph.invoke({}, config=config)
