from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

from requests.exceptions import HTTPError
from loguru import logger


class EmptyState(TypedDict):
    pass


def node_a() -> EmptyState:
    logger.info(f"node a 运行")
    raise HTTPError("网络连接超时...")


builder = StateGraph(state_schema=EmptyState)  # type: ignore[arg-type]
builder.add_node(
    "node_a",
    node_a,  # type: ignore[arg-type]
    retry_policy=RetryPolicy(
        max_attempts=3,
        jitter=False
    )
)

builder.add_edge(START, "node_a")
builder.add_edge("node_a", END)

graph = builder.compile()

try:
    graph.invoke({})
except HTTPError as e:
    logger.info("重试次数耗尽: {}", e)
