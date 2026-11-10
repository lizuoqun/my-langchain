import time
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy

from operator import add
from loguru import logger


class EmptyState(TypedDict):
    user: str
    invoke_counts: Annotated[int, add]


def node_a(state: EmptyState) -> dict[str, int]:
    logger.info("node_a 被调用, user: {}", state["user"])
    time.sleep(3)  # 模拟耗时操作
    logger.info("node_a 耗时操作执行完毕")

    return {
        "invoke_counts": 1
    }


builder = StateGraph(state_schema=EmptyState)  # type: ignore[arg-type]
builder.add_node(
    "node_a",
    node_a,  # type: ignore[arg-type]
    cache_policy=CachePolicy(ttl=10)
)
builder.add_edge(START, "node_a")
builder.add_edge("node_a", END)

graph = builder.compile(cache=InMemoryCache())
logger.info("运行结果: {}\n\n", graph.invoke({"user": "小明", "invoke_counts": 0}))
logger.info("运行结果: {}\n\n", graph.invoke({"user": "小明", "invoke_counts": 0}))
logger.info("运行结果: {}\n\n", graph.invoke({"user": "小花", "invoke_counts": 0}))
logger.info("运行结果: {}\n\n", graph.invoke({"user": "小花", "invoke_counts": 2}))
time.sleep(10)  # 确保第一次调用缓存失效
logger.info("运行结果: {}", graph.invoke({"user": "小明", "invoke_counts": 0}))
