from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from loguru import logger


# 定义全局共享状态
class OverAllState(TypedDict):
    raw_text: str  # 未清洗文本
    cleaned_text: str  # 清洗后的文本


# 构建子图
def subgraph_strip_node(state: OverAllState) -> dict[str, str]:
    logger.info("子图 subgraph_strip_node 节点执行了")
    raw_text_res = state["raw_text"]
    stripped_text = raw_text_res.strip()

    return {
        "cleaned_text": stripped_text
    }


def subgraph_punctuate_node(state: OverAllState) -> dict[str, str]:
    logger.info("子图 subgraph_punctuate_node 节点执行了")
    cleaned_text_res = state["cleaned_text"]
    punctuation = interrupt("您希望在句尾添加的标点符号是？[。/，/！/？/；]")
    punctuation1 = interrupt("您希望在句尾添加的标点符号是？[。/，/！/？/；]")
    punctuated_text = cleaned_text_res + punctuation + punctuation1

    return {
        "cleaned_text": punctuated_text
    }


builder = StateGraph(state_schema=OverAllState)  # type: ignore[arg-type]
builder.add_node("subgraph_strip_node", subgraph_strip_node)  # type: ignore[arg-type]
builder.add_node("subgraph_punctuate_node", subgraph_punctuate_node)  # type: ignore[arg-type]

builder.add_edge(START, "subgraph_strip_node")
builder.add_edge("subgraph_strip_node", "subgraph_punctuate_node")
builder.add_edge("subgraph_punctuate_node", END)

# ==================================================
# 【策略切换点】只需修改这一行
subgraph = builder.compile()  # Per-invocation（默认）
# subgraph = builder.compile(checkpointer=True)   # Per-thread
# subgraph = builder.compile(checkpointer=False)  # Stateless（中断不可用）
# ==================================================

# 构建父图
builder = StateGraph(state_schema=OverAllState)  # type: ignore[arg-type]
builder.add_node("subgraph_node", subgraph)

builder.add_edge(START, "subgraph_node")
builder.add_edge("subgraph_node", END)

checkpointer = InMemorySaver()
parent_graph = builder.compile(checkpointer=checkpointer)

raw_text = "   LangGraph 真有意思        "
config: RunnableConfig = {"configurable": {"thread_id": "123"}}
# 首次调用
interrupted_res = parent_graph.invoke(
    {"raw_text": raw_text},
    config=config
)
print("=" * 30, "->   中断信息   <-", "=" * 30)
print(interrupted_res)

from IPython.display import display, Image

display(
    Image(
        parent_graph
        .get_graph(xray=True)
        .draw_mermaid_png()
    )
)

# 恢复调用
parent_graph.invoke(
    Command(resume="！"),
    config=config
)

res = parent_graph.invoke(
    Command(resume="！"),
    config=config
)

cleaned_text = res["cleaned_text"]
print("=" * 30, "->   原始文本   <-", "=" * 30)
print(raw_text)
print("=" * 30, "-> 清洗后的文本 <-", "=" * 30)
print(cleaned_text)
