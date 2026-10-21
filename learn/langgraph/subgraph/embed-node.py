from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 定义全局共享状态
class OverAllState(TypedDict):
    raw_text: str  # 未清洗文本
    cleaned_text: str  # 清洗后的文本


# 构建子图
def subgraph_strip_node(state: OverAllState) -> dict[str, str]:
    raw_text = state["raw_text"]
    stripped_text = raw_text.strip()

    return {
        "cleaned_text": stripped_text
    }


def subgraph_punctuate_node(state: OverAllState) -> dict[str, str]:
    cleaned_text = state["cleaned_text"]
    punctuated_text = cleaned_text + "。"

    return {
        "cleaned_text": punctuated_text
    }


builder = StateGraph(state_schema=OverAllState)  # type: ignore[arg-type]
builder.add_node("subgraph_strip_node", subgraph_strip_node)  # type: ignore[arg-type]
builder.add_node("subgraph_punctuate_node", subgraph_punctuate_node)  # type: ignore[arg-type]

builder.add_edge(START, "subgraph_strip_node")
builder.add_edge("subgraph_strip_node", "subgraph_punctuate_node")
builder.add_edge("subgraph_punctuate_node", END)

subgraph = builder.compile()

# 构建父图
builder = StateGraph(state_schema=OverAllState)  # type: ignore[arg-type]
builder.add_node("subgraph_node", subgraph)

builder.add_edge(START, "subgraph_node")
builder.add_edge("subgraph_node", END)

parent_graph = builder.compile()

raw_text = "   LangGraph 真有意思        "
res = parent_graph.invoke({"raw_text": raw_text})
cleaned_text = res["cleaned_text"]
print("=" * 30, "->   原始文本   <-", "=" * 30)
print(raw_text)
print("=" * 30, "-> 清洗后的文本 <-", "=" * 30)
print(cleaned_text)
