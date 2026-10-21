from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 构建子图
class SubgraphState(TypedDict):
    raw_text: str  # 未清洗文本
    stripped_text: str  # 去除收尾空格的文本
    punctuated_text: str  # 句尾添加句号的文本


def subgraph_strip_node(state: SubgraphState) -> dict[str, str]:
    raw_text = state["raw_text"]
    stripped_text = raw_text.strip()

    return {
        "stripped_text": stripped_text
    }


def subgraph_punctuate_node(state: SubgraphState) -> dict[str, str]:
    stripped_text = state["stripped_text"]
    punctuated_text = stripped_text + "。"

    return {
        "punctuated_text": punctuated_text
    }


builder = StateGraph(state_schema=SubgraphState)  # type: ignore[arg-type]
builder.add_node("subgraph_strip_node", subgraph_strip_node)  # type: ignore[arg-type]
builder.add_node("subgraph_punctuate_node", subgraph_punctuate_node)  # type: ignore[arg-type]

builder.add_edge(START, "subgraph_strip_node")
builder.add_edge("subgraph_strip_node", "subgraph_punctuate_node")
builder.add_edge("subgraph_punctuate_node", END)

subgraph = builder.compile()


# 构建父图
class ParentState(TypedDict):
    input_text: str  # 输入的未清洗的文本
    cleaned_text: str  # 清洗后的文本


def call_subgraph(state: ParentState) -> dict[str, str]:
    input_text = state["input_text"]

    res = subgraph.invoke({"raw_text": input_text})
    cleaned_text = res["punctuated_text"]
    return {
        "cleaned_text": cleaned_text
    }


builder = StateGraph(state_schema=ParentState)  # type: ignore[arg-type]
builder.add_node("call_subgraph", call_subgraph)  # type: ignore[arg-type]

builder.add_edge(START, "call_subgraph")
builder.add_edge("call_subgraph", END)

parent_graph = builder.compile()

input_text = "   LangGraph 真有意思        "
res = parent_graph.invoke({"input_text": input_text})
cleaned_text = res["cleaned_text"]
print("=" * 30, "->   原始文本   <-", "=" * 30)
print(input_text)
print("=" * 30, "-> 清洗后的文本 <-", "=" * 30)
print(cleaned_text)
