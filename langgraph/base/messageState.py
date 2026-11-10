from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langchain.messages import HumanMessage

from dotenv import load_dotenv

load_dotenv(override=True)

model = ChatDeepSeek(
    model='deepseek-v4-flash',
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    }
)


class OverAllState(MessagesState):
    username: str
    output: str


def node_a(state: OverAllState) -> dict[str, str]:
    return {
        "messages": [HumanMessage("你好，我是 " + state["username"])]
    }


def llm_node(state: OverAllState) -> OverAllState:
    res = model.invoke(state["messages"])

    return {
        "messages": [res],
        "output": res.content
    }


builder = StateGraph(state_schema=OverAllState)  # type: ignore[arg-type]
builder.add_node("node_a", node_a)  # type: ignore[arg-type]
builder.add_node("llm_node", llm_node)  # type: ignore[arg-type]
builder.add_edge(START, "node_a")
builder.add_edge("node_a", "llm_node")
builder.add_edge("llm_node", END)

graph = builder.compile()
response = graph.invoke({"username": "Modify"})
print(response)
