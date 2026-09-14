from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.messages import HumanMessage
from rich import print as rprint

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
    output: str


def llm_node(state: OverAllState) -> dict[str, list[AIMessage]]:
    messages = state["messages"]
    ai_msg = model.invoke(messages)
    return {
        "messages": [ai_msg],
    }


def output_node(state: OverAllState) -> dict[str, str]:
    return {
        "output": str(state["messages"][-1].content)
    }


builder = StateGraph(state_schema=OverAllState)  # type: ignore[arg-type]
builder.add_node("llm_node", llm_node)  # type: ignore[arg-type]
builder.add_node("output_node", output_node)  # type: ignore[arg-type]
builder.add_edge(START, "llm_node")
builder.add_edge("llm_node", "output_node")
builder.add_edge("output_node", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config: RunnableConfig = {
    "configurable": {
        "thread_id": "thread-001"
    }
}

graph.invoke({"messages": HumanMessage("我叫Modify")}, config=config)

history_checkpoints = list(graph.get_state_history(config=config))
# rprint(history_checkpoints)

new_checkpointer = graph.get_state(config=config)
rprint(new_checkpointer)
