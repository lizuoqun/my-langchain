from langgraph.graph import StateGraph, START, END, MessagesState

from langchain.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek

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


def llm_node(state: MessagesState) -> MessagesState:
    messages = state["messages"]
    response = model.invoke(messages)

    return {
        "messages": [response],
    }


builder = StateGraph(state_schema=MessagesState)  # type: ignore[arg-type]
builder.add_node("llm_node", llm_node)  # type: ignore[arg-type]
builder.add_edge(START, "llm_node")
builder.add_edge("llm_node", END)

graph = builder.compile()

chunks = graph.stream(
    {
        "messages": [HumanMessage(content="你好!")]
    },
    stream_mode=["values", "messages"],
)

for chunk in chunks:
    print(chunk)
