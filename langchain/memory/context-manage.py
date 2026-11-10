import os
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, after_model
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_API_BASE = os.getenv("DEFAULT_API_BASE")

model = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEFAULT_API_BASE,
    extra_body={"thinking": {"type": "disabled"}}
)


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    messages = state["messages"]

    if len(messages) <= 3:
        return None

    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }


@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    messages = state["messages"]
    if len(messages) > 5:
        to_delete = len(messages) - 5
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:to_delete]]}
    return None


agent = create_agent(
    model=model,
    middleware=[trim_messages],
    checkpointer=InMemorySaver()
)

config: RunnableConfig = {
    "configurable": {
        "thread_id": "1"
    }
}

agent.invoke({"messages": [HumanMessage("你好，我叫张三")]}, config=config)
agent.invoke({"messages": [HumanMessage("bin是不是垃圾桶？")]}, config=config)
agent.invoke({"messages": [HumanMessage("1+1=？")]}, config=config)
final_response = agent.invoke({"messages": [HumanMessage("你好，我叫什么？")]}, config=config)

for msg in final_response["messages"]:
    msg.pretty_print()
