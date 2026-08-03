import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_API_BASE = os.getenv("DEFAULT_API_BASE")

model = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEFAULT_API_BASE,
    extra_body={"thinking": {"type": "disabled"}}
)

agent = create_agent(
    model=model,
    checkpointer=InMemorySaver()
)

config: RunnableConfig = {
    "configurable": {
        "thread_id": "1"
    }
}

response = agent.invoke({
    "messages": [
        HumanMessage("你好，我叫张三")
    ]
}, config=config)

for msg in response["messages"]:
    msg.pretty_print()

print("\n" + "*" * 100 + "\n")

response_next = agent.invoke({
    "messages": [
        HumanMessage("我叫什么？")
    ]
}, config=config)

for msg in response_next["messages"]:
    msg.pretty_print()
