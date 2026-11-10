import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_API_BASE = os.getenv("DEFAULT_API_BASE")

model = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEFAULT_API_BASE,
)

messages: list[BaseMessage] = [
    SystemMessage("你是一个专业的翻译员"),
    HumanMessage("帮我把您吃了吗翻译成英语")
]

response1 = model.invoke(messages)

print(f"AI的回复：{response1.content}")

messages.append(AIMessage(response1.content))

messages.append(HumanMessage("我刚刚问了你什么问题?"))

response2 = model.invoke(messages)
print(f"AI的回复：{response2.content}")
