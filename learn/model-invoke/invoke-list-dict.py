import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_API_BASE = os.getenv("DEFAULT_API_BASE")

model = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEFAULT_API_BASE,
)

messages = [
    {"role": "system", "content": "你是一个专业的翻译员"},
    {"role": "user", "content": "帮我把你好世界翻译成英语"}
]

response1 = model.invoke(messages)

print(f"AI的回复：{response1.content}")

messages.append({"role": "assistant", "content": response1.content})

messages.append({"role": "user", "content": "我刚刚问了你什么问题？"})

response2 = model.invoke(messages)
print(f"AI的回复：{response2.content}")
