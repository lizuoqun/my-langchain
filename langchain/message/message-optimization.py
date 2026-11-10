import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_API_BASE = os.getenv("DEFAULT_API_BASE")

model = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEFAULT_API_BASE,
)

# 假设这是和AI进行了很多的一段对话
messages = [
    SystemMessage("你是一个翻译大师"),
    HumanMessage("你好"),
    AIMessage("Hello"),
    HumanMessage("你叫什么名字？"),
    AIMessage("What's your name?"),
    HumanMessage("我叫张三"),
    AIMessage("My name is Zhang San"),
    HumanMessage("你多大了"),
    AIMessage("I'm an AI, so I don't have an age."),
    HumanMessage("我今年18岁"),
    AIMessage("I'm 18 years old")
]

"""
message：已经进行过的对话
max_pairs：最多保留的对话对数
"""


def keep_recent_messages(messages, max_pairs=3):
    system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
    other_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
    recent_messages = other_messages[-(max_pairs * 2):]
    return system_messages + recent_messages


optimized_message = keep_recent_messages(messages, 1)

print(optimized_message)
