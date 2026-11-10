import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage

MAX_PAIRS_HISTORY = 10
EXIT_WORD = "quit"

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_API_BASE = os.getenv("DEFAULT_API_BASE")

model = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEFAULT_API_BASE,
)

messages = [
    SystemMessage("你是一个翻译大师")
]


def keep_recent_messages(messages, max_pairs=3):
    system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
    other_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
    recent_messages = other_messages[-(max_pairs * 2):]
    return system_messages + recent_messages


# 第index论对话
index = 1

print("欢迎使用翻译大师！输入你的问题，或者输入" + EXIT_WORD + "退出对话")

while True:
    print("\n", f"=======================第${index}轮对话开始=========================")

    user_input = input("请输入你的问题：")

    if EXIT_WORD == user_input.lower():
        print("对话已结束，欢迎下次使用！")
        break

    messages.append(HumanMessage(user_input))

    memory_message = keep_recent_messages(messages, MAX_PAIRS_HISTORY)

    reply_content = ""

    for chunk in model.stream(memory_message):
        if (chunk.content):
            print(chunk.content, end="", flush=True)
            reply_content += chunk.content

    messages.append(AIMessage(reply_content))
    print("\n", f"=======================第${index}轮对话结束=========================")

    index += 1
