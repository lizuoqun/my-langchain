import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from rich import print as rprint

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_API_BASE = os.getenv("DEFAULT_API_BASE")

model = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEFAULT_API_BASE
)


@tool(parse_docstring=True)
def get_weather(city: str):
    """
    用来获取城市的天气信息

    Args:
        city:城市名称
    """
    return f"{city}的天气是晴天"


agent = create_agent(model=model, tools=[get_weather])

messages = [
    SystemMessage("你是一个天气助手，可以根据城市名称获取天气信息"),
    HumanMessage("广州的天气咋样")
]

# 流式输出 stream_mode  Literal["values", "updates", "checkpoints", "tasks", "debug", "messages", "custom"]
for chunk in agent.stream(
        {"messages": messages},
        # stream_mode="values"
        stream_mode="updates"
):
    rprint(chunk)
    print("*" * 50)
