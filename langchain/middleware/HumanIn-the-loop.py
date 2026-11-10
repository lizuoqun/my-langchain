import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from rich import print as rprint

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_API_BASE = os.getenv("DEFAULT_API_BASE")

model = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEFAULT_API_BASE
)


@tool
def get_weather(location: str) -> str:
    """
    根据位置获取天气信息.

    Args:
        location:城市、地点名称
    """
    return f"{location}阳光明媚"


@tool
def get_news(location: str) -> str:
    """根据位置获取新闻信息."""
    return f"{location}新闻更新"


@tool
def read_email_tool(email_id: str) -> str:
    """根据邮件id读取邮件内容."""
    return f"邮件内容：{email_id}"


@tool
def send_email_tool(email_id: str) -> str:
    """根据邮件id发送邮件."""
    return f"邮件已发送：{email_id}"


agent = create_agent(
    model,
    tools=[get_weather, get_news, read_email_tool, send_email_tool],
    checkpointer=InMemorySaver(),

    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "get_weather": True,
                "get_news": True,
                "read_email_tool": False,
                "send_email_tool": InterruptOnConfig(
                    allowed_decisions=["approve", "reject"],
                ),
            },
            description_prefix="中断啦"

        )
    ]
)

messages = [
    HumanMessage(
        "帮我查询广州的天气，获取深圳的新闻信息，读取邮件id为1的邮件内容，给邮件id为2的邮箱发送邮件，同时帮我做这四件事"),
]

config = RunnableConfig(configurable={"thread_id": "1"})

result = agent.invoke({"messages": messages}, config=config)

# rprint(result)


weather_decision = {
    "type": "edit",
    "edited_action": {
        "name": "get_weather",
        "args": {"location": "佛山"}
    }
}
news_decision = {
    "type": "approve",
}
send_email_decision = {
    "type": "approve"
}
decisions = {
    "decisions": []
}

interrupts = result.get("__interrupt__", [])
action_requests = interrupts[0].value["action_requests"]

for action_request in action_requests:
    if action_request["name"] == "get_weather":
        decisions["decisions"].append(weather_decision)
    if action_request["name"] == "get_news":
        decisions["decisions"].append(news_decision)
    if action_request["name"] == "send_email_tool":
        decisions["decisions"].append(send_email_decision)

if interrupts:
    # 审批通过
    resumed_response = agent.invoke(
        Command(resume=decisions),
        config=config,  # 必须是同一个 thread_id
    )

    for msg in resumed_response["messages"]:
        msg.pretty_print()
