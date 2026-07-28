import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy, AutoStrategy
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from rich import print as rprint

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_API_BASE = os.getenv("DEFAULT_API_BASE")

model = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEFAULT_API_BASE,
    extra_body={"thinking": {"type": "disabled"}}
)


class UserInfo(BaseModel):
    name: str = Field(..., description="用户姓名")
    age: int = Field(..., description="用户年龄")
    city: str = Field(..., description="用户所在城市")


# deepseek-v4-flash 模型不支持 response_format 参数
# agent = create_agent(name='信息提取Agent', model=model, response_format=ProviderStrategy(UserInfo))
agent = create_agent(name='信息提取Agent', model=model, response_format=ToolStrategy(UserInfo))
# agent = create_agent(name='信息提取Agent', model=model, response_format=AutoStrategy(UserInfo))
# agent = create_agent(name='信息提取Agent', model=model, response_format=None)

messages = [
    HumanMessage("用户信息提取，张三今年16岁来自上海")
]

result = agent.invoke({"messages": messages})

rprint(result)
