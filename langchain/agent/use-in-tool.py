import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_tavily import TavilySearch
from rich import print as rprint

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_API_BASE = os.getenv("DEFAULT_API_BASE")

model = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEFAULT_API_BASE
)

web_search = TavilySearch(
    tavily_api_key=os.getenv("TAVILY_API_KEY"),
    max_results=2
)

# rprint(web_search.invoke("2026年MSI冠军是那个战队"))

agent = create_agent(name='联网搜索Agent', model=model, tools=[web_search], system_prompt="")

messages = [
    SystemMessage("你是一个联网查询助手"),
    HumanMessage("2018年S冠军是那个战队")
]

result = agent.invoke({"messages": messages})

print(result['messages'][-1].content)

rprint(result)
