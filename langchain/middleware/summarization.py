import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_API_BASE = os.getenv("DEFAULT_API_BASE")

model = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEFAULT_API_BASE,
    profile={
        "max_input_tokens": 128_000
    }
)

agent = create_agent(
    model,
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=[
                ("tokens", 100),
                ("messages", 6),
                ("fraction", 0.001)
            ],
            keep=("messages", 2),
            summary_prompt="对历史消息摘要，消息列表如下\n{messages}"
        )
    ]
)

messages = [
    SystemMessage("你是个非常友好的AI助手"),
    HumanMessage("一句话介绍LangChain"),
    AIMessage(
        "LangChain是一个用于构建和编排由大型语言模型（LLM）驱动的应用开发框架，通过将复杂任务链式组合，让AI能像搭积木一样调用工具、记忆和外部数据。"),
    HumanMessage("推荐其的js版本还是py版本进行开发"),
    AIMessage("如果你是纯后端开发或做AI应用原型，无脑选 Python。如果你是前端/全栈工程师，且不想维护两套语言，选 JS 版。"),
    HumanMessage("推不推荐java版本，一句话回答")
]

result = agent.invoke({"messages": messages})

for msg in result["messages"]:
    msg.pretty_print()
