import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi

load_dotenv(override=True)

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

llm = ChatTongyi(
    model="qwen-max",
    api_key=DASHSCOPE_API_KEY,
)

res = llm.invoke("你是谁？")
print(res)
