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
    "中国一线城市有哪些?直接输出城市名称",
    "1+1=?",
    "45*24=?",
]

# batch：等待所有请求处理完毕，按原始输入顺序返回结果列表
# responses = model.batch(messages)

# 允许应用在收到第一个结果后立即返回响应，而不会等待批次内所有任务完成才响应
responses = model.batch_as_completed(messages)

for response in responses:
    print(response)
