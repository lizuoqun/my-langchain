import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_API_BASE = os.getenv("DEFAULT_API_BASE")

model = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEFAULT_API_BASE,
    # 启动推理思考
    extra_body={"thinking": {"type": "enabled"}}
)

messages = [
    HumanMessage(content="一句话告诉我langchain是干嘛的？")
]

response = model.invoke(messages)
print("------------response.content------------")
print(response.content)

print('\n\n\n')

# 在这里可以获取到模型的推理过程
print("------------response.content_blocks------------")
print(response.content_blocks)
