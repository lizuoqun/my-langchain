import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_deepseek import ChatDeepSeek

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_API_BASE = os.getenv("DEFAULT_API_BASE")

model = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEFAULT_API_BASE,
)

# 查看模型的配置信息
print(model.profile)

# 不推荐访问实例上的“model_fields”属性。相反，您应该从模型类访问此属性
# print(model.model_fields)
# print(model.model_fields.keys())
print(ChatDeepSeek.model_fields)
print(ChatDeepSeek.model_fields.keys())

# 美化输出
# response = model.invoke("帮我把你好世界翻译成英语")
# response.pretty_print()
