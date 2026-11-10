import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_API_BASE = os.getenv("DEFAULT_API_BASE")

# 可以不传api_key和base_url，在ChatDeepSeek类中会自动去env文件中找这两个配置
model = ChatDeepSeek(
    model="deepseek-v4-flash",
    # api_key=DEEPSEEK_API_KEY,
    # base_url=DEFAULT_API_BASE
)

# 对于特定的模型供应商有这种对应的类进行初始化
# ChatZhipuAI 智谱
# ChatTongyi 阿里云百炼

# print(model.invoke("你是谁？"))

# 使用OpenAI这个进行兼容写法
openAiModel = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=SecretStr(DEEPSEEK_API_KEY) if DEEPSEEK_API_KEY else None,
    base_url=DEFAULT_API_BASE,
)

print(openAiModel.invoke("你是谁？"))
