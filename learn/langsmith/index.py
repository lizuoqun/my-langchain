import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from rich import print as rprint

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_API_BASE = os.getenv("DEFAULT_API_BASE")

model = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEFAULT_API_BASE,
    # 指定可调整参数
    configurable_fields=("model", "model_provider", "temperature", "max_tokens"),
)

config: RunnableConfig = {
    # 在LangSmith中这次运行会显示为
    "run_name": "测试 自定义 run_name",
    # 打上标签便于分类查找
    "tags": ["my_tag1", "my_tag2"],
    "metadata": {
        # 记录用户ID
        "user_id": "测试用户ID",
        # 记录会话ID
        "session_id": "测试会话ID"
    },
    "configurable": {
        "model": "deepseek-v4-flash",
        "model_provider": "deepseek",
        "temperature": 0.7,
        "max_tokens": 100
    }
}

response = model.invoke("1 + 2 = ？", config=config)

rprint(response)
