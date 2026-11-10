import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, StructuredOutputValidationError, \
    MultipleStructuredOutputsError
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


# 自定义错误处理函数
def custom_error_handler(error: Exception) -> str:
    """自定义错误处理器"""
    error_str = str(error)
    print(f"捕获到错误类型：{type(error).__name__}")
    print(f"错误详情：{error_str}")
    if isinstance(error, StructuredOutputValidationError):
        return "数据格式有误，请检查字段是否符合要求。"
    elif isinstance(error, MultipleStructuredOutputsError):
        return "检测到多个响应，请选择最相关的一个进行返回。"
    else:
        return f"Error: {error_str}"


agent = create_agent(name='信息提取Agent', model=model, response_format=ToolStrategy(UserInfo, handle_errors=True))
# handle_errors: bool | str | type[Exception] | tuple[type[Exception], ...]
# handle_errors=True        handle_errors=False
# handle_errors="请检查输入数据"
# handle_errors=(MultipleStructuredOutputsError,StructuredOutputValidationError)  对指定异常类型进行捕获
# handle_errors=custom_error_handler

messages = [
    HumanMessage("用户信息提取，张三今年16岁来自上海")
]

result = agent.invoke({"messages": messages})

rprint(result)
