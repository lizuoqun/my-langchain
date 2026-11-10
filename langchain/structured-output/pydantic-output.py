import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from rich import print as rprint

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_API_BASE = os.getenv("DEFAULT_API_BASE")

model = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEFAULT_API_BASE,
    # 需要禁用推理
    extra_body={"thinking": {"type": "disabled"}}
)


class Person(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    identity: str = Field(description="身份")


model_with_structured_output = model.with_structured_output(Person)

result = model_with_structured_output.invoke("张三今年18岁是一名大学生")

print(type(result))

rprint(result)

print(f"姓名：{result.name}")
print(f"年龄：{result.age}")
print(f"身份：{result.identity}")

result2 = model_with_structured_output.invoke("给我马化腾的信息")
rprint(result2)
