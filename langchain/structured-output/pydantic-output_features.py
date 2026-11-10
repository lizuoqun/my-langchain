import os
from enum import Enum
from typing import Optional, Literal, List

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from rich import print as rprint

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_API_BASE = os.getenv("DEFAULT_API_BASE")

model = init_chat_model(
    model="deepseek:deepseek-v4-pro",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEFAULT_API_BASE,
    # 需要禁用推理
    extra_body={"thinking": {"type": "disabled"}}
)


class Adults(str, Enum):
    ADULT = "成年人"
    CHILD = "未成年人"


class Person(BaseModel):
    name: str = Field(description="姓名")
    # Optional可选，没有的话会使用None，default指定默认值
    age: Optional[int] = Field(default=20, description="年龄")
    identity: str = Field(description="身份")
    isAdult: Adults = Field(description="是否成年")
    isAdult2: Literal["成年人", "未成年人"] = Field(description="是否成年")


class PersonList(BaseModel):
    people: List[Person]


model_with_structured_output = model.with_structured_output(Person)
result = model_with_structured_output.invoke("张三是一名大学生")
rprint(result)

print("\n\n")

model_with_structured_output_list = model.with_structured_output(PersonList)
result_list = model_with_structured_output_list.invoke("张三，李四是一名邮递员，王五是一名34岁的设计师")
rprint(result_list)

print("\n\n")


# -----------------------------------------嵌套结构-----------------------------------------------
class Actor(BaseModel):
    """演员信息"""
    name: str = Field(description="演员姓名", min_length=2, max_length=100)
    role: str = Field(description="饰演的角色")


class Movie(BaseModel):
    """电影信息"""
    title: str = Field(description="电影标题")
    year: int = Field(description="上映年份")
    director: str = Field(description="导演")
    cast: Optional[List[Actor]] = Field(default=None, description="演员列表")
    rating: Optional[float] = Field(default=None, description="评分")


model_with_structured_output_movie = model.with_structured_output(Movie)
result_movie = model_with_structured_output_movie.invoke("请介绍电影《肖申克的救赎》，需要上映年份、导演、演员列表和评分")
rprint(result_movie)
print("\n\n")


# -----------------------------------------数据条件限制-----------------------------------------------

class Product(BaseModel):
    """产品信息（严格验证）"""
    name: str = Field(description="产品名称（字符串类型）", min_length=2)
    price: float = Field(description="价格，数字类型", gt=0)
    stock: int = Field(description="库存，整数类型", ge=0)


model_with_structured_output_product = model.with_structured_output(Product)
response = model_with_structured_output_product.invoke("华为mate 80 promax 价格是7999，当前库存100")
print(response)
