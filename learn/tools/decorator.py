# 定义工具
from typing import Literal

from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from rich import print as rprint
from pydantic import BaseModel, Field


class WeatherInput(BaseModel):
    city: str = Field(
        description="城市名称",
        default="北京"
    )
    dt: str = Field(
        description="日期",
        default="2026-07-24"
    )
    unit: Literal["c", "f"] = Field(
        description="温度单位",
        default="c"
    )


weather_schema = {
    'type': 'function',
    'function': {
        'name': 'get_weather_tools',
        'description': '获取指定城市的天气信息',
        'parameters': {
            'properties': {
                'city': {
                    'default': '北京',
                    'description': '城市名称',
                    'type': 'string'
                },
                'dt': {
                    'default': '2026-07-24',
                    'description': '日期',
                    'type': 'string'
                },
                'unit': {
                    'default': 'c',
                    'description': '温度单位',
                    'enum': ['c', 'f'],
                    'type': 'string'
                }
            },
            'type': 'object'
        }
    }
}


# 也可以定义json直接指定
# @tool(args_schema=weather_schema)

@tool("get_weather_tools", description="获取指定城市的天气信息", args_schema=WeatherInput)
def get_weather(city: str, dt: str, unit: Literal["c", "f"]):
    return f"日期：{dt}，{city}阳光明媚，单位：{unit}"


rprint(convert_to_openai_tool(get_weather))
