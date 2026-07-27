# 定义工具
from langchain_core.utils.function_calling import convert_to_openai_tool
from rich import print as rprint


def get_weather(city: str, dt: str = '2026-07-24'):
    """
    获取指定城市的天气信息

    Args:
        city: str
        dt: str = '2026-07-24'

    Returns:
        str: 当前城市的天气信息
    """
    return f"{city}阳光明媚"


rprint(convert_to_openai_tool(get_weather))
