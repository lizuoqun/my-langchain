from langchain_core.tools import tool, BaseTool


# @tool 装饰器会用函数的 docstring 作为工具的描述信息（description）
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    return city + '阳光明媚'


tool_instance: BaseTool = get_weather  # type: ignore
result = tool_instance.invoke({"city": "长沙"})

print(result)
