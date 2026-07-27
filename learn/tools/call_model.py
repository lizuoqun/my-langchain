import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

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


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    return city + '阳光明媚'


messages = [
    HumanMessage("北京天气如何？")
]

model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke(messages)

if response.tool_calls:
    print('AI想调用的工具：', response.tool_calls)
else:
    print('AI的回答：', response.content)

messages.append(response)

tool_calls = response.tool_calls

for tool_call in tool_calls:
    if tool_call["name"] == "get_weather":
        # 返回的是ToolMessage类型消息
        tool_response = get_weather.invoke(tool_call)
        print('调用get_weather后返回的数据类型：', type(tool_response))
        messages.append(tool_response)

for msg in messages:
    print("每一条消息：----", msg)

final_response = model_with_tools.invoke(messages)
print(f"最终的回答: \n{final_response}")
